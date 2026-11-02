"""E2E (step 10c-fix-8): a real bot "Живые игроки" snapshot updates GameState.

Wires the real :class:`~features.scanner.scanner.MessageScanner` to a migrated
in-memory SQLite database. Only the two external boundaries are faked:

- Telegram: a subscription-only stand-in (no network);
- the analyzer: a counting double (the snapshot must not trigger a flush).

The scenario sends one realistic multi-line bot message through
``scanner._on_event`` and asserts that ``AliveListEvent`` is applied: the two
listed players stay alive, the absent one dies (A2-1), and the bot-reported
role composition / headcount land on ``GameState``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from domain.events import AliveListEvent, AnyEvent
from domain.game_state import GameState
from domain.roles import RoleRegistry, load_roles
from features.analyzer.analyzer import AnalysisResult, GameAnalyzer
from features.scanner.scanner import MessageScanner
from features.scanner.trigger_policy import TriggerMode, TriggerPolicy
from infra.db.connection import close, connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.games import GamesRepository
from infra.llm.schemas import AnalysisResponse
from infra.telegram.client import TelegramClientWrapper

ROOT = Path(__file__).resolve().parents[2]
ROLES_PATH = ROOT / "configs" / "roles.yaml"

TS = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)
CHAT_ID = -1001234567890
BOT_ID = 777

#: A, B, C — the third one is absent from the bot's alive list.
PLAYERS: tuple[tuple[int, str], ...] = (
    (1, "user_1"),
    (2, "user_2"),
    (3, "user_3"),
)

ALIVE_LIST_TEXT = """Живые игроки:
1. user_1
2. user_2

Из них: 👨🏼‍💼 Адвокат, 🤵🏼 Мафия - 1

Всего: 2"""


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeTelegramWrapper:
    """Subscription-only stand-in for :class:`TelegramClientWrapper`."""

    def __init__(self) -> None:
        self.handler: Callable[[Any], Any] | None = None
        self.chat_id: int | None = None
        self.add_calls: int = 0
        self.remove_calls: int = 0

    def add_message_handler(
        self,
        handler: Callable[[Any], Any],
        *,
        chat_id: int,
        bot_only: bool = False,
        deduplicate: bool = True,
    ) -> None:
        """Store the handler registered for ``chat_id``."""
        self.handler = handler
        self.chat_id = chat_id
        self.add_calls += 1

    async def remove_all_handlers(self) -> None:
        """Detach every handler."""
        self.handler = None
        self.remove_calls += 1


class CountingAnalyzer:
    """Analyzer double; in QUEUE mode the snapshot must never flush."""

    def __init__(self) -> None:
        self.calls: int = 0

    async def analyze(
        self,
        game_state: GameState,
        *,
        persist: bool = True,
        previous_predictions: Any = None,
    ) -> AnalysisResult:
        """Count the call and report a successful empty analysis."""
        self.calls += 1
        return AnalysisResult(
            response=AnalysisResponse(analysis=[], game_read="ok", reasoning="ok"),
            error=None,
            duration_sec=0.001,
            reasoning_chars=0,
            warnings=(),
        )


def bot_event(text: str, *, second: int = 0) -> SimpleNamespace:
    """Build a Telegram-shaped event announced by the game bot."""
    return SimpleNamespace(
        sender_id=BOT_ID,
        sender=SimpleNamespace(first_name="Мафия Бот"),
        message=SimpleNamespace(text=text),
        date=TS + timedelta(seconds=second),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> RoleRegistry:
    """Load the configured role definitions once per module."""
    return load_roles(ROLES_PATH)


@pytest.fixture
def connection() -> Iterator[sqlite3.Connection]:
    """Provide a migrated in-memory database."""
    database = connect(":memory:")
    apply_migrations(database)
    try:
        yield database
    finally:
        close(database)


# ---------------------------------------------------------------------------
# 1. Real alive-list bot message → AliveListEvent → GameState
# ---------------------------------------------------------------------------


async def test_e2e_real_alive_list(
    connection: sqlite3.Connection,
    registry: RoleRegistry,
) -> None:
    """A real bot snapshot flows through the scanner into GameState (A2-1)."""
    game_id = GamesRepository(connection).create_game(chat_id=CHAT_ID, player_count=len(PLAYERS))
    game_state = GameState(game_id=game_id, registry=registry)
    for player_id, display_name in PLAYERS:
        game_state.register_player(player_id, display_name)

    applied: list[AnyEvent] = []

    async def on_event(event: AnyEvent) -> None:
        applied.append(event)

    telegram = FakeTelegramWrapper()
    analyzer = CountingAnalyzer()
    scanner = MessageScanner(
        telegram=cast(TelegramClientWrapper, telegram),
        game_state=game_state,
        analyzer=cast(GameAnalyzer, analyzer),
        registry=registry,
        trigger_policy=TriggerPolicy(mode=TriggerMode.QUEUE),
        chat_id=CHAT_ID,
        bot_id=BOT_ID,
        on_event=on_event,
    )

    await scanner.start()
    try:
        await scanner._on_event(bot_event(ALIVE_LIST_TEXT))
    finally:
        await scanner.shutdown()

    # --- An AliveListEvent was produced, applied and emitted ---------------
    assert len(applied) == 1
    assert isinstance(applied[0], AliveListEvent)
    event = applied[0]
    assert isinstance(event, AliveListEvent)
    assert event.payload.alive_player_names == ["user_1", "user_2"]
    assert game_state.events_by_kind("alive_list") == (event,)
    assert telegram.chat_id == CHAT_ID
    assert telegram.add_calls == 1
    assert telegram.remove_calls == 1

    # --- A2-1: listed players live, the absent one is dead ----------------
    assert game_state.player(1).alive is True
    assert game_state.player(2).alive is True
    assert game_state.player(3).alive is False
    assert len(game_state.alive_players()) == 2
    assert len(game_state.players()) == 3  # no unknown names registered

    # --- Bot-reported composition reached GameState -----------------------
    assert game_state.role_count is not None
    assert game_state.role_count == {"Адвокат": 1, "Мафия": 1}
    assert game_state.total_players == 2

    # QUEUE mode never flushes on its own: no LLM call was made.
    assert analyzer.calls == 0
    assert scanner.buffer_size == 1
