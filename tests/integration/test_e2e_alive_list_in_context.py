"""E2E (step 10c-fix-9): AliveListEvent flows from the scanner into the LLM context.

Wires the real :class:`~features.scanner.scanner.MessageScanner` to a migrated
in-memory SQLite database. Only Telegram is faked (a subscription-only stand-in,
no network).

The scenario sends one realistic bot "Живые игроки" message through
``scanner._on_event`` so the parser produces a real ``AliveListEvent`` with
``role_count={"Мафия": 5}``. The event is applied to ``GameState``, then
:func:`~features.analyzer.context_builder.build_context` and
:func:`~features.analyzer.context_builder.to_llm_messages` render it — the user
prompt must contain the role composition ("Мафия" and "5").
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jinja2
import pytest

from domain.events import AliveListEvent, AnyEvent
from domain.game_state import GameState
from domain.roles import RoleRegistry, load_roles
from features.analyzer.analyzer import AnalysisResult, GameAnalyzer
from features.analyzer.context_builder import build_context, to_llm_messages
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

PLAYERS: tuple[tuple[int, str], ...] = (
    (1, "user_1"),
    (2, "user_2"),
    (3, "user_3"),
    (4, "user_4"),
    (5, "user_5"),
)

ALIVE_LIST_TEXT = """Живые игроки:
1. user_1
2. user_2
3. user_3
4. user_4
5. user_5

Из них: 🤵🏼 Мафия - 5

Всего: 5"""


# ---------------------------------------------------------------------------#
# Test doubles                                                               #
# ---------------------------------------------------------------------------#


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
    """Analyzer double; the snapshot must not trigger a flush in QUEUE mode."""

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


# ---------------------------------------------------------------------------#
# Fixtures                                                                   #
# ---------------------------------------------------------------------------#


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


# ---------------------------------------------------------------------------#
# 1. Real alive-list bot message → AliveListEvent → GameState → LLM context  #
# ---------------------------------------------------------------------------#


async def test_e2e_alive_list_flows_to_llm_messages(
    connection: sqlite3.Connection,
    registry: RoleRegistry,
) -> None:
    """The bot's role composition reaches the rendered LLM prompt."""
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

    # --- The scanner really produced and applied an AliveListEvent -------
    assert len(applied) == 1
    assert isinstance(applied[0], AliveListEvent)
    assert game_state.role_count == {"Мафия": 5}
    assert game_state.total_players == 5

    # --- GameState → LLM context -----------------------------------------
    context = build_context(game_state=game_state, registry=registry)
    assert context.role_count == {"Мафия": 5}
    assert context.total_players == 5

    # --- Context → rendered LLM messages ---------------------------------
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("infra/llm/prompts"), autoescape=False
    )
    messages = to_llm_messages(context, registry=registry, jinja_env=jinja_env)

    assert len(messages) == 2
    assert [message["role"] for message in messages] == ["system", "user"]
    user_content = messages[1]["content"]
    assert "Мафия" in user_content
    assert "5" in user_content

    # QUEUE mode never flushes on its own: no LLM call was made.
    assert analyzer.calls == 0
