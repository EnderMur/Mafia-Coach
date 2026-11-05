"""End-to-end scanner tests: Telegram events → GameState → analyzer → database.

These tests wire the real :class:`~features.scanner.scanner.MessageScanner`, the
real :class:`~features.analyzer.analyzer.GameAnalyzer` and a migrated in-memory
SQLite database together. Only the two external boundaries are faked:

- Telegram: a fake wrapper that stores the handler instead of talking to the
  network (the real one is covered by ``tests/unit/infra/telegram``);
- LM Studio: a fake LLM client, so no model is ever involved.

The scenario mirrors one real club game: registration, night, day, a vote, a
lynch and the role reveal of the hanged player.

Scope note: the scanner applies events to ``GameState`` and persists
**predictions**. Raw event persistence belongs to the orchestrator (step 10), so
this file asserts that the scanner itself writes no rows into ``events`` —
keeping the boundary explicit instead of accidental.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jinja2
import pytest

from domain.events import MessageEvent, PhaseChangeEvent, PhaseChangePayload
from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles
from features.analyzer.analyzer import AnalysisResult, GameAnalyzer
from features.scanner.scanner import MessageScanner
from features.scanner.trigger_policy import TriggerMode, TriggerPolicy
from infra.db.connection import close, connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.events import EventsRepository
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository
from infra.llm.client import LLMClient, ReasoningSink
from infra.llm.schemas import AnalysisResponse, PlayerPrediction
from infra.telegram.client import TelegramClientWrapper

ROOT = Path(__file__).resolve().parents[2]
ROLES_PATH = ROOT / "configs" / "roles.yaml"
PROMPTS_DIR = ROOT / "infra" / "llm" / "prompts"

TS = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)
CHAT_ID = -1001234567890
BOT_ID = 777
VASYA = 100
PETYA = 101
LENA = 102
DEBOUNCE = 0.05
DEBOUNCE_WAIT = 0.15
PLAYERS: tuple[tuple[int, str], ...] = ((VASYA, "Вася"), (PETYA, "Петя"), (LENA, "Лена"))


# ---------------------------------------------------------------------------
# Test doubles (duplicated on purpose: integration tests stay self-contained)
# ---------------------------------------------------------------------------


class FakeLLMClient(LLMClient):
    """Streaming LLM double that always predicts "Вася is Мафия"."""

    def __init__(self, response: AnalysisResponse | None = None) -> None:
        self.calls: int = 0
        self._response = response or AnalysisResponse(
            analysis=[
                PlayerPrediction(
                    player_id=VASYA,
                    player_name="Вася",
                    suspected_role="Мафия",
                    confidence=0.7,
                    reason="Давит на мирных и путается в показаниях.",
                    reasons=["давит на мирных", "путается в показаниях"],
                )
            ],
            game_read="Вася ведёт себя как мафия.",
            reasoning="Вася трижды сменил версию и давил на Петю.",
        )

    async def healthcheck(self) -> None:
        """Implement the no-op fake health check."""

    async def analyze(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        silent_player_ids: frozenset[int] = frozenset(),
    ) -> AnalysisResponse:
        """Serve non-streaming calls from the same canned response."""
        return await self.analyze_stream(
            messages=messages,
            known_player_ids=known_player_ids,
            role_names_ru=role_names_ru,
            role_alias_resolver=role_alias_resolver,
            silent_player_ids=silent_player_ids,
        )

    async def analyze_stream(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        on_reasoning_chunk: ReasoningSink | None = None,
        log_reasoning_chunks: bool = True,
        silent_player_ids: frozenset[int] = frozenset(),
    ) -> AnalysisResponse:
        """Count the call, emit one reasoning chunk, return the prediction."""
        self.calls += 1
        assert messages, "prompt messages must not be empty"
        if on_reasoning_chunk is not None:
            on_reasoning_chunk("Вася нервничает")
        return self._response

    async def aclose(self) -> None:
        """Implement the no-op fake close operation."""


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
    """Analyzer double that only counts flushes (batch-debounce scenario)."""

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
        await asyncio.sleep(0)
        return AnalysisResult(
            response=AnalysisResponse(analysis=[], game_read="ok", reasoning="ok"),
            error=None,
            duration_sec=0.001,
            reasoning_chars=0,
            warnings=(),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> RoleRegistry:
    """Load the configured role definitions once per module."""
    return load_roles(ROLES_PATH)


@pytest.fixture
def jinja_env() -> jinja2.Environment:
    """Create a Jinja environment over the real prompt templates."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False,
    )


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
# Telegram-shaped events
# ---------------------------------------------------------------------------


@dataclass
class FakeMessage:
    """Message body as seen through a Telethon event."""

    text: str | None


@dataclass
class FakeSender:
    """Sender entity subset used by the scanner."""

    first_name: str | None = None
    last_name: str | None = None


def player_event(player_id: int, name: str, text: str, *, second: int) -> SimpleNamespace:
    """Build an event for a speaking player."""
    return SimpleNamespace(
        sender_id=player_id,
        sender=FakeSender(first_name=name),
        message=FakeMessage(text=text),
        date=TS + timedelta(seconds=second),
    )


def bot_event(text: str, *, second: int) -> SimpleNamespace:
    """Build an event announced by the game bot."""
    return SimpleNamespace(
        sender_id=BOT_ID,
        sender=FakeSender(first_name="Мафия Бот"),
        message=FakeMessage(text=text),
        date=TS + timedelta(seconds=second),
    )


def game_script() -> list[SimpleNamespace]:
    """Return exactly 20 realistic events (15 player messages, 5 bot messages)."""
    return [
        player_event(VASYA, "Вася", "Всем привет, я мирный житель", second=0),
        player_event(PETYA, "Петя", "Я тоже мирный, играем", second=1),
        player_event(LENA, "Лена", "Я готова, погнали", second=2),
        player_event(VASYA, "Вася", "Мне кажется, Петя мафия", second=3),
        player_event(PETYA, "Петя", "Я мирный, Вася давит без причин", second=4),
        player_event(LENA, "Лена", "Не ссорьтесь, давайте слушать", second=5),
        bot_event("🌙 Наступает ночь", second=6),
        player_event(LENA, "Лена", "Ночью было тихо", second=7),
        player_event(VASYA, "Вася", "Кто-то не проснулся?", second=8),
        bot_event("🌝 День 1", second=9),
        player_event(PETYA, "Петя", "Кого вешаем сегодня?", second=10),
        player_event(LENA, "Лена", "Я против Васи, он агрессивный", second=11),
        bot_event("Вася голосовал за: Петя", second=12),
        bot_event("Сегодня повешен Вася", second=13),
        bot_event("Вася был Мафией", second=14),
        player_event(PETYA, "Петя", "Ого, Вася был мафией!", second=15),
        player_event(LENA, "Лена", "Я так и знала, он давил", second=16),
        player_event(VASYA, "Вася", "Я не мафия, это подстава", second=17),
        player_event(PETYA, "Петя", "Кто следующий по подозрению?", second=18),
        player_event(LENA, "Лена", "Давайте думать про Дона", second=19),
    ]


def make_game(database: sqlite3.Connection) -> int:
    """Insert one game with three players and return its id."""
    game_id = GamesRepository(database).create_game(chat_id=CHAT_ID, player_count=len(PLAYERS))
    players_repository = PlayersRepository(database)
    for player_id, name in PLAYERS:
        players_repository.upsert_player(player_id=player_id, username=None, display_name=name)
    return game_id


# ---------------------------------------------------------------------------
# 1. Full cycle with the real analyzer and a real database
# ---------------------------------------------------------------------------


async def test_e2e_scanner_full_cycle(
    connection: sqlite3.Connection,
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
) -> None:
    """1. A whole game flows from fake Telegram events into predictions in SQLite."""
    game_id = make_game(connection)
    predictions_repo = PredictionsRepository(connection)
    llm_client = FakeLLMClient()
    analyzer = GameAnalyzer(
        llm_client=llm_client,
        predictions_repo=predictions_repo,
        registry=registry,
        jinja_env=jinja_env,
    )
    game_state = GameState(game_id=game_id, registry=registry)
    telegram = FakeTelegramWrapper()
    scanner = MessageScanner(
        telegram=cast(TelegramClientWrapper, telegram),
        game_state=game_state,
        analyzer=analyzer,
        registry=registry,
        trigger_policy=TriggerPolicy(mode=TriggerMode.EVERY_MESSAGE),
        chat_id=CHAT_ID,
        bot_id=BOT_ID,
    )

    await scanner.start()
    script = game_script()
    for event in script:
        await scanner._on_event(event)
    await scanner.shutdown()

    # --- Scanner lifecycle -------------------------------------------------
    assert telegram.chat_id == CHAT_ID
    assert telegram.add_calls == 1
    assert scanner.is_started is False
    assert telegram.remove_calls == 1

    # --- Game state --------------------------------------------------------
    assert game_state.current_phase() is Phase.DAY
    assert game_state.current_day() == 1
    deaths = game_state.events_by_kind("death")
    assert len(deaths) == 1
    assert game_state.player(VASYA).alive is False
    assert game_state.player(PETYA).alive is True
    revealed_player = game_state.player(VASYA)
    assert revealed_player.known_role is not None
    assert revealed_player.known_role.id == "mafia"

    # --- Identity map ------------------------------------------------------
    assert scanner.name_to_id == {"Вася": VASYA, "Петя": PETYA, "Лена": LENA}

    # --- Event kinds applied through the pipeline -------------------------
    kinds = {event.kind for event in game_state.events()}
    assert kinds == {"message", "death", "vote", "role_reveal", "phase_change"}
    assert len(game_state.events_by_kind("message")) == 15
    assert all(isinstance(event, MessageEvent) for event in game_state.events_by_kind("message"))

    # --- Persistence: the analyzer wrote predictions ----------------------
    assert llm_client.calls > 0
    predictions = predictions_repo.list_predictions(game_id)
    assert len(predictions) == 1
    # The fake keeps predicting Vasya after his death; those predictions are dropped.
    history_count = connection.execute("SELECT COUNT(*) FROM predictions_history").fetchone()[0]
    assert 0 < history_count < llm_client.calls
    assert {row["player_id"] for row in predictions} == {VASYA}
    assert {row["predicted_role"] for row in predictions} == {"Мафия"}

    # --- Boundary: raw events are persisted by the orchestrator, not here --
    assert EventsRepository(connection).list_events(game_id) == []
    assert scanner.buffer_size == 0


# ---------------------------------------------------------------------------
# 2. Batch debounce mode
# ---------------------------------------------------------------------------


async def test_e2e_scanner_batch_debounce(registry: RoleRegistry) -> None:
    """2. In ``batch_debounce`` a quiet burst is analyzed exactly once."""
    game_state = GameState(game_id=1, registry=registry)
    game_state.apply(
        PhaseChangeEvent(
            ts=TS,
            game_id=1,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
            ),
        )
    )
    analyzer = CountingAnalyzer()
    telegram = FakeTelegramWrapper()
    scanner = MessageScanner(
        telegram=cast(TelegramClientWrapper, telegram),
        game_state=game_state,
        analyzer=cast(GameAnalyzer, analyzer),
        registry=registry,
        trigger_policy=TriggerPolicy(
            mode=TriggerMode.BATCH_DEBOUNCE,
            batch_debounce_sec=DEBOUNCE,
        ),
        chat_id=CHAT_ID,
        bot_id=BOT_ID,
    )

    await scanner.start()
    await scanner._on_event(player_event(VASYA, "Вася", "Первая реплика за минуту", second=0))
    await scanner._on_event(player_event(PETYA, "Петя", "Вторая реплика за минуту", second=1))
    await scanner._on_event(player_event(LENA, "Лена", "Третья реплика за минуту", second=2))

    assert analyzer.calls == 0
    assert scanner.buffer_size == 3

    await asyncio.sleep(DEBOUNCE_WAIT)

    assert analyzer.calls == 1
    assert scanner.buffer_size == 0

    await scanner.shutdown()

    assert scanner.is_started is False
    assert telegram.remove_calls == 1
