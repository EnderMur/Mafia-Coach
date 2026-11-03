"""End-to-end backfill test (step 11a): history replay from ``GAME_START``.

Wires the real :class:`~features.scanner.scanner.MessageScanner`, the real
:class:`~features.analyzer.analyzer.GameAnalyzer`, and a migrated in-memory
SQLite database together. ``FakeTelegram.list_recent_messages`` returns 20
messages in Telethon's newest-first order with ``GAME_START`` in the middle;
``scanner.backfill()`` ignores the previous game, replays events after
``GAME_START``, and invokes the analyzer on phase events (plus a final flush
for trailing speech).
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

from domain.events import MessageEvent, VoteEvent
from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles
from features.analyzer.analyzer import GameAnalyzer
from features.scanner.scanner import MessageScanner
from features.scanner.trigger_policy import TriggerMode, TriggerPolicy
from infra.db.connection import close, connect
from infra.db.migrations.runner import apply_migrations
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
OLD_PLAYER = 99
VASYA = 100
PETYA = 101
LENA = 102


@dataclass
class FakeSender:
    """Sender entity subset used by the scanner."""

    first_name: str | None = None
    last_name: str | None = None


class FakeLLMClient(LLMClient):
    """Streaming LLM double that predicts ``Вася`` is ``Мафия``."""

    def __init__(self) -> None:
        self.calls: int = 0
        self._response = AnalysisResponse(
            analysis=[
                PlayerPrediction(
                    player_id=VASYA,
                    player_name="Вася",
                    suspected_role="Мафия",
                    confidence=0.75,
                    reasons=["активно давит в первый день"],
                )
            ],
            game_read="Вася агрессивно ищет жертву.",
            reasoning="По репликам после ночи Вася переводит подозрения на Петю.",
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
    ) -> AnalysisResponse:
        """Serve non-streaming calls from the same canned response."""
        return await self.analyze_stream(
            messages=messages,
            known_player_ids=known_player_ids,
            role_names_ru=role_names_ru,
            role_alias_resolver=role_alias_resolver,
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
    ) -> AnalysisResponse:
        """Count the call, emit one reasoning chunk, and return predictions."""
        self.calls += 1
        assert messages, "prompt messages must not be empty"
        if on_reasoning_chunk is not None:
            on_reasoning_chunk("Анализирую историю с начала партии")
        return self._response

    async def aclose(self) -> None:
        """Implement the no-op fake close operation."""


class FakeTelegram:
    """Read-only Telegram stand-in returning canned history newest-first."""

    def __init__(self, history_newest_first: list[SimpleNamespace]) -> None:
        self._history = history_newest_first
        self.handler: Callable[[Any], Any] | None = None
        self.chat_id: int | None = None
        self.list_calls: list[tuple[int, int]] = []

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

    async def remove_all_handlers(self) -> None:
        """Detach every handler."""
        self.handler = None

    async def list_recent_messages(
        self,
        *,
        chat_id: int,
        limit: int = 50,
    ) -> list[Any]:
        """Return up to ``limit`` messages in Telethon's newest-first order."""
        self.list_calls.append((chat_id, limit))
        return list(self._history[:limit])


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
    """Provide a migrated in-memory SQLite database."""
    database = connect(":memory:")
    apply_migrations(database)
    try:
        yield database
    finally:
        close(database)


def _player_msg(player_id: int, name: str, text: str, *, second: int) -> SimpleNamespace:
    return SimpleNamespace(
        sender_id=player_id,
        sender=FakeSender(first_name=name),
        text=text,
        date=TS + timedelta(seconds=second),
    )


def _bot_msg(text: str, *, second: int) -> SimpleNamespace:
    return SimpleNamespace(
        sender_id=BOT_ID,
        sender=FakeSender(first_name="Мафия Бот"),
        text=text,
        date=TS + timedelta(seconds=second),
    )


def build_20_messages() -> list[SimpleNamespace]:
    """Build 20 chronological messages with ``GAME_START`` in the middle (index 9)."""
    messages = [
        # --- Old finished game (indices 0..8, must be ignored) ------------
        _player_msg(OLD_PLAYER, "Старожил", "Прошлая игра уже подходит к концу", second=0),
        _player_msg(VASYA, "Вася", "В прошлой игре я точно мирный", second=1),
        _bot_msg("🌙 Наступает ночь 3", second=2),
        _bot_msg("🌝 День 3", second=3),
        _player_msg(OLD_PLAYER, "Старожил", "Голосуем внимательно в финале", second=4),
        _bot_msg("Старожил голосовал за: Вася", second=5),
        _bot_msg("Сегодня повешен Вася", second=6),
        _bot_msg("Вася был Мафией", second=7),
        _player_msg(OLD_PLAYER, "Старожил", "Отличная была партия, всем спасибо!", second=8),
        # --- Current game start marker (index 9) --------------------------
        _bot_msg("Город ветров начинает свою игру над жителями!", second=9),
        # --- Current ongoing game (indices 10..19) ------------------------
        _player_msg(VASYA, "Вася", "Новая партия, всем удачи!", second=10),
        _player_msg(PETYA, "Петя", "Всем привет, играем спокойно", second=11),
        _player_msg(LENA, "Лена", "Я готова искать мафию", second=12),
        _bot_msg("🌙 Наступает ночь 1", second=13),
        _bot_msg(
            "Живые игроки:\n1. Вася\n2. Петя\n3. Лена\n\n"
            "Из них: 🤵🏼 Мафия - 1, 👨🏼 Мирный житель - 2\n\nВсего: 3",
            second=14,
        ),
        _bot_msg("🌝 День 1", second=15),
        _player_msg(VASYA, "Вася", "Петя слишком тихо сидел перед ночью", second=16),
        _player_msg(PETYA, "Петя", "Вася сразу нападает без аргументов", second=17),
        _bot_msg("Вася голосовал за: Петя", second=18),
        _player_msg(LENA, "Лена", "Подождите, давайте разберём голоса", second=19),
    ]
    assert len(messages) == 20
    return messages


async def test_e2e_backfill_full_cycle(
    connection: sqlite3.Connection,
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
) -> None:
    """5. Backfill loads 20 messages, keeps only post-marker events, and runs analyzer."""
    game_id = GamesRepository(connection).create_game(chat_id=CHAT_ID, player_count=0)
    players_repo = PlayersRepository(connection)
    predictions_repo = PredictionsRepository(connection)
    llm_client = FakeLLMClient()
    analyzer = GameAnalyzer(
        llm_client=llm_client,
        predictions_repo=predictions_repo,
        registry=registry,
        jinja_env=jinja_env,
    )
    game_state = GameState(game_id=game_id, registry=registry)
    chronological = build_20_messages()
    telegram = FakeTelegram(list(reversed(chronological)))

    async def persist_player(player_id: int, display_name: str) -> None:
        await asyncio.to_thread(
            players_repo.upsert_player,
            player_id=player_id,
            username=display_name,
            display_name=display_name,
        )

    async def flush_complete() -> None:
        pass

    scanner = MessageScanner(
        telegram=cast(TelegramClientWrapper, telegram),
        game_state=game_state,
        analyzer=analyzer,
        registry=registry,
        trigger_policy=TriggerPolicy(
            mode=TriggerMode.BATCH_DEBOUNCE,
            batch_debounce_sec=10.0,
            batch_max_messages=50,
        ),
        chat_id=CHAT_ID,
        bot_id=BOT_ID,
        on_player_registered=persist_player,
        on_flush_complete=flush_complete,
    )

    await scanner.start()
    applied = await scanner.backfill()
    await scanner.shutdown()

    assert telegram.list_calls == [(CHAT_ID, 5000)]
    assert applied == 10

    # Old game (before GAME_START) was completely ignored.
    assert OLD_PLAYER not in {player.player_id for player in game_state.players()}
    assert "Старожил" not in scanner.name_to_id
    assert game_state.events_by_kind("death") == ()
    assert game_state.events_by_kind("role_reveal") == ()
    assert game_state.player(VASYA).alive is True

    # Current game events after GAME_START are applied in chronological order.
    assert game_state.current_phase() is Phase.DAY
    assert game_state.current_day() == 1
    assert game_state.total_players == 3
    assert {player.player_id for player in game_state.players()} == {VASYA, PETYA, LENA}

    applied_messages = game_state.events_by_kind("message")
    assert len(applied_messages) == 6
    assert all(isinstance(event, MessageEvent) for event in applied_messages)

    votes = game_state.events_by_kind("vote")
    assert len(votes) == 1
    assert isinstance(votes[0], VoteEvent)
    assert votes[0].payload.voter_id == VASYA
    assert votes[0].payload.target_id == PETYA

    # Replay suppresses phase-triggered flushes; analyze the full state once.
    assert llm_client.calls == 1
    assert scanner.buffer_size == 0
    predictions = predictions_repo.list_predictions(game_id)
    # The bot reported exactly 3 slots (Мафия ×1, Мирный житель ×2), so the
    # analyzer fills the two players the fake LLM skipped.
    assert len(predictions) == 3
    roles_by_player = {int(row["player_id"]): str(row["predicted_role"]) for row in predictions}
    assert roles_by_player[VASYA] == "Мафия"
    assert roles_by_player[PETYA] == "Мирный житель"
    assert roles_by_player[LENA] == "Мирный житель"
