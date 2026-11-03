"""Tests for ``features.scanner.scanner`` — async orchestration of the pipeline.

The scanner is exercised through its public lifecycle (``start``/``shutdown``)
and through Telethon-shaped fake events fed straight into ``_on_event``: the
real callback is installed by the wrapper, but the test owns the event loop, so
no Telethon and no network are involved.

Timers use a 50 ms debounce and a 100 ms sleep: short enough to keep the suite
fast, long enough to be stable on a loaded machine.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from app.orchestrator import Orchestrator
from domain.events import (
    AnyEvent,
    DeathEvent,
    MessageEvent,
    PhaseChangeEvent,
    PhaseChangePayload,
    VoteEvent,
)
from domain.exceptions import ConfigError, TelegramFloodWaitError
from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles
from features.analyzer.analyzer import AnalysisResult, GameAnalyzer
from features.scanner.scanner import MessageScanner, _sender_display_name
from features.scanner.trigger_policy import TriggerMode, TriggerPolicy
from infra.llm.schemas import AnalysisResponse
from infra.telegram.client import TelegramClientWrapper

if TYPE_CHECKING:
    from loguru import Record

ROLES_PATH = Path(__file__).resolve().parents[4] / "configs" / "roles.yaml"
TS = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)
GAME_ID = 1
CHAT_ID = -100
BOT_ID = 42
VASYA = 100
PETYA = 101
LENA = 102
DEBOUNCE = 0.05
DEBOUNCE_WAIT = 0.1

END_GAME_TEXT = """Игра окончена!

Победители:
    1. Вася - 🤵🏼 Мафия

Другие пользователи:
    2. Петя - 👨🏼 Мирный Житель

Игра длилась: 20 минут"""


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTelegramWrapper:
    """Minimal stand-in for :class:`TelegramClientWrapper`'s subscription API."""

    def __init__(self) -> None:
        self.handler: Callable[[Any], Any] | None = None
        self.chat_id: int | None = None
        self.add_calls: int = 0
        self.removed: bool = False
        self.remove_calls: int = 0
        self.recent_messages: list[Any] = []
        self.list_calls: list[tuple[int, int]] = []
        self.flood_wait_once: int | None = None

    def add_message_handler(
        self,
        handler: Callable[[Any], Any],
        *,
        chat_id: int,
        bot_only: bool = False,
        deduplicate: bool = True,
    ) -> None:
        """Record the handler and the chat id it was registered for."""
        self.handler = handler
        self.chat_id = chat_id
        self.add_calls += 1

    async def remove_all_handlers(self) -> None:
        """Record that all handlers were detached."""
        self.removed = True
        self.remove_calls += 1
        self.handler = None

    async def list_recent_messages(
        self,
        *,
        chat_id: int,
        limit: int = 50,
    ) -> list[Any]:
        """Return canned messages (newest-first) or raise FloodWait once."""
        self.list_calls.append((chat_id, limit))
        if self.flood_wait_once is not None:
            wait = self.flood_wait_once
            self.flood_wait_once = None
            raise TelegramFloodWaitError(wait)
        return list(self.recent_messages[:limit])


class FakeAnalyzer:
    """Counting analyzer that mimics :class:`GameAnalyzer`'s contract."""

    def __init__(self) -> None:
        self.calls: int = 0
        self.last_game_state: GameState | None = None
        self.next_success: bool = True
        self.raise_error: Exception | None = None
        self.on_analyze: Callable[[], Awaitable[None]] | None = None
        self.next_reasoning_text = ""

    async def analyze(
        self,
        game_state: GameState,
        *,
        persist: bool = True,
        previous_predictions: Any = None,
    ) -> AnalysisResult:
        """Count the call and return a canned success or failure result."""
        self.calls += 1
        self.last_game_state = game_state
        # The real analyzer awaits network I/O; yielding here models that
        # suspension point so re-entrancy guards are actually exercised.
        await asyncio.sleep(0)
        if self.on_analyze is not None:
            await self.on_analyze()
        if self.raise_error is not None:
            raise self.raise_error
        if self.next_success:
            return AnalysisResult(
                response=AnalysisResponse(analysis=[], game_read="ok", reasoning="ok"),
                error=None,
                duration_sec=0.001,
                reasoning_chars=0,
                warnings=(),
                reasoning_text=self.next_reasoning_text,
            )
        return AnalysisResult(
            response=None,
            error="boom",
            duration_sec=0.001,
            reasoning_chars=0,
        )


@dataclass
class FakeMessage:
    """Message body as seen through a Telethon event."""

    text: str | None


@dataclass
class FakeSender:
    """Sender entity subset used by ``_sender_display_name``."""

    first_name: str | None = None
    last_name: str | None = None
    title: str | None = None
    username: str | None = None


def make_event(
    *,
    sender_id: int | None,
    sender: Any = None,
    text: str | None = None,
    date: Any = TS,
    message: Any = None,
) -> SimpleNamespace:
    """Build a Telethon-shaped new-message event."""
    if message is None:
        message = FakeMessage(text=text) if text is not None else None
    return SimpleNamespace(
        sender_id=sender_id,
        sender=sender,
        message=message,
        date=date,
    )


def player_event(player_id: int, name: str, text: str) -> SimpleNamespace:
    """Build an event for a speaking player with a resolvable display name."""
    return make_event(sender_id=player_id, sender=FakeSender(first_name=name), text=text)


def bot_event(text: str) -> SimpleNamespace:
    """Build an event announced by the game bot."""
    return make_event(sender_id=BOT_ID, text=text)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> RoleRegistry:
    """Load the real role registry (single source of truth)."""
    return load_roles(ROLES_PATH)


@pytest.fixture
def game_state(registry: RoleRegistry) -> GameState:
    """Provide an empty game state for a fresh game."""
    return GameState(game_id=GAME_ID, registry=registry)


def start_game(game_state: GameState) -> None:
    """Enter an active phase without triggering a scanner flush."""
    game_state.apply(
        PhaseChangeEvent(
            ts=TS,
            game_id=GAME_ID,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
            ),
        )
    )


@pytest.fixture
def log_records() -> Iterator[list[Record]]:
    """Capture WARNING-and-above loguru records emitted during a test."""
    records: list[Record] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def build_scanner(
    *,
    game_state: GameState,
    registry: RoleRegistry,
    mode: TriggerMode = TriggerMode.QUEUE,
    debounce: float = DEBOUNCE,
    max_messages: int = 10,
    analyzer: FakeAnalyzer | None = None,
    telegram: FakeTelegramWrapper | None = None,
    on_event: Callable[[AnyEvent], Awaitable[None]] | None = None,
    on_reasoning: Callable[[str], Awaitable[None]] | None = None,
    on_flush_complete: Callable[[], Awaitable[None]] | None = None,
) -> tuple[MessageScanner, FakeAnalyzer, FakeTelegramWrapper]:
    """Assemble a scanner with fakes and the requested trigger policy."""
    fake_analyzer = analyzer if analyzer is not None else FakeAnalyzer()
    fake_telegram = telegram if telegram is not None else FakeTelegramWrapper()
    scanner = MessageScanner(
        telegram=cast(TelegramClientWrapper, fake_telegram),
        game_state=game_state,
        analyzer=cast(GameAnalyzer, fake_analyzer),
        registry=registry,
        trigger_policy=TriggerPolicy(
            mode=mode,
            batch_debounce_sec=debounce,
            batch_max_messages=max_messages,
        ),
        chat_id=CHAT_ID,
        bot_id=BOT_ID,
        on_event=on_event,
        on_reasoning=on_reasoning,
        on_flush_complete=on_flush_complete,
    )
    return scanner, fake_analyzer, fake_telegram


def _messages(records: list[Record]) -> list[str]:
    """Return only the formatted message texts of captured records."""
    return [record["message"] for record in records]


# ---------------------------------------------------------------------------
# 1-7. Lifecycle, filtering, identity
# ---------------------------------------------------------------------------


async def test_start_subscribes_once(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """1. ``start`` subscribes once; the second call is a no-op."""
    scanner, _analyzer, telegram = build_scanner(game_state=game_state, registry=registry)

    await scanner.start()
    first_handler = telegram.handler
    await scanner.start()

    assert telegram.add_calls == 1
    assert telegram.chat_id == CHAT_ID
    assert telegram.handler is first_handler
    assert scanner.is_started is True


async def test_process_event_ignores_no_sender(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """2. Events without a sender id are dropped before any work happens."""
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
    )

    await scanner._on_event(make_event(sender_id=None, text="анонимное сообщение"))

    assert scanner.buffer_size == 0
    assert analyzer.calls == 0
    assert scanner.name_to_id == {}


@pytest.mark.parametrize("text", [None, "", "   ", "ок"])
async def test_process_event_irrelevant(
    game_state: GameState,
    registry: RoleRegistry,
    text: str | None,
) -> None:
    """3. Empty, blank and too-short messages never reach the buffer."""
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
    )

    await scanner._on_event(player_event(VASYA, "Вася", text or ""))

    assert scanner.buffer_size == 0
    assert analyzer.calls == 0
    assert game_state.events() == ()


async def test_player_speech_added_to_buffer(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """4. Player speech becomes a ``MessageEvent`` applied to ``GameState``."""
    scanner, analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)

    await scanner._on_event(player_event(VASYA, "Вася", "Привет, я мирный житель"))

    assert scanner.buffer_size == 1
    assert len(game_state.events_by_kind("message")) == 1
    assert analyzer.calls == 0


async def test_live_game_start_re_registers_players(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """Known Telegram users can be tracked again after an END → new-game reset."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)
    await scanner._register_player(VASYA, "Вася")
    game_state.apply(
        PhaseChangeEvent(
            ts=TS,
            game_id=GAME_ID,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION,
                to_phase=Phase.END,
                day_number=0,
            ),
        )
    )

    await scanner._on_event(bot_event("Город ветров начинает игру"))

    assert game_state.current_phase() is Phase.REGISTRATION
    assert game_state.players() == ()
    await scanner._on_event(player_event(VASYA, "Вася", "Мы снова играем вместе"))
    assert len(game_state.players()) == 1
    assert game_state.player(VASYA).display_name == "Вася"


async def test_on_event_callback_invoked_for_player_speech(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """An applied player-speech event is forwarded to the callback once."""
    emitted: list[AnyEvent] = []

    async def on_event(event: AnyEvent) -> None:
        emitted.append(event)

    scanner, _analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        on_event=on_event,
    )

    await scanner._on_event(player_event(VASYA, "Вася", "Привет, я мирный житель"))

    assert len(emitted) == 1
    assert emitted[0] is game_state.events()[0]


async def test_on_event_callback_errors_swallowed(
    game_state: GameState,
    registry: RoleRegistry,
    log_records: list[Record],
) -> None:
    """Callback failures are logged without undoing the applied event."""

    async def on_event(_event: AnyEvent) -> None:
        raise RuntimeError("database unavailable")

    scanner, _analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        on_event=on_event,
    )

    await scanner._on_event(player_event(VASYA, "Вася", "Привет, я мирный житель"))

    assert len(game_state.events_by_kind("message")) == 1
    assert any(
        "on_event callback failed: RuntimeError" in message for message in _messages(log_records)
    )


async def test_name_to_id_populated(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """5. The Telegram display name is mapped to the sender id."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)

    await scanner._on_event(player_event(VASYA, "Вася", "Всем привет"))

    assert scanner.name_to_id == {"Вася": VASYA}


async def test_name_to_id_excludes_bot(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """6b. The game bot's own title never enters the nickname map."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)

    await scanner._on_event(
        make_event(
            sender_id=BOT_ID,
            sender=FakeSender(first_name="Мафия Бот"),
            text="🌙 Наступает ночь",
        )
    )

    assert scanner.name_to_id == {}
    assert game_state.current_phase() is Phase.NIGHT


async def test_name_to_id_does_not_overwrite(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """6. Two players sharing a nickname: the first observed spelling wins."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)

    await scanner._on_event(player_event(VASYA, "Вася", "Я первый"))
    await scanner._on_event(player_event(LENA, "Вася", "Я тоже Вася"))

    assert scanner.name_to_id["Вася"] == VASYA


async def test_fallback_name_used_when_sender_none(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """7. An unresolvable sender entity falls back to ``user_<id>``."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)

    await scanner._on_event(make_event(sender_id=VASYA, text="Я без имени"))

    assert scanner.name_to_id == {f"user_{VASYA}": VASYA}
    assert game_state.players()[0].display_name == f"user_{VASYA}"
    event = game_state.events_by_kind("message")[0]
    assert isinstance(event, MessageEvent)
    assert event.payload.author_id == VASYA
    assert event.payload.author_name == f"user_{VASYA}"


@pytest.mark.parametrize(
    ("sender", "expected"),
    [
        (FakeSender(first_name="Иван", last_name="Петров"), "Иван Петров"),
        (FakeSender(first_name="Иван"), "Иван"),
        (FakeSender(first_name="Иван", last_name=""), "Иван"),
        (FakeSender(title="Комната мафии"), "Комната мафии"),
        (FakeSender(username="ivan"), "ivan"),
        (FakeSender(), None),
        (SimpleNamespace(), None),
        (None, None),
    ],
)
def test_sender_display_name_priority(sender: Any, expected: str | None) -> None:
    """7b. ``_sender_display_name`` follows the documented priority order."""
    assert _sender_display_name(sender) == expected


# ---------------------------------------------------------------------------
# 8-13. Bot messages, phases, trigger modes
# ---------------------------------------------------------------------------


async def test_bot_message_resolves_death(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """8. A death announcement becomes an applied ``DeathEvent``."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)
    await scanner._on_event(player_event(VASYA, "Вася", "Я мирный житель"))

    await scanner._on_event(bot_event("Сегодня повешен Вася"))

    deaths = game_state.events_by_kind("death")
    assert len(deaths) == 1
    death = deaths[0]
    assert isinstance(death, DeathEvent)
    assert death.payload.victim_id == VASYA
    assert game_state.player(VASYA).alive is False


async def test_phase_event_forces_flush(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """9. In ``batch_debounce`` a phase change flushes without waiting."""
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.BATCH_DEBOUNCE,
    )

    await scanner._on_event(bot_event("🌙 Наступает ночь"))

    assert game_state.current_phase() is Phase.NIGHT
    assert analyzer.calls == 1
    assert scanner.buffer_size == 0


async def test_message_debounce(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """10. In ``batch_debounce`` a message flushes only after the quiet period."""
    start_game(game_state)
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.BATCH_DEBOUNCE,
    )

    await scanner._on_event(player_event(VASYA, "Вася", "Спокойный монолог"))
    assert analyzer.calls == 0
    assert scanner.buffer_size == 1

    await asyncio.sleep(DEBOUNCE_WAIT)

    assert analyzer.calls == 1
    assert scanner.buffer_size == 0


async def test_buffer_size_forces_flush(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """11. Reaching ``batch_max_messages`` flushes without waiting for the timer."""
    start_game(game_state)
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.BATCH_DEBOUNCE,
        max_messages=2,
    )

    await scanner._on_event(player_event(VASYA, "Вася", "Первое слово"))
    assert analyzer.calls == 0

    await scanner._on_event(player_event(PETYA, "Петя", "Второе слово"))

    assert analyzer.calls == 1
    assert scanner.buffer_size == 0


async def test_every_message_mode_flushes_every_time(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """12. ``every_message`` analyzes after each buffered event."""
    start_game(game_state)
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
    )

    await scanner._on_event(player_event(VASYA, "Вася", "Раз, я мирный"))
    await scanner._on_event(player_event(PETYA, "Петя", "Два, я тоже мирный"))
    await scanner._on_event(player_event(LENA, "Лена", "Три, и я мирная"))

    assert analyzer.calls == 3
    assert scanner.buffer_size == 0


async def test_queue_mode_no_auto_flush(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """13. ``queue`` never auto-flushes until the buffer is full."""
    start_game(game_state)
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.QUEUE,
        max_messages=10,
    )

    for index in range(5):
        await scanner._on_event(player_event(VASYA, "Вася", f"Сообщение номер {index}"))

    assert analyzer.calls == 0
    assert scanner.buffer_size == 5

    for index in range(5, 10):
        await scanner._on_event(player_event(VASYA, "Вася", f"Сообщение номер {index}"))

    assert analyzer.calls == 1
    assert scanner.buffer_size == 0


# ---------------------------------------------------------------------------
# 14-19. Shutdown, failures, properties
# ---------------------------------------------------------------------------


async def test_shutdown_cancels_timer(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """14. ``shutdown`` cancels the debounce timer before it can flush."""
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.BATCH_DEBOUNCE,
    )
    await scanner.start()
    await scanner._on_event(player_event(VASYA, "Вася", "Не успеет проанализироваться"))
    # One loop turn so the timer task actually starts sleeping before it is
    # cancelled — otherwise the coroutine would never enter its ``try`` block.
    await asyncio.sleep(0)

    await scanner.shutdown()
    await asyncio.sleep(DEBOUNCE_WAIT)

    assert analyzer.calls == 0
    assert scanner.buffer_size == 1


async def test_shutdown_removes_handlers(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """15. ``shutdown`` detaches handlers exactly once (idempotent)."""
    scanner, _analyzer, telegram = build_scanner(game_state=game_state, registry=registry)
    await scanner.start()

    await scanner.shutdown()
    await scanner.shutdown()

    assert telegram.removed is True
    assert telegram.remove_calls == 1
    assert telegram.handler is None
    assert scanner.is_started is False


async def test_apply_failure_logged_does_not_crash(
    game_state: GameState,
    registry: RoleRegistry,
    monkeypatch: pytest.MonkeyPatch,
    log_records: list[Record],
) -> None:
    """16. A rejected event is logged, dropped, and does not stop the loop."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)
    original_apply = game_state.apply
    attempts: list[str] = []

    def flaky_apply(event: AnyEvent) -> None:
        attempts.append(event.kind)
        if len(attempts) == 1:
            raise ConfigError("сломанное событие")
        original_apply(event)

    monkeypatch.setattr(game_state, "apply", flaky_apply)

    await scanner._on_event(player_event(VASYA, "Вася", "Первое сообщение"))
    assert scanner.buffer_size == 0
    assert any("GameState.apply(message) failed" in text for text in _messages(log_records))

    await scanner._on_event(player_event(VASYA, "Вася", "Второе сообщение"))
    assert scanner.buffer_size == 1
    assert len(game_state.events_by_kind("message")) == 1


async def test_flush_in_progress_no_reentry(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """17. Overlapping flush triggers call the analyzer only once."""
    start_game(game_state)
    scanner, analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)

    await asyncio.gather(scanner._flush(), scanner._flush())

    assert analyzer.calls == 1
    assert analyzer.last_game_state is game_state


async def test_buffer_size_property(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """18. ``buffer_size`` mirrors the number of buffered events."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)
    assert scanner.buffer_size == 0

    await scanner._on_event(player_event(VASYA, "Вася", "Один"))
    await scanner._on_event(player_event(PETYA, "Петя", "Два"))

    assert scanner.buffer_size == 2


async def test_is_started_property(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """19. ``is_started`` follows the lifecycle transitions."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)
    assert scanner.is_started is False

    await scanner.start()
    assert scanner.is_started is True

    await scanner.shutdown()
    assert scanner.is_started is False


# ---------------------------------------------------------------------------
# 20+. Defensive paths and remaining branches
# ---------------------------------------------------------------------------


async def test_handler_never_raises(
    game_state: GameState,
    registry: RoleRegistry,
    log_records: list[Record],
) -> None:
    """20. An analyzer crash escapes ``_flush`` but never the Telegram handler."""
    start_game(game_state)
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
    )
    analyzer.raise_error = RuntimeError("llm down")

    await scanner._on_event(player_event(VASYA, "Вася", "Провоцируем падение"))

    assert analyzer.calls == 1
    assert any("handler failed: RuntimeError: llm down" in text for text in _messages(log_records))


async def test_event_without_date_ignored(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """21. A message without a timestamp cannot become a domain event."""
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
    )

    await scanner._on_event(
        make_event(
            sender_id=VASYA,
            sender=FakeSender(first_name="Вася"),
            text="Без даты",
            date=None,
        )
    )

    assert scanner.buffer_size == 0
    assert analyzer.calls == 0


async def test_message_without_text_attribute_ignored(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """22. A payload without any text (sticker, service message) is irrelevant."""
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
    )

    await scanner._on_event(make_event(sender_id=VASYA, message=SimpleNamespace()))

    assert scanner.buffer_size == 0
    assert analyzer.calls == 0


async def test_bot_message_without_text(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """23. ``_handle_bot_message`` ignores empty text without touching the buffer."""
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
    )

    await scanner._handle_bot_message(None, TS)
    await scanner._handle_bot_message("", TS)
    await scanner._handle_bot_message("   ", TS)

    assert scanner.buffer_size == 0
    assert analyzer.calls == 0


@pytest.mark.parametrize(
    "text",
    [
        "абракадабра без смысла",
        "Регистрация открыта",
    ],
)
async def test_bot_message_without_events_is_ignored(
    game_state: GameState,
    registry: RoleRegistry,
    text: str,
) -> None:
    """24. Informational and unknown bot messages produce no events and no flush."""
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
    )

    await scanner._on_event(bot_event(text))

    assert scanner.buffer_size == 0
    assert analyzer.calls == 0
    assert game_state.events() == ()


async def test_player_speech_with_blank_text_is_dropped(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """24b. The synchronous player handler rejects blank text on its own."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)

    applied = await scanner._handle_player_speech(
        author_id=VASYA,
        author_name="Вася",
        text="   ",
        ts=TS,
    )

    assert applied is None
    assert scanner.buffer_size == 0
    assert game_state.events() == ()


async def test_flush_failure_logs_warning(
    game_state: GameState,
    registry: RoleRegistry,
    log_records: list[Record],
) -> None:
    """25. A failed analysis is a warning, not an exception."""
    start_game(game_state)
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
    )
    analyzer.next_success = False

    await scanner._on_event(player_event(VASYA, "Вася", "Анализ сломается"))

    assert analyzer.calls == 1
    assert any("flush failed: boom" in text for text in _messages(log_records))


async def test_messages_arriving_during_flush_are_rescheduled(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """26. Events buffered while the LLM thinks are re-debounced, never starved."""
    start_game(game_state)
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.BATCH_DEBOUNCE,
        max_messages=1,
    )
    await scanner.start()
    injected: list[AnyEvent | None] = []

    async def inject_during_analysis() -> None:
        if injected:
            return
        injected.append(
            await scanner._handle_player_speech(
                author_id=PETYA,
                author_name="Петя",
                text="Слово, пришедшее во время анализа",
                ts=TS,
            )
        )

    analyzer.on_analyze = inject_during_analysis

    await scanner._on_event(player_event(VASYA, "Вася", "Запускает первый анализ"))

    assert analyzer.calls == 1
    assert scanner.buffer_size == 1

    await asyncio.sleep(DEBOUNCE_WAIT)

    assert analyzer.calls == 2
    assert scanner.buffer_size == 0


async def test_apply_safely_already_dead_debug(
    game_state: GameState,
    registry: RoleRegistry,
    log_records: list[Record],
) -> None:
    """27. A second death for the same player is rejected without a flush."""
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
    )
    await scanner._on_event(player_event(VASYA, "Вася", "Я пока живой"))
    await scanner._on_event(bot_event("Сегодня повешен Вася"))
    calls_after_first_death = analyzer.calls

    sink = logger.add(lambda message: log_records.append(message.record), level="DEBUG")
    try:
        await scanner._on_event(bot_event("Сегодня повешен Вася"))
    finally:
        logger.remove(sink)

    assert len(game_state.events_by_kind("death")) == 1
    assert analyzer.calls == calls_after_first_death
    assert any(
        "is already dead" in r["message"] and r["level"].name == "DEBUG" for r in log_records
    )


async def test_bot_last_words_creates_player_and_message(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """28. Last words are attributed to the dying player as a message event."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)
    await scanner._on_event(player_event(VASYA, "Вася", "Я мирный житель"))

    await scanner._on_event(
        bot_event("Кто-то из жителей слышал, как Вася кричал перед смертью:\nМафия это Петя")
    )

    messages = game_state.events_by_kind("message")
    assert len(messages) == 2
    last_words = messages[-1]
    assert isinstance(last_words, MessageEvent)
    assert last_words.payload.author_id == VASYA
    assert "Мафия это Петя" in last_words.payload.text


async def test_bot_last_words_registers_unknown_player(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """29. A player known only by nickname is introduced from the bot message."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)
    scanner._name_to_id["Лена"] = LENA

    await scanner._on_event(
        bot_event("Кто-то из жителей слышал, как Лена кричал перед смертью:\nПрощайте")
    )

    assert game_state.player(LENA).display_name == "Лена"
    assert scanner._registered_ids == {LENA}


async def test_bot_vote_does_not_register_players(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """30. Votes resolve ids from the nickname map and introduce no players."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)
    await scanner._on_event(player_event(VASYA, "Вася", "Я мирный житель"))
    await scanner._on_event(player_event(PETYA, "Петя", "И я мирный житель"))
    registered_before = set(scanner._registered_ids)

    await scanner._on_event(bot_event("Вася голосовал за: Петя"))

    votes = game_state.events_by_kind("vote")
    assert len(votes) == 1
    vote = votes[0]
    assert isinstance(vote, VoteEvent)
    assert vote.payload.voter_id == VASYA
    assert vote.payload.target_id == PETYA
    assert scanner._registered_ids == registered_before


async def test_role_reveal_marks_known_player(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """31. A public role reveal updates the player and keeps the death order."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)
    await scanner._on_event(player_event(VASYA, "Вася", "Я мирный житель"))

    await scanner._on_event(bot_event("Сегодня повешен Вася"))
    await scanner._on_event(bot_event("Вася был Мафией"))

    revealed_player = game_state.player(VASYA)
    assert revealed_player.revealed is True
    assert revealed_player.known_role is not None
    assert revealed_player.known_role.id == "mafia"


async def test_register_player_rejects_blank_name(
    game_state: GameState, registry: RoleRegistry
) -> None:
    """32. A blank display name is never registered (``Player`` requires text)."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)

    await scanner._register_player(VASYA, "   ")

    assert game_state.players() == ()
    assert scanner._registered_ids == set()


async def test_shutdown_without_start_is_noop(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """33. ``shutdown`` before ``start`` neither raises nor touches the wrapper."""
    scanner, _analyzer, telegram = build_scanner(game_state=game_state, registry=registry)

    await scanner.shutdown()

    assert telegram.remove_calls == 0
    assert scanner.is_started is False


async def test_flush_forwards_successful_reasoning(registry: RoleRegistry) -> None:
    """Analyzer reasoning is forwarded to the optional callback after success."""
    received: list[str] = []

    async def on_reasoning(text: str) -> None:
        received.append(text)

    analyzer = FakeAnalyzer()
    analyzer.next_reasoning_text = "Краткий разбор партии"
    game_state = GameState(game_id=GAME_ID, registry=registry)
    start_game(game_state)
    scanner, _fake_analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        analyzer=analyzer,
        on_reasoning=on_reasoning,
    )

    await scanner._flush()

    assert received == ["Краткий разбор партии"]


@pytest.mark.parametrize("callback_fails", [False, True])
async def test_on_flush_complete_callback_invoked(
    game_state: GameState,
    registry: RoleRegistry,
    log_records: list[Record],
    callback_fails: bool,
) -> None:
    """A successful flush notifies completion once, after empty reasoning too."""
    start_game(game_state)
    steps: list[str] = []

    async def on_reasoning(text: str) -> None:
        assert text == ""
        steps.append("reasoning")

    async def on_flush_complete() -> None:
        steps.append("flush_complete")
        if callback_fails:
            raise RuntimeError("stats unavailable")

    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
        on_reasoning=on_reasoning,
        on_flush_complete=on_flush_complete,
    )

    await scanner._on_event(player_event(VASYA, "Вася", "Я мирный житель"))

    assert analyzer.calls == 1
    assert steps == ["reasoning", "flush_complete"]
    assert scanner.buffer_size == 0
    if callback_fails:
        assert any(
            "on_flush_complete callback failed: RuntimeError: stats unavailable" in text
            for text in _messages(log_records)
        )
    else:
        assert log_records == []


async def test_on_flush_complete_not_called_on_error(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """Failed analysis emits neither reasoning nor a flush-completed notification."""
    start_game(game_state)
    steps: list[str] = []

    async def on_reasoning(_text: str) -> None:
        steps.append("reasoning")

    async def on_flush_complete() -> None:
        steps.append("flush_complete")

    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
        on_reasoning=on_reasoning,
        on_flush_complete=on_flush_complete,
    )
    analyzer.next_success = False

    await scanner._on_event(player_event(VASYA, "Вася", "Я мирный житель"))

    assert analyzer.calls == 1
    assert steps == []
    assert scanner.buffer_size == 0


async def test_flush_ignores_failed_analysis_reasoning(registry: RoleRegistry) -> None:
    """Failed analyzer results do not emit reasoning to the window."""
    received: list[str] = []

    async def on_reasoning(text: str) -> None:
        received.append(text)

    analyzer = FakeAnalyzer()
    analyzer.next_success = False
    analyzer.next_reasoning_text = "не должно появиться"
    game_state = GameState(game_id=GAME_ID, registry=registry)
    start_game(game_state)
    scanner, _fake_analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        analyzer=analyzer,
        on_reasoning=on_reasoning,
    )

    await scanner._flush()

    assert received == []


async def test_player_registration_calls_on_player_registered_callback(game_state, registry):
    registered = []

    async def callback(player_id, name):
        registered.append((player_id, name))

    scanner, _, _ = build_scanner(game_state=game_state, registry=registry)
    scanner._on_player_registered = callback
    await scanner._on_event(player_event(VASYA, "Вася", "Я мирный житель"))
    assert registered == [(VASYA, "Вася")]


async def test_player_registration_dedupes_repeated_players(game_state, registry):
    registered = []

    async def callback(player_id, name):
        registered.append((player_id, name))
        await asyncio.sleep(0)

    scanner, _, _ = build_scanner(game_state=game_state, registry=registry)
    scanner._on_player_registered = callback
    await asyncio.gather(
        *[scanner._on_event(player_event(VASYA, "Вася", "Я мирный житель")) for _ in range(2)]
    )
    assert registered == [(VASYA, "Вася")]
    assert len(game_state.events()) == 2


async def test_player_registration_callback_error_swallowed(game_state, registry, log_records):
    async def callback(player_id, name):
        raise RuntimeError("failure")

    scanner, _, _ = build_scanner(game_state=game_state, registry=registry)
    scanner._on_player_registered = callback
    await scanner._on_event(player_event(VASYA, "Вася", "Я мирный житель"))
    assert len(game_state.events()) == 1
    assert any("on_player_registered failed" in text for text in _messages(log_records))


async def test_bot_markdown_links_resolve_through_parser(game_state, registry):
    scanner, _, _ = build_scanner(game_state=game_state, registry=registry)
    await scanner._on_event(player_event(VASYA, "Вася", "Я мирный житель"))
    await scanner._on_event(player_event(PETYA, "Петя", "Я тоже мирный житель"))
    await scanner._on_event(
        bot_event("[Вася](tg://user?id=100) голосовал за: [Петя](tg://user?id=101)")
    )
    await scanner._on_event(bot_event("Сегодня повешен [Вася](tg://user?id=100)"))
    vote = game_state.events_by_kind("vote")[0]
    assert isinstance(vote, VoteEvent)
    assert vote.payload.target_id == PETYA
    assert not game_state.player(VASYA).alive


async def test_flush_skips_registration_phase(
    game_state: GameState, registry: RoleRegistry
) -> None:
    scanner, analyzer, _ = build_scanner(game_state=game_state, registry=registry)
    for player_id, name in ((VASYA, "Вася"), (PETYA, "Петя"), (LENA, "Лена")):
        await scanner._on_event(player_event(player_id, name, "Игровое сообщение"))
    assert scanner.buffer_size == 3
    await scanner._flush()
    assert analyzer.calls == 0
    assert scanner.buffer_size == 0


async def test_flush_skips_end_phase(game_state: GameState, registry: RoleRegistry) -> None:
    scanner, analyzer, _ = build_scanner(game_state=game_state, registry=registry)
    await scanner._on_event(player_event(VASYA, "Вася", "Игровое сообщение"))
    await scanner._on_event(bot_event("🌙 Наступает ночь"))
    # Transition via a domain event so the test does not depend on bot end wording.
    game_state.apply(
        PhaseChangeEvent(
            ts=TS,
            game_id=GAME_ID,
            payload=PhaseChangePayload(from_phase=Phase.NIGHT, to_phase=Phase.END, day_number=0),
        )
    )
    await scanner._on_event(player_event(PETYA, "Петя", "После игры"))
    before = analyzer.calls
    await scanner._flush()
    assert analyzer.calls == before
    assert scanner.buffer_size == 0


async def test_flush_works_after_registration(
    game_state: GameState, registry: RoleRegistry
) -> None:
    scanner, analyzer, _ = build_scanner(game_state=game_state, registry=registry)
    await scanner._on_event(player_event(VASYA, "Вася", "Регистрация"))
    await scanner._flush()
    assert analyzer.calls == 0
    await scanner._on_event(bot_event("🌙 Наступает ночь"))
    await scanner._flush()
    assert analyzer.calls == 1
    assert scanner.buffer_size == 0


# ---------------------------------------------------------------------------
# Backfill (step 11a)
# ---------------------------------------------------------------------------


def history_message(
    *,
    sender_id: int,
    text: str,
    sender: Any = None,
    second: int = 0,
) -> SimpleNamespace:
    """Build one Telethon-shaped history item as yielded by ``iter_messages``."""
    return SimpleNamespace(
        sender_id=sender_id,
        sender=sender,
        text=text,
        date=TS + timedelta(seconds=second),
    )


async def test_backfill_applies_after_game_start(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """Messages before ``GAME_START`` are ignored; later ones enter ``GameState``."""
    scanner, _analyzer, telegram = build_scanner(game_state=game_state, registry=registry)
    chronological = [
        history_message(
            sender_id=999,
            sender=FakeSender(first_name="Старожил"),
            text="Реплика из прошлой партии",
            second=0,
        ),
        history_message(
            sender_id=BOT_ID,
            text="Город ветров начинает свою игру над жителями!",
            second=1,
        ),
        history_message(
            sender_id=VASYA,
            sender=FakeSender(first_name="Вася"),
            text="Всем привет в новой игре",
            second=2,
        ),
        history_message(
            sender_id=PETYA,
            sender=None,
            text="Я тоже в игре",
            second=3,
        ),
    ]
    # Telethon's iter_messages returns newest messages first.
    telegram.recent_messages = list(reversed(chronological))

    applied = await scanner.backfill()

    assert applied == 2
    assert telegram.list_calls == [(CHAT_ID, 5000)]
    assert 999 not in {player.player_id for player in game_state.players()}
    assert "Старожил" not in scanner.name_to_id
    messages = game_state.events_by_kind("message")
    assert len(messages) == 2
    assert [event.payload.text for event in messages if isinstance(event, MessageEvent)] == [
        "Всем привет в новой игре",
        "Я тоже в игре",
    ]
    assert game_state.player(PETYA).display_name == f"user_{PETYA}"


async def test_backfill_skips_completed_game(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """A final report after the latest start means the game must not be replayed."""
    scanner, analyzer, telegram = build_scanner(game_state=game_state, registry=registry)
    chronological = [
        history_message(
            sender_id=BOT_ID,
            text="Город ветров начинает свою игру над жителями!",
            second=0,
        ),
        history_message(
            sender_id=VASYA,
            sender=FakeSender(first_name="Вася"),
            text="Сообщение из завершённой игры",
            second=1,
        ),
        history_message(sender_id=BOT_ID, text=END_GAME_TEXT, second=2),
        history_message(
            sender_id=PETYA,
            sender=FakeSender(first_name="Петя"),
            text="Позднее сообщение",
            second=3,
        ),
    ]
    telegram.recent_messages = list(reversed(chronological))

    info_records: list[Record] = []
    sink_id = logger.add(lambda message: info_records.append(message.record), level="INFO")
    try:
        applied = await scanner.backfill()
    finally:
        logger.remove(sink_id)

    assert applied == 0
    assert game_state.events() == ()
    assert game_state.players() == ()
    assert analyzer.calls == 0
    assert any("game is completed" in record["message"] for record in info_records)


async def test_backfill_applies_current_game(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """An old END_GAME before the latest start does not block the active game."""
    scanner, _analyzer, telegram = build_scanner(game_state=game_state, registry=registry)
    chronological = [
        history_message(sender_id=BOT_ID, text=END_GAME_TEXT, second=0),
        history_message(
            sender_id=BOT_ID,
            text="Город ветров начинает свою игру над жителями!",
            second=1,
        ),
        history_message(
            sender_id=VASYA,
            sender=FakeSender(first_name="Вася"),
            text="Новое обсуждение 1",
            second=2,
        ),
        history_message(
            sender_id=PETYA,
            sender=FakeSender(first_name="Петя"),
            text="Новое обсуждение 2",
            second=3,
        ),
    ]
    telegram.recent_messages = list(reversed(chronological))

    applied = await scanner.backfill()

    assert applied == 2
    messages = game_state.events_by_kind("message")
    assert len(messages) == 2
    assert [event.payload.text for event in messages if isinstance(event, MessageEvent)] == [
        "Новое обсуждение 1",
        "Новое обсуждение 2",
    ]
    assert {player.display_name for player in game_state.players()} == {"Вася", "Петя"}


async def test_backfill_no_game_start_returns_zero(
    game_state: GameState,
    registry: RoleRegistry,
    log_records: list[Record],
) -> None:
    """Missing ``GAME_START`` logs a warning, touches nothing, and returns 0."""
    scanner, analyzer, telegram = build_scanner(game_state=game_state, registry=registry)
    chronological = [
        history_message(
            sender_id=VASYA,
            sender=FakeSender(first_name="Вася"),
            text="Первое сообщение без старта",
            second=0,
        ),
        history_message(
            sender_id=PETYA,
            sender=FakeSender(first_name="Петя"),
            text="Второе сообщение без старта",
            second=1,
        ),
    ]
    telegram.recent_messages = list(reversed(chronological))

    applied = await scanner.backfill(limit=100)

    assert applied == 0
    assert telegram.list_calls == [(CHAT_ID, 100)]
    assert game_state.events() == ()
    assert game_state.players() == ()
    assert analyzer.calls == 0
    assert any("GAME_START not found" in message for message in _messages(log_records))


async def test_backfill_flood_wait_retries(
    game_state: GameState,
    registry: RoleRegistry,
    monkeypatch: pytest.MonkeyPatch,
    log_records: list[Record],
) -> None:
    """``TelegramFloodWaitError`` sleeps ``wait_seconds`` and retries once."""
    scanner, analyzer, telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.BATCH_DEBOUNCE,
        on_flush_complete=AsyncMock(),
    )
    telegram.flood_wait_once = 3
    chronological = [
        history_message(
            sender_id=BOT_ID,
            text="Город ветров начинает свою игру над жителями!",
            second=0,
        ),
        history_message(
            sender_id=BOT_ID,
            text="🌙 Наступает ночь",
            second=1,
        ),
    ]
    telegram.recent_messages = list(reversed(chronological))

    real_sleep = asyncio.sleep
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        if delay > 0:
            slept.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("features.scanner.scanner.asyncio.sleep", fake_sleep)

    applied = await scanner.backfill()

    assert applied == 1
    assert telegram.list_calls == [(CHAT_ID, 5000), (CHAT_ID, 5000)]
    assert slept == [3]
    assert game_state.current_phase() is Phase.NIGHT
    assert analyzer.calls == 1
    assert scanner.buffer_size == 0
    assert any("flood wait 3s" in message for message in _messages(log_records))


async def test_backfill_flushes_at_end(
    game_state: GameState,
    registry: RoleRegistry,
) -> None:
    """Player messages trailing after ``GAME_START`` trigger a single end flush."""
    start_game(game_state)
    scanner, analyzer, telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.BATCH_DEBOUNCE,
        debounce=10.0,
        max_messages=50,
        on_flush_complete=AsyncMock(),
    )
    chronological = [
        history_message(
            sender_id=BOT_ID,
            text="Город ветров начинает свою игру над жителями!",
            second=0,
        ),
        history_message(
            sender_id=VASYA,
            sender=FakeSender(first_name="Вася"),
            text="Первое обсуждение после ночи",
            second=1,
        ),
        history_message(
            sender_id=PETYA,
            sender=FakeSender(first_name="Петя"),
            text="Второе обсуждение после ночи",
            second=2,
        ),
        history_message(
            sender_id=LENA,
            sender=FakeSender(first_name="Лена"),
            text="Третье обсуждение после ночи",
            second=3,
        ),
    ]
    telegram.recent_messages = list(reversed(chronological))

    applied = await scanner.backfill()

    assert applied == 3
    assert len(game_state.events_by_kind("message")) == 3
    assert analyzer.calls == 1
    assert scanner.buffer_size == 0
    assert scanner._debounce_task is None


async def test_backfill_applies_without_intermediate_flush(
    game_state: GameState, registry: RoleRegistry
) -> None:
    """Replay all 72 messages before the only analysis, even in every-message mode."""
    start_game(game_state)
    scanner, analyzer, telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.EVERY_MESSAGE,
        on_flush_complete=AsyncMock(),
    )
    chronological = [
        history_message(sender_id=BOT_ID, text="Город ветров начинает свою игру над жителями!"),
        *[
            history_message(
                sender_id=VASYA,
                sender=FakeSender(first_name="Вася"),
                text=f"Сообщение {i}",
                second=i + 1,
            )
            for i in range(72)
        ],
    ]
    telegram.recent_messages = list(reversed(chronological))
    event_counts: list[int] = []

    async def record_context() -> None:
        event_counts.append(len(game_state.events_by_kind("message")))

    analyzer.on_analyze = record_context
    assert await scanner.backfill() == 72
    assert analyzer.calls == 1
    assert event_counts == [72]
    assert scanner.buffer_size == 0


async def test_backfill_forces_flush_even_at_end(
    game_state: GameState,
    registry: RoleRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A phase-END marker without a final report still gets a forced flush.

    A normal ``_flush`` would clear the buffer and skip the analyzer, leaving
    retro with no predictions; backfill must call ``_flush_force`` so the
    single final analysis still happens. A real END_GAME report is skipped by
    the completed-game guard above.
    """
    start_game(game_state)
    scanner, analyzer, telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.BATCH_DEBOUNCE,
        debounce=10.0,
        max_messages=200,
        on_flush_complete=AsyncMock(),
    )
    chronological = [
        history_message(
            sender_id=BOT_ID,
            text="Город ветров начинает свою игру над жителями!",
            second=0,
        ),
        *[
            history_message(
                sender_id=VASYA,
                sender=FakeSender(first_name="Вася"),
                text=f"Сообщение {i}",
                second=i + 1,
            )
            for i in range(98)
        ],
        history_message(sender_id=BOT_ID, text="Победа мафии", second=99),
    ]
    telegram.recent_messages = list(reversed(chronological))

    force_calls: list[bool] = []
    real_force_flush = scanner._flush_force

    async def force_spy() -> None:
        force_calls.append(True)
        await real_force_flush()

    monkeypatch.setattr(scanner, "_flush_force", force_spy)

    applied = await scanner.backfill()

    assert applied == 99
    assert game_state.current_phase() is Phase.END
    assert force_calls == [True]
    assert analyzer.calls == 1
    assert scanner.buffer_size == 0
    assert scanner._debounce_task is None


def test_backfill_timeout_300s() -> None:
    """The orchestrator gives backfill the requested five-minute budget."""
    source = inspect.getsource(Orchestrator._backfill_safe)
    assert "300.0" in source
    assert "60.0" not in source


async def test_register_player_adds_canonical_form(
    game_state: GameState, registry: RoleRegistry
) -> None:
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)
    await scanner._register_player(1, "Мирный Житель ^‿^")
    assert scanner.name_to_id["Мирный Житель ^‿^"] == 1
    assert scanner.name_to_id["^‿^"] == 1


async def test_backfill_restores_flushing_after_failure(
    game_state: GameState, registry: RoleRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    scanner, _analyzer, telegram = build_scanner(game_state=game_state, registry=registry)
    telegram.recent_messages = [
        history_message(sender_id=VASYA, text="test", second=1),
        history_message(sender_id=BOT_ID, text="Город ветров начинает свою игру над жителями!"),
    ]
    scanner._flushing = True

    async def fail(_event: Any) -> None:
        assert scanner._flushing
        raise RuntimeError("replay failed")

    monkeypatch.setattr(scanner, "_process_event", fail)
    with pytest.raises(RuntimeError, match="replay failed"):
        await scanner.backfill()
    assert scanner._flushing is True


async def test_alive_list_calls_on_player_registered(game_state, registry):
    from domain.game_state import _placeholder_player_id

    scanner, _, _ = build_scanner(game_state=game_state, registry=registry)
    callback = AsyncMock()
    scanner._on_player_registered = callback
    await scanner._on_event(bot_event("Живые игроки:\n1. Never Wrote\nВсего: 1"))
    callback.assert_awaited_once_with(_placeholder_player_id("Never Wrote"), "Never Wrote")
    assert scanner._name_to_id["Never Wrote"] == _placeholder_player_id("Never Wrote")


async def test_flush_allows_alive_list_in_registration_phase(
    game_state: GameState, registry: RoleRegistry
) -> None:
    """A bot snapshot during REGISTRATION is a full game state, not noise."""
    scanner, analyzer, _ = build_scanner(game_state=game_state, registry=registry)
    await scanner._on_event(
        bot_event(
            "Живые игроки:\n1. user_1\n2. user_2\n\n"
            "Из них: 🤵🏼 Мафия - 1, 👨🏼 Мирный житель - 1\n\nВсего: 2"
        )
    )
    assert scanner.buffer_size > 0
    assert game_state.current_phase() is Phase.REGISTRATION

    await scanner._flush()

    assert analyzer.calls == 1
    assert scanner.buffer_size == 0
