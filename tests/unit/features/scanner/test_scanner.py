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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from loguru import logger

from domain.events import AnyEvent, DeathEvent, MessageEvent, VoteEvent
from domain.exceptions import ConfigError
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


class FakeAnalyzer:
    """Counting analyzer that mimics :class:`GameAnalyzer`'s contract."""

    def __init__(self) -> None:
        self.calls: int = 0
        self.last_game_state: GameState | None = None
        self.next_success: bool = True
        self.raise_error: Exception | None = None
        self.on_analyze: Callable[[], None] | None = None

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
            self.on_analyze()
        if self.raise_error is not None:
            raise self.raise_error
        if self.next_success:
            return AnalysisResult(
                response=AnalysisResponse(analysis=[], game_read="ok", reasoning="ok"),
                error=None,
                duration_sec=0.001,
                reasoning_chars=0,
                warnings=(),
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

    assert scanner.name_to_id == {}
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
        "Город ветров начинает игру",
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

    applied = scanner._handle_player_speech(
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
    scanner, analyzer, _telegram = build_scanner(
        game_state=game_state,
        registry=registry,
        mode=TriggerMode.BATCH_DEBOUNCE,
        max_messages=1,
    )
    await scanner.start()
    injected: list[AnyEvent | None] = []

    def inject_during_analysis() -> None:
        if injected:
            return
        injected.append(
            scanner._handle_player_speech(
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


async def test_duplicate_death_is_rejected(
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

    await scanner._on_event(bot_event("Сегодня повешен Вася"))

    assert len(game_state.events_by_kind("death")) == 1
    assert analyzer.calls == calls_after_first_death
    assert any("is already dead" in text for text in _messages(log_records))


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


def test_register_player_rejects_blank_name(game_state: GameState, registry: RoleRegistry) -> None:
    """32. A blank display name is never registered (``Player`` requires text)."""
    scanner, _analyzer, _telegram = build_scanner(game_state=game_state, registry=registry)

    scanner._register_player(VASYA, "   ")

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
