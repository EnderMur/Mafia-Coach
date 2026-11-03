"""Tests for the headless command loop and event persistence."""

from __future__ import annotations

import asyncio
import inspect
import io
import queue
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from app.bootstrap import BootstrapResult, bootstrap
from app.orchestrator import Orchestrator, _StdinReader
from domain.events import (
    AliveListEvent,
    AliveListPayload,
    DeathEvent,
    DeathPayload,
    EndGameEvent,
    EndGamePayload,
    MessageEvent,
    MessagePayload,
    PhaseChangeEvent,
    PhaseChangePayload,
)
from domain.game_state import _placeholder_player_id
from domain.phases import Phase
from features.scanner.scanner import MessageScanner
from features.ui.protocol import ReasoningMessage, StatsMessage, StatsRow
from infra.llm.schemas import AnalysisResponse, PlayerPrediction
from tests.unit.app.conftest import FakeLLMClient, FakeTelegram, create_settings_file

GAME_CHAT_ID = -100123
PLAYER_ID = 100
EVENT_TS = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)


class FakeScanner:
    """Scanner test double with observable lifecycle state."""

    def __init__(self, chat_id: int) -> None:
        self.chat_id = chat_id
        self.started = False
        self.shutdown_called = False
        self.buffer_size = 0
        self.backfill_calls = 0
        self.backfill_result = 0
        self.backfill_error: Exception | None = None
        self.backfill_started = asyncio.Event()
        self.backfill_gate: asyncio.Event | None = None

    async def start(self) -> None:
        """Record scanner startup."""
        self.started = True

    async def backfill(self) -> int:
        """Record history backfill or raise the configured failure."""
        self.backfill_calls += 1
        self.backfill_started.set()
        if self.backfill_gate is not None:
            await self.backfill_gate.wait()
        if self.backfill_error is not None:
            raise self.backfill_error
        return self.backfill_result

    async def shutdown(self) -> None:
        """Record scanner shutdown."""
        self.shutdown_called = True


class QueueInput:
    """Blocking, thread-safe stdin stand-in for commands sent during run()."""

    def __init__(self) -> None:
        self._lines: queue.Queue[str] = queue.Queue()

    def readline(self) -> str:
        """Wait for the next input line like a terminal's readline."""
        return self._lines.get()

    def send(self, line: str) -> None:
        """Make one command available to the reader thread."""
        self._lines.put(line)


async def make_result(
    tmp_path: Path,
    *,
    chat_id: int | None = GAME_CHAT_ID,
) -> tuple[BootstrapResult, FakeTelegram, FakeLLMClient]:
    """Create migrated app resources with injected offline clients."""
    telegram = FakeTelegram()
    llm_client = FakeLLMClient()
    result = await bootstrap(
        create_settings_file(tmp_path, chat_id=chat_id),
        telegram=telegram,  # type: ignore[arg-type]
        llm_client=llm_client,
    )
    return result, telegram, llm_client


async def test_default_scanner_uses_resolved_bot_id(tmp_path: Path) -> None:
    """A bot id resolved by the Telegram wrapper is used when config has a username."""
    result, _telegram, _llm = await make_result(tmp_path)
    result.config = result.config.model_copy(
        update={"telegram": result.config.telegram.model_copy(update={"bot_id": None})}
    )
    orchestrator = Orchestrator(bootstrap_result=result)

    scanner = orchestrator._default_scanner(GAME_CHAT_ID)

    assert scanner._bot_id == 42
    await orchestrator.shutdown()


async def test_run_with_chat_id_starts_scanner(tmp_path: Path) -> None:
    """Configured chat starts the scanner and /quit releases the main loop."""
    result, _telegram, _llm = await make_result(tmp_path)
    scanner = FakeScanner(GAME_CHAT_ID)
    command_input = QueueInput()
    orchestrator = Orchestrator(
        bootstrap_result=result,
        scanner_factory=lambda _chat_id: cast(MessageScanner, scanner),
        stdin=command_input,  # type: ignore[arg-type]
        enable_ui=False,
    )

    run_task = asyncio.create_task(orchestrator.run())
    await asyncio.sleep(0.02)
    assert scanner.started is True
    await orchestrator._handle_quit()
    # The injected synchronous reader is blocked in a worker thread; release it
    # as well as setting the quit event so the default executor can close cleanly.
    command_input.send("/quit\n")
    await asyncio.wait_for(run_task, timeout=2.0)
    assert orchestrator._command_task is None
    await asyncio.wait_for(orchestrator.shutdown(), timeout=2.0)

    assert scanner.shutdown_called is True


async def test_run_without_chat_id_calls_attach(tmp_path: Path) -> None:
    """Unattached startup chooses a dialog, starts the scanner, and exits on EOF."""
    result, telegram, _llm = await make_result(tmp_path, chat_id=None)
    scanners: list[FakeScanner] = []

    def scanner_factory(chat_id: int) -> MessageScanner:
        scanner = FakeScanner(chat_id)
        scanners.append(scanner)
        return cast(MessageScanner, scanner)

    orchestrator = Orchestrator(
        bootstrap_result=result,
        scanner_factory=scanner_factory,
        stdin=io.StringIO("1\n"),
        enable_ui=False,
    )
    await asyncio.wait_for(orchestrator.run(), timeout=1.0)

    assert telegram.iter_dialogs_calls == [{"dialogs_limit": 50, "groups_only": True}]
    assert len(scanners) == 1
    assert scanners[0].chat_id == -1001
    assert scanners[0].started is True
    assert result.game_state.game_id > 0
    await orchestrator.shutdown()
    assert scanners[0].shutdown_called is True


async def test_handle_status_prints_game_info(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Status output includes game id, phase, day, alive count, and buffer size."""
    result, _telegram, _llm = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result)

    await orchestrator._handle_status()
    output = capsys.readouterr().out

    assert "Game #" in output
    assert "phase=" in output
    assert "current_day=" in output
    assert "alive=" in output
    assert "messages in buffer: 0" in output
    await orchestrator.shutdown()


async def test_handle_status_top5_predictions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Status prints exactly the five highest-confidence predictions in order."""
    result, _telegram, _llm = await make_result(tmp_path)
    for player_id in range(PLAYER_ID, PLAYER_ID + 7):
        result.repos.players.upsert_player(
            player_id=player_id,
            username=None,
            display_name=f"Игрок {player_id}",
        )
    confidences = [0.1, 0.95, 0.7, 0.3, 0.9, 0.2, 0.6]
    for index, confidence in enumerate(confidences):
        result.repos.predictions.insert_prediction(
            game_id=result.game_state.game_id,
            player_id=PLAYER_ID + index,
            predicted_role="Мафия",
            confidence=confidence,
            reasoning="проверка",
        )

    orchestrator = Orchestrator(bootstrap_result=result)
    await orchestrator._handle_status()
    output = capsys.readouterr().out
    prediction_lines = [line for line in output.splitlines() if line.startswith("player ")]

    assert len(prediction_lines) == 5
    assert [line.rsplit("(", maxsplit=1)[1] for line in prediction_lines] == [
        "95%)",
        "90%)",
        "70%)",
        "60%)",
        "30%)",
    ]
    await orchestrator.shutdown()


async def test_handle_command_unknown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Unknown commands are reported without raising."""
    result, _telegram, _llm = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result)

    await orchestrator._handle_command("/xyz")

    assert "Unknown command" in capsys.readouterr().out
    await orchestrator.shutdown()


async def test_handle_quit_sets_event(tmp_path: Path) -> None:
    """The /quit command signals the loop's quit event."""
    result, _telegram, _llm = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result)

    await orchestrator._handle_command("/quit")

    assert orchestrator._quit_event.is_set() is True
    await orchestrator.shutdown()


async def test_persist_event_inserts_into_db(tmp_path: Path) -> None:
    """The scanner callback writes a typed event to the events repository."""
    result, _telegram, _llm = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result)
    event = MessageEvent(
        ts=EVENT_TS,
        game_id=result.game_state.game_id,
        payload=MessagePayload(
            author_id=PLAYER_ID,
            author_name="Игрок",
            text="Тестовое сообщение для сохранения",
        ),
    )

    await orchestrator._persist_event(event)

    stored_events = result.repos.events.list_events(result.game_state.game_id)
    assert stored_events == [event]
    await orchestrator.shutdown()


async def test_end_game_schedules_retro_with_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """11b-fix-3. END_GAME defers retro so the final flush can finish first.

    ``_persist_event`` stores the event, creates a pending retro task and
    returns without running retro inline; retro is awaited only after the
    grace period. The production delay is 5 s, patched down here to keep the
    unit test fast.
    """
    result, _telegram, _llm = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)
    retro = AsyncMock()
    orchestrator._run_retro = retro  # type: ignore[method-assign]
    monkeypatch.setattr("app.orchestrator._RETRO_DELAY_SEC", 0.01)
    event = EndGameEvent(
        ts=EVENT_TS,
        game_id=result.game_state.game_id,
        payload=EndGamePayload(
            winners=[("Вася", "Мафия")],
            losers=[("Петя", "Мирный Житель")],
            duration_minutes=20,
        ),
    )

    await orchestrator._persist_event(event)

    assert result.repos.events.list_events(result.game_state.game_id) == [event]
    retro.assert_not_awaited()
    assert orchestrator._pending_retro is True
    pending = orchestrator._pending_retro_task
    assert pending is not None
    assert not pending.done()

    await asyncio.wait_for(pending, timeout=1.0)

    retro.assert_awaited_once_with()
    assert orchestrator._pending_retro is False
    await orchestrator.shutdown()


async def test_schedule_retro_waits_for_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retro waits until the scanner's in-flight flush finishes."""
    result, _telegram, _llm = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)
    scanner = orchestrator._default_scanner(GAME_CHAT_ID)
    orchestrator._scanner = scanner
    scanner._flushing = True
    retro = AsyncMock()
    orchestrator._run_retro = retro  # type: ignore[method-assign]

    real_sleep = asyncio.sleep
    poll_count = 0

    async def fake_sleep(delay: float) -> None:
        nonlocal poll_count
        if delay == 0.5:
            poll_count += 1
            if poll_count == 3:
                scanner._flushing = False
        await real_sleep(0)

    monkeypatch.setattr("app.orchestrator.asyncio.sleep", fake_sleep)

    await asyncio.wait_for(orchestrator._schedule_retro(delay=0), timeout=1.0)

    assert poll_count == 3
    assert scanner.is_flushing is False
    retro.assert_awaited_once_with()
    await orchestrator.shutdown()


async def test_shutdown_idempotent(tmp_path: Path) -> None:
    """Calling orchestrator shutdown twice is harmless."""
    result, telegram, llm_client = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result)

    await orchestrator.shutdown()
    await orchestrator.shutdown()

    assert telegram.aclose_calls == 0
    assert llm_client.aclose_calls == 0


async def test_handle_attach_selects_chat(tmp_path: Path) -> None:
    """Interactive attach selects the requested dialog and starts its scanner."""
    result, telegram, _llm = await make_result(tmp_path, chat_id=None)
    created_scanners: list[FakeScanner] = []

    def scanner_factory(chat_id: int) -> MessageScanner:
        scanner = FakeScanner(chat_id)
        created_scanners.append(scanner)
        return cast(MessageScanner, scanner)

    orchestrator = Orchestrator(
        bootstrap_result=result,
        scanner_factory=scanner_factory,
        stdin=io.StringIO("2\n"),
    )
    await orchestrator._handle_attach(None)

    assert len(created_scanners) == 1
    assert created_scanners[0].chat_id == -1002
    assert created_scanners[0].started is True
    assert telegram.get_chat_calls == [-1002]
    await orchestrator.shutdown()
    assert created_scanners[0].shutdown_called is True


async def test_handle_attach_direct_chat_id(tmp_path: Path) -> None:
    """An explicit numeric argument bypasses dialog selection."""
    result, telegram, _llm = await make_result(tmp_path, chat_id=None)
    created_scanners: list[FakeScanner] = []

    def scanner_factory(chat_id: int) -> MessageScanner:
        scanner = FakeScanner(chat_id)
        created_scanners.append(scanner)
        return cast(MessageScanner, scanner)

    orchestrator = Orchestrator(
        bootstrap_result=result,
        scanner_factory=scanner_factory,
    )
    await orchestrator._handle_command("/attach -100999")

    assert telegram.iter_dialogs_calls == []
    assert telegram.get_chat_calls == [-100999]
    assert created_scanners[0].chat_id == -100999
    assert created_scanners[0].started is True
    # 10a-fix-2: attach never backfills; run() schedules it in the background.
    assert created_scanners[0].backfill_calls == 0
    await orchestrator.shutdown()


async def test_run_backfills_in_background_after_scanner_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Backfill is scheduled by run() once a scanner and a game exist."""
    result, _telegram, _llm = await make_result(tmp_path)
    scanner = FakeScanner(GAME_CHAT_ID)
    scanner.backfill_result = 7
    orchestrator = Orchestrator(
        bootstrap_result=result,
        scanner_factory=lambda _chat_id: cast(MessageScanner, scanner),
        stdin=io.StringIO(""),
        enable_ui=False,
    )

    run_task = asyncio.create_task(orchestrator.run())
    await asyncio.wait_for(scanner.backfill_started.wait(), timeout=2.0)
    await asyncio.sleep(0.02)
    await asyncio.wait_for(run_task, timeout=2.0)

    output = capsys.readouterr().out
    assert "Backfill: loading recent messages..." not in output
    assert "Backfilled 7 messages." in output
    assert scanner.backfill_calls == 1
    await orchestrator.shutdown()


async def test_backfill_failure_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A backfill error is printed and logged without detaching the scanner."""
    result, _telegram, _llm = await make_result(tmp_path)
    scanner = FakeScanner(-100999)
    scanner.backfill_error = RuntimeError("history unavailable")

    orchestrator = Orchestrator(
        bootstrap_result=result,
        scanner_factory=lambda _chat_id: cast(MessageScanner, scanner),
        enable_ui=False,
    )
    await orchestrator._start_scanner(-100999)
    await orchestrator._backfill_safe()

    output = capsys.readouterr().out
    assert "Backfill failed: RuntimeError: history unavailable" in output
    assert scanner.started is True
    assert orchestrator._scanner is not None
    await orchestrator.shutdown()


async def test_backfill_timeout_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hanging backfill is cut at the 300 s budget and reported."""
    result, _telegram, _llm = await make_result(tmp_path)
    scanner = FakeScanner(GAME_CHAT_ID)
    scanner.backfill_gate = asyncio.Event()
    orchestrator = Orchestrator(
        bootstrap_result=result,
        scanner_factory=lambda _chat_id: cast(MessageScanner, scanner),
        enable_ui=False,
    )
    await orchestrator._start_scanner(GAME_CHAT_ID)

    real_wait_for = asyncio.wait_for

    async def short_wait_for(awaitable: Any, timeout: float) -> Any:
        assert timeout == 300.0
        return await real_wait_for(awaitable, timeout=0.01)

    monkeypatch.setattr("app.orchestrator.asyncio.wait_for", short_wait_for)
    await orchestrator._backfill_safe()

    assert "Backfill: timed out after 300s." in capsys.readouterr().out
    await orchestrator.shutdown()


async def test_handle_attach_invalid_chat_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-numeric direct attach argument is rejected without Telegram I/O."""
    result, telegram, _llm = await make_result(tmp_path, chat_id=None)
    orchestrator = Orchestrator(bootstrap_result=result)

    await orchestrator._handle_command("/attach nope")

    assert "chat_id должен быть числом" in capsys.readouterr().out
    assert telegram.get_chat_calls == []
    await orchestrator.shutdown()


async def test_handle_command_empty_is_noop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty command line has no output or side effect."""
    result, _telegram, _llm = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result)

    await orchestrator._handle_command("")

    assert capsys.readouterr().out == ""
    assert orchestrator._quit_event.is_set() is False
    await orchestrator.shutdown()


async def test_run_returns_if_quit_already_signaled(tmp_path: Path) -> None:
    """A pre-signaled quit avoids starting a configured scanner."""
    result, _telegram, _llm = await make_result(tmp_path)
    scanner = FakeScanner(GAME_CHAT_ID)
    orchestrator = Orchestrator(
        bootstrap_result=result,
        scanner_factory=lambda _chat_id: cast(MessageScanner, scanner),
    )

    await orchestrator._handle_quit()
    await orchestrator.run()

    assert scanner.started is False
    await orchestrator.shutdown()


class FakeWindowHost:
    """Spy window host used to verify best-effort orchestrator callbacks."""

    def __init__(self) -> None:
        self.stats: list[StatsMessage] = []
        self.reasoning: list[ReasoningMessage] = []
        self.shutdown_calls = 0
        self.start_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def send_stats(self, message: StatsMessage) -> None:
        self.stats.append(message)

    async def send_reasoning(self, message: ReasoningMessage) -> None:
        self.reasoning.append(message)

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


async def test_orchestrator_send_stats_empty_predictions_clears_rows(tmp_path: Path) -> None:
    """An empty prediction table sends an empty frame instead of leaving stale rows."""
    result, _telegram, _llm = await make_result(tmp_path)
    ui = FakeWindowHost()
    ui.stats.append(
        StatsMessage(
            game_id=result.game_state.game_id,
            phase="day",
            day=1,
            rows=[StatsRow(player_name="Вася", role="Мафия", confidence=0.9)],
        )
    )
    orchestrator = Orchestrator(
        bootstrap_result=result,
        window_host=ui,  # type: ignore[arg-type]
    )

    await orchestrator._send_stats()

    assert len(ui.stats) == 2
    assert ui.stats[-1].game_id == result.game_state.game_id
    assert ui.stats[-1].rows == []
    await orchestrator.shutdown()


async def test_send_stats_includes_all_players(tmp_path: Path) -> None:
    """One prediction and two unknown players are all shown in confidence/name order."""
    result, _telegram, _llm = await make_result(tmp_path)
    for player_id, name in ((100, "Вера"), (101, "Борис"), (102, "Аня")):
        result.game_state.register_player(player_id, name)
    result.repos.players.upsert_player(player_id=100, username=None, display_name="Вера")
    result.repos.predictions.upsert_prediction(
        game_id=result.game_state.game_id,
        player_id=100,
        predicted_role="Мафия",
        confidence=0.8,
        reasoning="голосование",
    )
    ui = FakeWindowHost()
    orchestrator = Orchestrator(bootstrap_result=result, window_host=ui)  # type: ignore[arg-type]
    try:
        await orchestrator._send_stats()
        assert ui.stats[-1].rows == [
            StatsRow(player_name="Вера", role="Мафия", confidence=0.8),
            StatsRow(player_name="Аня", role="—", confidence=0.0),
            StatsRow(player_name="Борис", role="—", confidence=0.0),
        ]
    finally:
        await orchestrator.shutdown()


async def test_send_stats_skips_placeholder(tmp_path: Path) -> None:
    """Synthetic non-positive player IDs must never be sent to the window."""
    result, _telegram, _llm = await make_result(tmp_path)
    result.game_state.register_player(100, "Реальный")
    result.game_state.register_player(-12345, "Плейсхолдер")
    result.game_state.register_player(0, "Нулевой плейсхолдер")
    ui = FakeWindowHost()
    orchestrator = Orchestrator(bootstrap_result=result, window_host=ui)  # type: ignore[arg-type]
    try:
        await orchestrator._send_stats()
        assert ui.stats[-1].rows == [StatsRow(player_name="Реальный", role="—", confidence=0.0)]
    finally:
        await orchestrator.shutdown()


async def test_send_stats_keeps_low_confidence_players_but_hides_prediction(tmp_path: Path) -> None:
    """The threshold is inclusive and a lower score renders an unknown role."""
    result, _telegram, _llm = await make_result(tmp_path)
    for player_id, name, confidence in ((100, "Анна", 0.15), (101, "Борис", 0.14)):
        result.game_state.register_player(player_id, name)
        result.repos.players.upsert_player(player_id=player_id, username=None, display_name=name)
        result.repos.predictions.upsert_prediction(
            game_id=result.game_state.game_id,
            player_id=player_id,
            predicted_role="Мафия",
            confidence=confidence,
            reasoning="голосование",
        )
    ui = FakeWindowHost()
    orchestrator = Orchestrator(bootstrap_result=result, window_host=ui)  # type: ignore[arg-type]
    try:
        await orchestrator._send_stats()
        assert ui.stats[-1].rows == [
            StatsRow(player_name="Анна", role="Мафия", confidence=0.15),
            StatsRow(player_name="Борис", role="—", confidence=0.0),
        ]
    finally:
        await orchestrator.shutdown()


async def test_send_stats_preserves_predictions_beyond_first_twenty(tmp_path: Path) -> None:
    """A large game does not turn valid predictions after the old cap into dashes."""
    result, _telegram, _llm = await make_result(tmp_path)
    for player_id in range(100, 125):
        name = f"Игрок {player_id}"
        result.game_state.register_player(player_id, name)
        result.repos.players.upsert_player(player_id=player_id, username=None, display_name=name)
        result.repos.predictions.upsert_prediction(
            game_id=result.game_state.game_id,
            player_id=player_id,
            predicted_role="Мафия",
            confidence=0.5,
            reasoning="наблюдение",
        )
    ui = FakeWindowHost()
    orchestrator = Orchestrator(bootstrap_result=result, window_host=ui)  # type: ignore[arg-type]
    try:
        await orchestrator._send_stats()
        assert len(ui.stats[-1].rows) == 25
        assert all(row.role == "Мафия" for row in ui.stats[-1].rows)
        assert [row.player_name for row in ui.stats[-1].rows] == [
            f"Игрок {player_id}" for player_id in range(100, 125)
        ]
    finally:
        await orchestrator.shutdown()


async def test_send_stats_logs_debug_info(tmp_path: Path) -> None:
    """_send_stats and _send_reasoning log at DEBUG without noisy INFO lines."""
    result, _telegram, _llm = await make_result(tmp_path)
    ui = FakeWindowHost()
    orchestrator = Orchestrator(
        bootstrap_result=result,
        window_host=ui,  # type: ignore[arg-type]
    )
    debug_captured = io.StringIO()
    info_captured = io.StringIO()
    debug_sink = logger.add(debug_captured, level="DEBUG", format="{message}")
    info_sink = logger.add(info_captured, level="INFO", format="{message}")
    try:
        await orchestrator._send_stats()
        await orchestrator._send_reasoning("краткий разбор")
    finally:
        logger.remove(debug_sink)
        logger.remove(info_sink)

    assert "send_stats: 0 rows" in debug_captured.getvalue()
    assert "send_reasoning: text_len=" in debug_captured.getvalue()
    assert info_captured.getvalue() == ""
    await orchestrator.shutdown()


async def test_orchestrator_send_stats_on_flush_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default factory refreshes stats once, after analyzer persistence."""
    result, _telegram, llm = await make_result(tmp_path)
    llm.response = AnalysisResponse(
        analysis=[
            PlayerPrediction(
                player_id=PLAYER_ID,
                player_name="Вася",
                suspected_role="Мирный житель",
                confidence=0.85,
                reasons=["Спокойно разбирает аргументы"],
            )
        ],
        game_read="Партия продолжается.",
        reasoning="Разбор сохранённых событий.",
    )
    ui = FakeWindowHost()
    orchestrator = Orchestrator(
        bootstrap_result=result,
        window_host=ui,  # type: ignore[arg-type]
    )
    send_stats = AsyncMock(wraps=orchestrator._send_stats)
    monkeypatch.setattr(orchestrator, "_send_stats", send_stats)
    await orchestrator._start_scanner(GAME_CHAT_ID)
    scanner = orchestrator._scanner
    assert scanner is not None
    assert scanner._on_flush_complete_callback is orchestrator._send_stats
    await scanner._register_player(PLAYER_ID, "Вася")
    result.game_state.apply(
        PhaseChangeEvent(
            ts=EVENT_TS,
            game_id=result.game_state.game_id,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
            ),
        )
    )

    await scanner._flush()

    send_stats.assert_awaited_once_with()
    assert len(ui.stats) == 1
    assert ui.stats[0].rows == [StatsRow(player_name="Вася", role="Мирный житель", confidence=0.85)]
    await orchestrator.shutdown()


async def test_orchestrator_persist_event_does_not_call_send_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Event insertion alone no longer refreshes stats before analysis."""
    result, _telegram, _llm = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)
    send_stats = AsyncMock()
    monkeypatch.setattr(orchestrator, "_send_stats", send_stats)
    event = MessageEvent(
        ts=EVENT_TS,
        game_id=result.game_state.game_id,
        payload=MessagePayload(
            author_id=PLAYER_ID,
            author_name="Игрок",
            text="Сообщение для сохранения",
        ),
    )

    await orchestrator._persist_event(event)

    send_stats.assert_not_awaited()
    assert result.repos.events.list_events(result.game_state.game_id) == [event]
    await orchestrator.shutdown()


async def test_orchestrator_send_reasoning_uses_placeholder(tmp_path: Path) -> None:
    """A non-thinking model produces an explicit explanation instead of a blank window."""
    result, _telegram, _llm = await make_result(tmp_path)
    ui = FakeWindowHost()
    orchestrator = Orchestrator(
        bootstrap_result=result,
        window_host=ui,  # type: ignore[arg-type]
    )
    placeholder = "(streaming reasoning unavailable — non-thinking model)"

    await orchestrator._send_reasoning("")

    assert ui.reasoning == [
        ReasoningMessage(text=placeholder, truncated=False, total_chars=len(placeholder))
    ]
    await orchestrator.shutdown()


async def test_orchestrator_sends_reasoning_after_scanner_flush(tmp_path: Path) -> None:
    """Successful analyzer reasoning is forwarded through scanner callback."""
    result, _telegram, _llm = await make_result(tmp_path)
    ui = FakeWindowHost()
    orchestrator = Orchestrator(
        bootstrap_result=result,
        window_host=ui,  # type: ignore[arg-type]
    )
    scanner = orchestrator._default_scanner(GAME_CHAT_ID)
    result.game_state.apply(
        PhaseChangeEvent(
            ts=EVENT_TS,
            game_id=result.game_state.game_id,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
            ),
        )
    )

    await scanner._flush()

    assert len(ui.reasoning) == 1
    assert "Проверяю" in ui.reasoning[0].text
    await orchestrator.shutdown()


async def test_non_thinking_json_reasoning_reaches_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON fallback flows from the analyzer through a real scanner flush to the UI."""
    result, _telegram, llm = await make_result(tmp_path)
    response = AnalysisResponse(analysis=[], game_read="Партия продолжается.", reasoning="My COT")
    analyze_stream = AsyncMock(return_value=response)
    monkeypatch.setattr(llm, "analyze_stream", analyze_stream)
    ui = FakeWindowHost()
    orchestrator = Orchestrator(bootstrap_result=result, window_host=ui)  # type: ignore[arg-type]
    try:
        scanner = orchestrator._default_scanner(GAME_CHAT_ID)
        result.game_state.apply(
            PhaseChangeEvent(
                ts=EVENT_TS,
                game_id=result.game_state.game_id,
                payload=PhaseChangePayload(
                    from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
                ),
            )
        )

        await scanner._flush()

        analyze_stream.assert_awaited_once()
        assert ui.reasoning == [ReasoningMessage(text="My COT", truncated=False, total_chars=6)]
    finally:
        await orchestrator.shutdown()


async def test_orchestrator_ui_disabled_is_noop(tmp_path: Path) -> None:
    """Explicit headless mode has no host and skips UI notifications."""
    result, _telegram, _llm = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)

    assert orchestrator._ui is None
    await orchestrator._send_stats()
    await orchestrator._send_reasoning("reasoning")

    await orchestrator.shutdown()


async def test_orchestrator_shutdown_closes_ui(tmp_path: Path) -> None:
    """The injected UI host is shut down along with application resources."""
    result, _telegram, _llm = await make_result(tmp_path)
    ui = FakeWindowHost()
    orchestrator = Orchestrator(
        bootstrap_result=result,
        window_host=ui,  # type: ignore[arg-type]
    )

    await orchestrator.shutdown()

    assert ui.shutdown_calls == 1


async def test_orchestrator_persist_player_on_registration(tmp_path: Path) -> None:
    result, _, _ = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)
    scanner = orchestrator._default_scanner(GAME_CHAT_ID)
    await scanner._register_player(PLAYER_ID, "Вася")
    player = result.repos.players.get_player(PLAYER_ID)
    assert player is not None
    assert player["username"] == "Вася"
    await orchestrator.shutdown()


async def test_orchestrator_persist_player_dedup(tmp_path: Path, monkeypatch) -> None:
    result, _, _ = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)
    original = result.repos.players.upsert_player
    calls = []

    def upsert(**kwargs):
        calls.append(kwargs)
        original(**kwargs)

    monkeypatch.setattr(result.repos.players, "upsert_player", upsert)
    await asyncio.gather(
        orchestrator._persist_player(PLAYER_ID, "Вася"),
        orchestrator._persist_player(PLAYER_ID, "Вася"),
    )
    assert len(calls) == 1
    await orchestrator.shutdown()


async def test_persist_player_adds_to_game_players(tmp_path: Path) -> None:
    """10c-fix-12. Registration writes both ``players`` and ``game_players``."""
    result, _, _ = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)

    await orchestrator._persist_player(PLAYER_ID, "Вася")

    game_id = result.game_state.game_id
    assert game_id > 0
    assert result.repos.players.get_player(PLAYER_ID) is not None
    rows = result.repos.game_players.list_game_players(game_id)
    assert len(rows) == 1
    assert rows[0]["game_id"] == game_id
    assert rows[0]["player_id"] == PLAYER_ID
    assert rows[0]["final_role"] is None
    await orchestrator.shutdown()


async def test_persist_player_add_game_players_idempotent(tmp_path: Path) -> None:
    """10c-fix-12. A repeated registration keeps exactly one participation row."""
    result, _, _ = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)
    game_id = result.game_state.game_id

    await orchestrator._persist_player(PLAYER_ID, "Вася")
    first_rows = result.repos.game_players.list_game_players(game_id)
    # A fresh orchestrator has an empty in-memory dedup set, so this really
    # exercises the repository's INSERT OR IGNORE path instead of the guard.
    second = Orchestrator(bootstrap_result=result, enable_ui=False)
    await second._persist_player(PLAYER_ID, "Вася")

    rows = result.repos.game_players.list_game_players(game_id)
    assert len(rows) == 1
    assert first_rows[0]["id"] == rows[0]["id"]
    await second.shutdown()
    await orchestrator.shutdown()


async def test_run_retro_persists_all_players(tmp_path: Path) -> None:
    """11b-fix-2. _run_retro persists END_GAME placeholders before retro_analyze.

    All players of game_state (including placeholders never registered by the
    scanner) must reach game_players, otherwise retro drops them as unknown.
    """
    result, _, _ = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)

    game_state = result.game_state
    game_state.register_player(PLAYER_ID, "Вася")
    game_state.register_player(-1, "Игрок 1")
    game_state.register_player(-2, "Игрок 2")
    assert PLAYER_ID not in orchestrator._registered_player_ids

    persist_spy = AsyncMock(wraps=orchestrator._persist_player)
    orchestrator._persist_player = persist_spy  # type: ignore[method-assign]

    await orchestrator._run_retro()

    assert persist_spy.await_count == 3
    persisted_ids = {call.args[0] for call in persist_spy.await_args_list}
    assert persisted_ids == {PLAYER_ID, -1, -2}
    game_id = result.game_state.game_id
    rows = result.repos.game_players.list_game_players(game_id)
    assert {row["player_id"] for row in rows} == {PLAYER_ID, -1, -2}
    await orchestrator.shutdown()


async def test_run_retro_merges_placeholder(tmp_path: Path) -> None:
    """A placeholder and its real twin are merged before retro (one row each)."""

    class RecordingGamePlayers:
        """Delegating proxy that records participation deletes."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.deleted: list[int] = []

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def delete_player(self, *, game_id: int, player_id: int) -> None:
            self.deleted.append(player_id)
            self._inner.delete_player(game_id=game_id, player_id=player_id)

    result, _, _ = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)

    game_state = result.game_state
    game_id = game_state.game_id
    placeholder_id = _placeholder_player_id("Вася")
    game_state.register_player(456, "Вася")
    game_state.register_player(placeholder_id, "Вася")
    game_state.register_player(457, "Петя")

    # SQLite foreign keys require players rows even for negative ids (they are
    # exactly what _persist_player writes for ALIVE_LIST placeholders).
    for player in game_state.players():
        result.repos.players.upsert_player(
            player_id=player.player_id,
            username=None,
            display_name=player.display_name,
        )
    repo = RecordingGamePlayers(result.repos.game_players)
    result.repos.game_players = repo  # type: ignore[assignment]
    for player in game_state.players():
        await orchestrator._persist_player(player.player_id, player.display_name)
    result.repos.predictions.upsert_prediction(
        game_id=game_id,
        player_id=placeholder_id,
        predicted_role="Мафия",
        confidence=0.8,
        reasoning="голосовал против дона",
    )

    await orchestrator._run_retro()

    predictions = result.repos.predictions.list_predictions(game_id)
    assert [int(row["player_id"]) for row in predictions] == [456]
    assert repo.deleted == [placeholder_id]
    remaining = {
        int(row["player_id"]) for row in result.repos.game_players.list_game_players(game_id)
    }
    assert placeholder_id not in remaining
    assert {456, 457} <= remaining
    assert result.repos.players.get_player(placeholder_id) is None
    assert result.repos.players.get_player(456) is not None
    await orchestrator.shutdown()


async def test_merge_predictions_db(tmp_path: Path) -> None:
    """merge_player keeps the real row and repoints the placeholder row."""
    result, _, _ = await make_result(tmp_path)
    game_id = result.game_state.game_id
    repo = result.repos.predictions
    result.repos.players.upsert_player(player_id=-7, username=None, display_name="Плейсхолдер")
    result.repos.players.upsert_player(player_id=-8, username=None, display_name="Второй")
    result.repos.players.upsert_player(player_id=456, username=None, display_name="Реальный")
    repo.upsert_prediction(
        game_id=game_id, player_id=-7, predicted_role="Мафия", confidence=0.7, reasoning="дубль"
    )

    # real id has no prediction yet → the placeholder row moves over.
    repo.merge_player(from_player_id=-7, to_player_id=456, game_id=game_id)
    rows = repo.list_predictions(game_id)
    assert [(int(row["player_id"]), row["predicted_role"]) for row in rows] == [(456, "Мафия")]

    # both rows exist → the real prediction wins, the duplicate is deleted.
    repo.upsert_prediction(
        game_id=game_id, player_id=-8, predicted_role="Мирный житель", confidence=0.3, reasoning="x"
    )
    repo.merge_player(from_player_id=-8, to_player_id=456, game_id=game_id)
    rows = repo.list_predictions(game_id)
    assert [(int(row["player_id"]), row["predicted_role"]) for row in rows] == [(456, "Мафия")]


async def test_handle_status_uses_total_players(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """_handle_status prefers GameState.total_players over len(players())."""
    result, _telegram, _llm = await make_result(tmp_path)
    for idx in range(1, 6):
        result.game_state.register_player(PLAYER_ID + idx, f"Игрок {idx}")
    result.game_state.apply(
        AliveListEvent(
            ts=EVENT_TS,
            game_id=result.game_state.game_id,
            payload=AliveListPayload(
                alive_player_names=[f"Игрок {idx}" for idx in range(1, 11)],
                total_players=10,
            ),
        )
    )
    orchestrator = Orchestrator(bootstrap_result=result)

    await orchestrator._handle_status()
    output = capsys.readouterr().out

    assert "alive=10/10" in output
    assert "alive=5/5" not in output
    await orchestrator.shutdown()


async def test_handle_status_falls_back_to_alive_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """_handle_status falls back to alive count when total_players is None."""
    result, _telegram, _llm = await make_result(tmp_path)
    for idx in range(1, 4):
        result.game_state.register_player(PLAYER_ID + idx, f"Игрок {idx}")
    result.game_state.apply(
        DeathEvent(
            ts=EVENT_TS,
            game_id=result.game_state.game_id,
            payload=DeathPayload(
                victim_id=PLAYER_ID + 1,
                victim_name="Игрок 1",
                cause="night_kill",
            ),
        )
    )
    assert result.game_state.total_players is None
    orchestrator = Orchestrator(bootstrap_result=result)

    await orchestrator._handle_status()
    output = capsys.readouterr().out

    # Дубли и placeholder-и не раздувают total: без данных бота это живые.
    assert "alive=2/2" in output
    assert "alive=2/3" not in output
    await orchestrator.shutdown()


# ---------------------------------------------------------------------------
# 10a-fix-2: stdin reader thread, DB lock, background backfill
# ---------------------------------------------------------------------------


async def test_stdin_reader_yields_lines() -> None:
    """The reader thread forwards every line and a final ``None`` at EOF."""
    reader = _StdinReader(io.StringIO("a\nb\n"))
    reader.start(asyncio.get_running_loop())
    try:
        first = await asyncio.wait_for(reader.get(), timeout=2.0)
        second = await asyncio.wait_for(reader.get(), timeout=2.0)
        eof = await asyncio.wait_for(reader.get(), timeout=2.0)
    finally:
        reader.stop()

    assert first == "a\n"
    assert second == "b\n"
    assert eof is None


async def test_db_lock_serializes_persist_and_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent persist/stats calls never hold the DB at the same time."""
    result, _telegram, _llm = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)
    event = MessageEvent(
        ts=EVENT_TS,
        game_id=result.game_state.game_id,
        payload=MessagePayload(author_id=PLAYER_ID, author_name="Игрок", text="текст"),
    )
    overlaps: list[str] = []
    active = 0
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        nonlocal active
        active += 1
        if active > 1:
            overlaps.append(getattr(func, "__name__", repr(func)))
        try:
            await asyncio.sleep(0)
            return await real_to_thread(func, *args, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr("app.orchestrator.asyncio.to_thread", spy_to_thread)
    monkeypatch.setattr(orchestrator, "_ui", FakeWindowHost())

    await asyncio.wait_for(
        asyncio.gather(
            orchestrator._persist_event(event),
            orchestrator._send_stats(),
            orchestrator._persist_event(event),
            orchestrator._send_stats(),
        ),
        timeout=5.0,
    )

    assert overlaps == []
    assert result.repos.events.list_events(result.game_state.game_id) == [event, event]
    await orchestrator.shutdown()


async def test_backfill_does_not_block_commands(tmp_path: Path) -> None:
    """A hung backfill leaves the command loop alive and is cancelled on quit."""
    result, _telegram, _llm = await make_result(tmp_path)
    scanner = FakeScanner(GAME_CHAT_ID)
    scanner.backfill_gate = asyncio.Event()  # never released: backfill hangs
    command_input = QueueInput()
    orchestrator = Orchestrator(
        bootstrap_result=result,
        scanner_factory=lambda _chat_id: cast(MessageScanner, scanner),
        stdin=command_input,  # type: ignore[arg-type]
        enable_ui=False,
    )

    run_task = asyncio.create_task(orchestrator.run())
    await asyncio.wait_for(scanner.backfill_started.wait(), timeout=2.0)
    await asyncio.sleep(0.02)

    assert orchestrator._backfill_task is not None
    assert not orchestrator._backfill_task.done()
    assert orchestrator._command_task is not None
    assert not orchestrator._command_task.done()

    command_input.send("/quit\n")
    await asyncio.wait_for(run_task, timeout=2.0)

    assert orchestrator._backfill_task is None
    assert scanner.backfill_calls == 1
    await asyncio.wait_for(orchestrator.shutdown(), timeout=2.0)


def test_backfill_timeout_300_in_source() -> None:
    source = inspect.getsource(Orchestrator._backfill_safe)
    assert "300.0" in source
    assert "60.0" not in source


async def test_handle_reset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result, _, _ = await make_result(tmp_path)
    old_game_id = result.game_state.game_id
    scanners: list[FakeScanner] = []

    def factory(chat_id: int) -> MessageScanner:
        scanner = FakeScanner(chat_id)
        scanners.append(scanner)
        return cast(MessageScanner, scanner)

    orchestrator = Orchestrator(bootstrap_result=result, scanner_factory=factory, enable_ui=False)
    await orchestrator._start_scanner(GAME_CHAT_ID)
    orchestrator._registered_player_ids.add(PLAYER_ID)
    result.game_state.register_player(PLAYER_ID, "Old")
    await orchestrator._handle_command("/reset")
    assert scanners[0].shutdown_called
    assert scanners[1].started
    assert result.game_state.game_id != old_game_id
    assert result.game_state.players() == ()
    assert not orchestrator._registered_player_ids
    assert "Reset: new game" in capsys.readouterr().out
    await orchestrator.shutdown()


async def test_handle_retro(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result, _, _ = await make_result(tmp_path)
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)
    retro = AsyncMock()
    orchestrator._run_retro = retro  # type: ignore[method-assign]
    await orchestrator._handle_command("/retro")
    retro.assert_awaited_once()
    assert "Retro: done" in capsys.readouterr().out
    retro.side_effect = RuntimeError("broken")
    await orchestrator._handle_command("/retro")
    assert "Retro failed: broken" in capsys.readouterr().out
    await orchestrator.shutdown()


# ---------------------------------------------------------------------------
# Part D: show_dead + status total via bot
# ---------------------------------------------------------------------------


async def test_send_stats_skips_dead_by_default(tmp_path: Path) -> None:
    """D. Мёртвые игроки не попадают в таблицу по умолчанию."""
    result, _telegram, _llm = await make_result(tmp_path)
    game_state = result.game_state
    alive_ids = (100, 101, 102)
    dead_ids = (103, 104)
    for player_id in alive_ids + dead_ids:
        game_state.register_player(player_id, f"Игрок {player_id}")
    for player_id in dead_ids:
        game_state.apply(
            DeathEvent(
                ts=EVENT_TS,
                game_id=game_state.game_id,
                payload=DeathPayload(
                    victim_id=player_id,
                    victim_name=f"Игрок {player_id}",
                    cause="lynch",
                ),
            )
        )
    assert result.config.analysis.show_dead is False
    ui = FakeWindowHost()
    orchestrator = Orchestrator(bootstrap_result=result, window_host=ui)  # type: ignore[arg-type]
    try:
        await orchestrator._send_stats()
        assert len(ui.stats[-1].rows) == 3
        assert {row.player_name for row in ui.stats[-1].rows} == {
            f"Игрок {player_id}" for player_id in alive_ids
        }
    finally:
        await orchestrator.shutdown()


async def test_send_stats_show_dead(tmp_path: Path) -> None:
    """D. show_dead=true показывает всех и помечает мёртвых."""
    result, _telegram, _llm = await make_result(tmp_path)
    result.config = result.config.model_copy(
        update={"analysis": result.config.analysis.model_copy(update={"show_dead": True})}
    )
    game_state = result.game_state
    for player_id in (100, 101, 102, 103, 104):
        game_state.register_player(player_id, f"Игрок {player_id}")
    for player_id in (103, 104):
        game_state.apply(
            DeathEvent(
                ts=EVENT_TS,
                game_id=game_state.game_id,
                payload=DeathPayload(
                    victim_id=player_id,
                    victim_name=f"Игрок {player_id}",
                    cause="lynch",
                ),
            )
        )
    result.repos.players.upsert_player(player_id=103, username=None, display_name="Игрок 103")
    result.repos.predictions.upsert_prediction(
        game_id=game_state.game_id,
        player_id=103,
        predicted_role="Мафия",
        confidence=0.8,
        reasoning="голосовал против дона",
    )
    ui = FakeWindowHost()
    orchestrator = Orchestrator(bootstrap_result=result, window_host=ui)  # type: ignore[arg-type]
    try:
        await orchestrator._send_stats()
        rows = ui.stats[-1].rows
        assert len(rows) == 5
        by_name = {row.player_name: row.role for row in rows}
        assert by_name["Игрок 103"] == "[мёртв] Мафия"
        assert by_name["Игрок 104"] == "[мёртв] —"
        assert by_name["Игрок 100"] == "—"
    finally:
        await orchestrator.shutdown()


async def test_status_total_uses_bot_when_consistent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D. total берётся у бота, пока он не меньше числа живых; иначе — живые."""
    result, _telegram, _llm = await make_result(tmp_path)
    game_state = result.game_state
    names = [f"Игрок {index}" for index in range(1, 8)]
    for index, name in enumerate(names, start=1):
        game_state.register_player(PLAYER_ID + index, name)
    game_state.apply(
        AliveListEvent(
            ts=EVENT_TS,
            game_id=game_state.game_id,
            payload=AliveListPayload(alive_player_names=names, total_players=10),
        )
    )
    orchestrator = Orchestrator(bootstrap_result=result)
    try:
        await orchestrator._handle_status()
        assert "alive=7/10" in capsys.readouterr().out

        game_state.apply(
            AliveListEvent(
                ts=EVENT_TS,
                game_id=game_state.game_id,
                payload=AliveListPayload(alive_player_names=names, total_players=4),
            )
        )
        await orchestrator._handle_status()
        assert "alive=7/7" in capsys.readouterr().out
    finally:
        await orchestrator.shutdown()


# ---------------------------------------------------------------------------
# Part E: bots whitelist via ALIVE_LIST
# ---------------------------------------------------------------------------


async def test_send_stats_whitelist(tmp_path: Path) -> None:
    """E. Utility-бот вне ALIVE_LIST не попадает в таблицу."""
    result, _telegram, _llm = await make_result(tmp_path)
    game_state = result.game_state
    for player_id, name in ((100, "A"), (101, "B"), (102, "EzzProd")):
        game_state.register_player(player_id, name)
    game_state.apply(
        AliveListEvent(
            ts=EVENT_TS,
            game_id=game_state.game_id,
            payload=AliveListPayload(alive_player_names=["A", "B"], total_players=2),
        )
    )
    ui = FakeWindowHost()
    orchestrator = Orchestrator(bootstrap_result=result, window_host=ui)  # type: ignore[arg-type]
    try:
        await orchestrator._send_stats()
        assert {row.player_name for row in ui.stats[-1].rows} == {"A", "B"}
    finally:
        await orchestrator.shutdown()


async def test_send_stats_no_whitelist(tmp_path: Path) -> None:
    """E. Пока ALIVE_LIST не приходил, показываются все real-игроки."""
    result, _telegram, _llm = await make_result(tmp_path)
    game_state = result.game_state
    for player_id, name in ((100, "A"), (101, "B"), (102, "C")):
        game_state.register_player(player_id, name)
    assert game_state.last_alive_list_names == frozenset()
    ui = FakeWindowHost()
    orchestrator = Orchestrator(bootstrap_result=result, window_host=ui)  # type: ignore[arg-type]
    try:
        await orchestrator._send_stats()
        assert {row.player_name for row in ui.stats[-1].rows} == {"A", "B", "C"}
    finally:
        await orchestrator.shutdown()


async def test_send_stats_whitelist_matches_username(tmp_path: Path) -> None:
    """E. ALIVE_LIST приходит display-именами; username тоже принимается."""
    result, _telegram, _llm = await make_result(tmp_path)
    game_state = result.game_state
    game_state.register_player(100, "Вася", username="vasya")
    game_state.register_player(101, "Петя", username="petya")
    game_state.apply(
        AliveListEvent(
            ts=EVENT_TS,
            game_id=game_state.game_id,
            payload=AliveListPayload(alive_player_names=["vasya", "Петя"], total_players=2),
        )
    )
    ui = FakeWindowHost()
    orchestrator = Orchestrator(bootstrap_result=result, window_host=ui)  # type: ignore[arg-type]
    try:
        await orchestrator._send_stats()
        assert len(ui.stats[-1].rows) == 2
    finally:
        await orchestrator.shutdown()


async def test_send_stats_shows_whitelisted_placeholder(tmp_path: Path) -> None:
    """Silent players from ALIVE_LIST get a row once the bot confirms them."""
    result, _telegram, _llm = await make_result(tmp_path)
    game_state = result.game_state
    game_state.register_player(PLAYER_ID, "A")
    game_state.apply(
        AliveListEvent(
            ts=EVENT_TS,
            game_id=game_state.game_id,
            payload=AliveListPayload(alive_player_names=["A", "| user_4"], total_players=2),
        )
    )
    ui = FakeWindowHost()
    orchestrator = Orchestrator(bootstrap_result=result, window_host=ui)  # type: ignore[arg-type]
    try:
        await orchestrator._send_stats()
        assert {row.player_name for row in ui.stats[-1].rows} == {"A", "| user_4"}
    finally:
        await orchestrator.shutdown()
