"""Tests for the headless command loop and event persistence."""

from __future__ import annotations

import asyncio
import io
import queue
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from app.bootstrap import BootstrapResult, bootstrap
from app.orchestrator import Orchestrator
from domain.events import MessageEvent, MessagePayload
from features.scanner.scanner import MessageScanner
from features.ui.protocol import ReasoningMessage, StatsMessage
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

    async def start(self) -> None:
        """Record scanner startup."""
        self.started = True

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


async def test_orchestrator_sends_stats_after_persist(tmp_path: Path) -> None:
    """A successful event insert produces a stats frame."""
    result, _telegram, _llm = await make_result(tmp_path)
    ui = FakeWindowHost()
    orchestrator = Orchestrator(
        bootstrap_result=result,
        window_host=ui,  # type: ignore[arg-type]
    )
    event = MessageEvent(
        ts=EVENT_TS,
        game_id=result.game_state.game_id,
        payload=MessagePayload(
            author_id=PLAYER_ID,
            author_name="Игрок",
            text="Сообщение для UI",
        ),
    )

    await orchestrator._persist_event(event)

    assert len(ui.stats) == 1
    assert ui.stats[0].game_id == result.game_state.game_id
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

    await scanner._flush()

    assert len(ui.reasoning) == 1
    assert "Проверяю" in ui.reasoning[0].text
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
