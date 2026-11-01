"""Lifecycle and IPC tests for WindowHost using tiny Python child processes."""

from __future__ import annotations

import asyncio
import sys
from typing import cast

import pytest

from features.ui.protocol import (
    IPCEnvelope,
    ReasoningMessage,
    StatsMessage,
    StatsRow,
    encode,
)
from features.ui.window_host import WindowHost, _auto_create_console


def _reader_command() -> tuple[str, ...]:
    """Return a Python -c child that exits after receiving shutdown."""
    code = "import sys; [None for line in sys.stdin if '\"shutdown\"' in line]"
    return (sys.executable, "-c", code)


async def test_start_launches_enabled_child() -> None:
    """Enabled children receive real process ids while disabled ones stay absent."""
    host = WindowHost(
        stats_enabled=True,
        reasoning_enabled=False,
        create_new_console=False,
        stats_command=_reader_command(),
    )

    await host.start()

    assert host.stats_running
    assert not host.reasoning_running
    assert host.stats_pid is not None
    await host.shutdown()


async def test_start_skips_disabled_children() -> None:
    """Disabled windows do not create subprocesses."""
    host = WindowHost(
        stats_enabled=False,
        reasoning_enabled=False,
        create_new_console=False,
    )

    await host.start()

    assert not host.stats_running
    assert not host.reasoning_running
    assert host.stats_pid is None
    assert host.reasoning_pid is None
    await host.shutdown()


class _FakeWriter:
    def __init__(self) -> None:
        self.data: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.data.append(data)

    async def drain(self) -> None:
        return None


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.pid = 456
        self.stdin = _FakeWriter()


async def test_send_stats_writes_one_json_line() -> None:
    """Stats data is encoded and written with exactly one line terminator."""
    host = WindowHost(create_new_console=False)
    process = _FakeProcess()
    host._processes["stats"] = cast(asyncio.subprocess.Process, process)
    message = StatsMessage(
        game_id=7,
        phase="day",
        day=2,
        rows=[StatsRow(player_name="Игрок", role="Мафия", confidence=0.8)],
    )

    await host.send_stats(message)

    assert process.stdin.data == [
        (encode(IPCEnvelope(type="stats", payload=message.model_dump())) + "\n").encode("utf-8")
    ]


async def test_send_to_dead_process_is_noop() -> None:
    """Closed children do not make UI calls raise."""
    host = WindowHost(create_new_console=False, stats_command=_reader_command())
    await host.start()
    await host.shutdown()

    await host.send_stats(StatsMessage(game_id=1, phase="day", day=1, rows=[]))


def test_send_reasoning_works_with_live_child() -> None:
    """Reasoning frames use the same newline-delimited pipe protocol."""

    async def run() -> None:
        host = WindowHost(
            stats_enabled=False,
            reasoning_enabled=True,
            create_new_console=False,
            reasoning_command=_reader_command(),
        )
        await host.start()
        await host.send_reasoning(ReasoningMessage(text="текст", total_chars=5))
        assert host.reasoning_running
        await host.shutdown()

    asyncio.run(run())


async def test_shutdown_terminates_child_that_ignores_stdin() -> None:
    """A child which ignores the shutdown envelope is terminated after timeout."""
    host = WindowHost(
        stats_enabled=True,
        reasoning_enabled=False,
        create_new_console=False,
        stats_command=(sys.executable, "-c", "import time; time.sleep(30)"),
    )
    await host.start()
    proc = host._processes["stats"]
    assert proc is not None

    await host.shutdown(timeout=0.1)

    assert proc.returncode is not None
    assert not host.stats_running


async def test_shutdown_is_idempotent() -> None:
    """Repeated shutdown is harmless."""
    host = WindowHost(
        stats_enabled=True,
        reasoning_enabled=False,
        create_new_console=False,
        stats_command=_reader_command(),
    )
    await host.start()

    await host.shutdown()
    await host.shutdown()


def test_auto_create_console_uses_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Platform-specific default is deterministic."""
    monkeypatch.setattr("features.ui.window_host.sys.platform", "win32")
    assert _auto_create_console() is True
    monkeypatch.setattr("features.ui.window_host.sys.platform", "linux")
    assert _auto_create_console() is False
