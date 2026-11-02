"""Standalone window CLI tests with no real subprocesses or blocking input."""

from __future__ import annotations

import importlib
from unittest.mock import Mock, patch

import pytest

from features.ui.window_host import WindowHost

run_windows = importlib.import_module("scripts.run_windows")


def test_run_windows_no_both_disabled(capsys: pytest.CaptureFixture[str]) -> None:
    """Disabling both windows fails before constructing a host."""
    with patch("scripts.run_windows.WindowHost") as constructor:
        assert run_windows.main(["--no-stats", "--no-reasoning"]) == 1

    constructor.assert_not_called()
    assert "Nothing to run: both windows disabled." in capsys.readouterr().out


def test_run_windows_starts_and_stops(capsys: pytest.CaptureFixture[str]) -> None:
    """EOF stops both default windows after a complete start/shutdown lifecycle."""
    host = Mock(spec=WindowHost)
    host.stats_running = True
    host.stats_pid = 101
    host.reasoning_running = True
    host.reasoning_pid = 102
    with (
        patch("scripts.run_windows.WindowHost", return_value=host) as constructor,
        patch("sys.stdin.readline", return_value="") as readline,
    ):
        assert run_windows.main([]) == 0

    constructor.assert_called_once_with(stats_enabled=True, reasoning_enabled=True)
    host.start.assert_awaited_once()
    host.shutdown.assert_awaited_once()
    readline.assert_called_once()
    output = capsys.readouterr().out
    assert "stats=True (pid=101), reasoning=True (pid=102)" in output
    assert "Windows stopped." in output


def test_run_windows_only_stats() -> None:
    """The reasoning flag disables only the reasoning child."""
    host = Mock(spec=WindowHost)
    with (
        patch("scripts.run_windows.WindowHost", return_value=host) as constructor,
        patch("sys.stdin.readline", return_value=""),
    ):
        assert run_windows.main(["--no-reasoning"]) == 0

    constructor.assert_called_once_with(stats_enabled=True, reasoning_enabled=False)
    host.start.assert_awaited_once()
    host.shutdown.assert_awaited_once()
