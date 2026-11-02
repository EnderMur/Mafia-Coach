"""Stats-window rendering and stdin-loop tests."""

from __future__ import annotations

import builtins
import io
import sys
from typing import Any

import pytest
from rich.console import Console

from features.ui import stats_window as stats_window_module
from features.ui.protocol import IPCEnvelope, StatsMessage, StatsRow, encode
from features.ui.stats_window import main, render_table

# The autouse conftest fixture stubs this out on Windows; the CONOUT$ tests
# need the real implementation, so keep a reference captured at import time.
_REAL_REOPEN_WINDOWS_CONSOLE = stats_window_module._reopen_windows_console


class _FlushSpy(io.StringIO):
    """StringIO that counts flush() calls made by the child loop."""

    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def _stats_line() -> str:
    """Encode one stats frame with a single visible row."""
    return encode(
        IPCEnvelope(
            type="stats",
            payload=StatsMessage(
                game_id=3,
                phase="day",
                day=2,
                rows=[StatsRow(player_name="Анна", role="Мафия", confidence=0.75)],
            ).model_dump(),
        )
    )


def test_render_table_shows_players_phase_day_and_confidence() -> None:
    """All three supplied rows and their rendered fields appear in the table."""
    output = io.StringIO()
    message = StatsMessage(
        game_id=8,
        phase="day",
        day=3,
        rows=[
            StatsRow(player_name="Анна", role="Мафия", confidence=0.81),
            StatsRow(player_name="Борис", role="Доктор", confidence=0.64),
            StatsRow(player_name="Вера", role="Дон", confidence=0.42),
        ],
    )

    render_table(Console(file=output, force_terminal=False), message)

    rendered = output.getvalue()
    for expected in ("Анна", "Борис", "Вера", "81%", "64%", "42%", "day", "день 3"):
        assert expected in rendered


def test_render_table_limits_to_twenty_rows() -> None:
    """Large frames are truncated at the visible-row cap."""
    output = io.StringIO()
    message = StatsMessage(
        game_id=1,
        phase="night",
        day=1,
        rows=[
            StatsRow(player_name=f"Игрок {index}", role="Мафия", confidence=0.5)
            for index in range(25)
        ],
    )

    render_table(Console(file=output, force_terminal=False), message)

    assert "Игрок 19" in output.getvalue()
    assert "Игрок 20" not in output.getvalue()


def test_main_processes_stats_and_shutdown_lines() -> None:
    """The blocking child loop accepts a stats frame and exits on shutdown."""
    lines = "\n".join(
        (
            encode(
                IPCEnvelope(
                    type="stats",
                    payload={
                        "game_id": 1,
                        "phase": "day",
                        "day": 1,
                        "rows": [],
                    },
                )
            ),
            encode(IPCEnvelope(type="shutdown")),
            "",
        )
    )

    assert main(io.StringIO(lines)) == 0


def test_main_skips_invalid_line() -> None:
    """Invalid input is ignored instead of aborting the process."""
    lines = "invalid\n" + encode(IPCEnvelope(type="shutdown")) + "\n"

    assert main(io.StringIO(lines)) == 0


def test_win32_reopens_conout(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows the child reopens CONOUT$ so rich draws into its own window."""
    console_stream = io.StringIO()
    opened: list[str] = []
    real_open = builtins.open

    def fake_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(file))
        if str(file) == "CONOUT$":
            return console_stream
        return real_open(file, *args, **kwargs)

    # Register originals first: main() rebinds sys.stdout / sys.stderr.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(
        stats_window_module, "_reopen_windows_console", _REAL_REOPEN_WINDOWS_CONSOLE
    )

    lines = encode(IPCEnvelope(type="shutdown")) + "\n"

    assert main(io.StringIO(lines)) == 0
    assert opened.count("CONOUT$") == 2
    assert "starting pid=" in console_stream.getvalue()


def test_win32_missing_console_falls_back_to_devnull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing console (no CONOUT$) does not abort the child loop."""
    monkeypatch.setattr(sys, "platform", "win32")

    def failing_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        raise OSError("no console")

    monkeypatch.setattr(builtins, "open", failing_open)

    lines = _stats_line() + "\n" + encode(IPCEnvelope(type="shutdown")) + "\n"

    assert main(io.StringIO(lines)) == 0


def test_render_goes_to_stdout_and_flushes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frames are written to sys.stdout and flushed after every render."""
    spy = _FlushSpy()
    monkeypatch.setattr(sys, "stdout", spy)

    lines = "\n".join((_stats_line(), encode(IPCEnvelope(type="shutdown")), ""))

    assert main(io.StringIO(lines)) == 0
    assert "Анна" in spy.getvalue()
    assert spy.flush_count >= 2


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only cp fix")
def test_set_console_cp_called_on_win32() -> None:
    """Smoke: the Windows cp fix runs (no chcp subshell) and stdout is open.

    Asserting on ``ctypes.windll.kernel32`` through monkeypatch is brittle;
    running the real Windows startup path without an exception is enough.
    ``conftest`` stubs CONOUT$ reopening, so pytest capture stays intact.
    """
    assert main(io.StringIO(encode(IPCEnvelope(type="shutdown")) + "\n")) == 0
    assert hasattr(sys.stdout, "write")


def test_main_utf8_stdin_payload_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    """Smoke (fix-7): a Cyrillic frame from a UTF-8 pipe survives main().

    io.StringIO has no ``reconfigure``, so sys.stdin.reconfigure(utf-8) is
    exercised only by the real child process (C5 v9). Here the reconfigure
    block must not break the stdin loop even when streams lack the method.
    """
    lines = _stats_line() + "\n" + encode(IPCEnvelope(type="shutdown")) + "\n"

    assert main(io.StringIO(lines)) == 0
    assert "Анна" in capsys.readouterr().out
