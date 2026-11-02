"""Reasoning-window rendering and stdin-loop tests."""

from __future__ import annotations

import builtins
import io
import os
import sys
from typing import Any

import pytest

from features.ui import reasoning_window as reasoning_window_module
from features.ui.protocol import IPCEnvelope, ReasoningMessage, encode
from features.ui.reasoning_window import main

# The autouse conftest fixture stubs this out on Windows; the CONOUT$ test
# needs the real implementation, so keep a reference captured at import time.
_REAL_REOPEN_WINDOWS_CONSOLE = reasoning_window_module._reopen_windows_console


def _line(text: str, *, truncated: bool = False) -> str:
    """Encode one reasoning envelope for the child loop."""
    return encode(
        IPCEnvelope(
            type="reasoning",
            payload=ReasoningMessage(
                text=text,
                truncated=truncated,
                total_chars=len(text),
            ).model_dump(),
        )
    )


def test_main_renders_three_reasoning_envelopes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each valid reasoning frame is printed to the output window."""
    content = "\n".join((_line("Первое"), _line("Второе"), _line("Третье"), ""))

    assert main(io.StringIO(content)) == 0
    output = capsys.readouterr().out
    assert all(text in output for text in ("Первое", "Второе", "Третье"))


def test_main_marks_truncated_reasoning(capsys: pytest.CaptureFixture[str]) -> None:
    """Truncated content is labeled for the user."""
    assert main(io.StringIO(_line("Сокращённый текст", truncated=True) + "\n")) == 0

    assert "усечено" in capsys.readouterr().out


def test_main_exits_on_shutdown() -> None:
    """Shutdown causes a clean exit code."""
    assert main(io.StringIO(encode(IPCEnvelope(type="shutdown")) + "\n")) == 0


def test_main_skips_invalid_line() -> None:
    """Invalid lines are ignored."""
    content = "not json\n" + encode(IPCEnvelope(type="shutdown")) + "\n"

    assert main(io.StringIO(content)) == 0


class _FlushSpy(io.StringIO):
    """StringIO that counts flush() calls made by the child loop."""

    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def test_main_reports_startup_pid_and_flushes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The child announces its pid and flushes after every rendered frame."""
    spy = _FlushSpy()
    monkeypatch.setattr(sys, "stdout", spy)

    assert main(io.StringIO(_line("Старт") + "\n")) == 0
    assert "starting pid=" in capsys.readouterr().err
    assert "Старт" in spy.getvalue()
    assert spy.flush_count >= 2


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
    monkeypatch.setattr(os, "system", lambda command: 0)  # chcp on POSIX
    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(
        reasoning_window_module, "_reopen_windows_console", _REAL_REOPEN_WINDOWS_CONSOLE
    )

    assert (
        main(io.StringIO(_line("Окно") + "\n" + encode(IPCEnvelope(type="shutdown")) + "\n")) == 0
    )
    assert opened.count("CONOUT$") == 2
    assert "starting pid=" in console_stream.getvalue()
    assert "Окно" in console_stream.getvalue()
