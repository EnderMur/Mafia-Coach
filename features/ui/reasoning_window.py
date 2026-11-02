"""Standalone child process that displays reasoning updates from stdin."""

from __future__ import annotations

import ctypes
import os
import sys
from typing import TextIO

from pydantic import ValidationError
from rich.console import Console

from features.ui.protocol import ReasoningMessage, decode

_WINDOW_NAME = "reasoning_window"


def _reopen_windows_console() -> None:
    """Reopen std streams on the console created by CREATE_NEW_CONSOLE.

    The parent launches this child with ``stdout=DEVNULL``, so ``sys.stdout``
    points at NUL. ``CONOUT$`` is the active console — the freshly created
    window — and reopening it lets ``rich.Console`` draw there.
    """
    try:
        sys.stdout = open(  # noqa: SIM115 — lifetime = process
            "CONOUT$",
            "w",
            encoding="utf-8",
            buffering=1,
        )
        sys.stderr = open(  # noqa: SIM115 — lifetime = process
            "CONOUT$",
            "w",
            encoding="utf-8",
            buffering=1,
        )
    except OSError:
        pass  # fallback: остаёмся на DEVNULL


def _log_startup() -> None:
    """Announce the child pid so the parent log proves the process is alive."""
    sys.stderr.write(f"[{_WINDOW_NAME}] starting pid={os.getpid()}\n")
    sys.stderr.flush()


def main(stream: TextIO | None = None) -> int:
    """Read IPC lines and render reasoning until shutdown or EOF."""
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
    if stdout_reconfigure is not None and stderr_reconfigure is not None:
        stdout_reconfigure(encoding="utf-8", errors="replace")
        stderr_reconfigure(encoding="utf-8", errors="replace")

    # Windows: set console output/input code page to UTF-8 for THIS
    # process (not a subshell). A chcp command in a child cmd.exe
    # changes cp only there and is lost when that subshell exits.
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except (AttributeError, OSError):
            pass  # нет kernel32/консоли — остаёмся на текущей cp
        _reopen_windows_console()

    # Startup debug: подтверждение, что child жив и открыл консоль
    _log_startup()
    sys.stdout.flush()  # новый консольный поток — сброс до первого кадра

    if stream is None:
        stream = sys.stdin
    console = Console(force_terminal=True, legacy_windows=True)
    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        try:
            env = decode(line)
        except ValueError:
            continue
        if env.type == "shutdown":
            return 0
        if env.type == "reasoning":
            try:
                msg = ReasoningMessage.model_validate(env.payload)
            except ValidationError:
                continue
            marker = " [dim](усечено)[/dim]" if msg.truncated else ""
            console.print(f"[cyan]{msg.text}[/cyan]{marker}")
            sys.stdout.flush()  # не буферизуем в pipe
    return 0


if __name__ == "__main__":
    sys.exit(main())
