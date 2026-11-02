"""Standalone child process that renders stats frames from stdin."""

from __future__ import annotations

import ctypes
import os
import sys
from typing import TextIO

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from features.ui.protocol import IPCEnvelope, StatsMessage, decode

_HEADER = "Stats"
_MAX_ROWS = 20
_WINDOW_NAME = "stats_window"


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


def _safe_clear(console: Console) -> None:
    """Clear the screen, falling back to blank lines when unsupported."""
    try:
        console.clear()
    except Exception:  # noqa: BLE001
        # Fallback: 25 пустых строк
        for _ in range(25):
            console.print("", highlight=False)


def render_table(console: Console, msg: StatsMessage) -> None:
    """Render one stats frame."""
    _safe_clear(console)
    table = Table(
        title=f"{_HEADER} — партия #{msg.game_id}, {msg.phase}, день {msg.day}",
        show_lines=False,
    )
    table.add_column("Игрок", style="cyan", no_wrap=True)
    table.add_column("Роль", style="magenta")
    table.add_column("Уверенность", justify="right", style="green")
    for row in msg.rows[:_MAX_ROWS]:
        table.add_row(row.player_name, row.role, f"{int(row.confidence * 100)}%")
    console.print(table)


def main(stream: TextIO | None = None) -> int:
    """Read IPC lines and render until shutdown or EOF."""
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
    console = Console(
        force_terminal=True,
        legacy_windows=True,
        color_system="standard",
        width=120,
    )
    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        try:
            env: IPCEnvelope = decode(line)
        except ValueError:
            continue
        if env.type == "shutdown":
            return 0
        if env.type == "stats":
            try:
                msg = StatsMessage.model_validate(env.payload)
            except ValidationError:
                continue
            render_table(console, msg)
            sys.stdout.flush()  # не буферизуем в pipe
    return 0


if __name__ == "__main__":
    sys.exit(main())
