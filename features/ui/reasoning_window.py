"""Standalone child process that displays reasoning updates from stdin."""

from __future__ import annotations

import sys
from typing import TextIO

from pydantic import ValidationError
from rich.console import Console

from features.ui.protocol import ReasoningMessage, decode


def main(stream: TextIO | None = None) -> int:
    """Read IPC lines and render reasoning until shutdown or EOF."""
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
    if stdout_reconfigure is not None and stderr_reconfigure is not None:
        stdout_reconfigure(encoding="utf-8", errors="replace")
        stderr_reconfigure(encoding="utf-8", errors="replace")

    # Windows cmd.exe (CREATE_NEW_CONSOLE): chcp 65001
    if sys.platform == "win32":
        import os

        os.system("chcp 65001 > nul")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
