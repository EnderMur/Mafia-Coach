"""Standalone child process that displays reasoning updates from stdin."""

from __future__ import annotations

import sys
from typing import TextIO

from pydantic import ValidationError
from rich.console import Console

from features.ui.protocol import ReasoningMessage, decode


def main(stream: TextIO | None = None) -> int:
    """Read IPC lines and render reasoning until shutdown or EOF."""
    if stream is None:
        stream = sys.stdin
    console = Console()
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
