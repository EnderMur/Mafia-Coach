"""Standalone child process that renders stats frames from stdin."""

from __future__ import annotations

import sys
from typing import TextIO

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from features.ui.protocol import IPCEnvelope, StatsMessage, decode

_HEADER = "Stats"
_MAX_ROWS = 20


def render_table(console: Console, msg: StatsMessage) -> None:
    """Render one stats frame."""
    console.clear()
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
    if stream is None:
        stream = sys.stdin
    console = Console()
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
