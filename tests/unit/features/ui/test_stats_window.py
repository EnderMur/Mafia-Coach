"""Stats-window rendering and stdin-loop tests."""

from __future__ import annotations

import io

from rich.console import Console

from features.ui.protocol import IPCEnvelope, StatsMessage, StatsRow, encode
from features.ui.stats_window import main, render_table


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
