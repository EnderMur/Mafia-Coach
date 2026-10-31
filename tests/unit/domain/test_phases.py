"""Tests for game phases."""

from __future__ import annotations

import pytest

from domain.phases import Phase


@pytest.mark.parametrize(
    ("phase", "value"),
    [
        (Phase.REGISTRATION, "registration"),
        (Phase.NIGHT, "night"),
        (Phase.DAY, "day"),
        (Phase.VOTE, "vote"),
        (Phase.END, "end"),
        (Phase.UNKNOWN, "unknown"),
    ],
)
def test_phase_values(phase: Phase, value: str) -> None:
    """Every phase exposes its contract value."""
    assert phase.value == value


def test_phase_from_night_string() -> None:
    """Phase values can be parsed from strings."""
    assert Phase("night") == Phase.NIGHT


def test_phase_from_unknown_string() -> None:
    """The unknown phase is available as a value."""
    assert Phase("unknown") == Phase.UNKNOWN
