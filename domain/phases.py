"""Game phases recognized by the domain."""

from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    """A phase in a Mafia game."""

    REGISTRATION = "registration"
    NIGHT = "night"
    DAY = "day"
    VOTE = "vote"
    END = "end"
    UNKNOWN = "unknown"
