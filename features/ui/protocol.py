"""Typed JSON-lines protocol shared by the UI window processes."""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION: Final[str] = "1"


class IPCEnvelope(BaseModel):
    """Wire format for parent-to-child messages (one JSON line)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["stats", "reasoning", "clear", "shutdown"]
    payload: dict[str, Any] = Field(default_factory=dict)


class StatsRow(BaseModel):
    """One row in the stats table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    #: Short per-player explanation (e.g. "нет данных", "прямое заявление роли").
    reason: str = Field(default="")
    #: Fired club pattern IDs, e.g. ["#2"]. Empty when no pattern applies.
    pattern_ids: list[str] = Field(default_factory=list)
    #: Human-readable pattern descriptions in the same order as ``pattern_ids``.
    pattern_descriptions: list[str] = Field(default_factory=list)


class StatsMessage(BaseModel):
    """Payload for a stats frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    game_id: int
    phase: str
    day: int
    rows: list[StatsRow]


class ReasoningMessage(BaseModel):
    """Payload for one reasoning update."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    truncated: bool = False
    total_chars: int = Field(ge=0)


def encode(envelope: IPCEnvelope) -> str:
    """Encode an envelope as a single JSON line without its terminator."""
    return envelope.model_dump_json()


def decode(line: str) -> IPCEnvelope:
    """Decode one JSON line, normalizing malformed input to ``ValueError``."""
    try:
        return IPCEnvelope.model_validate_json(line)
    except Exception as error:
        raise ValueError(f"invalid IPC line: {error}") from error
