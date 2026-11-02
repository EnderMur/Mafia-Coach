"""Typed domain events and their payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from domain.phases import Phase


@runtime_checkable
class DomainEvent(Protocol):
    """Structural protocol shared by all game events."""

    ts: datetime
    kind: str
    game_id: int


class MessagePayload(BaseModel):
    """Message content and its author."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    author_id: int
    author_name: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DeathPayload(BaseModel):
    """Player death details."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    victim_id: int
    victim_name: str = Field(min_length=1)
    cause: Literal["night_kill", "lynch", "unknown"] = "unknown"


class VotePayload(BaseModel):
    """A vote and optional target details.

    Target id and name are independent so partial bot messages can be preserved.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    voter_id: int
    voter_name: str = Field(min_length=1)
    target_id: int | None
    target_name: str | None


class RoleRevealPayload(BaseModel):
    """A publicly revealed role name as written by the game bot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: int
    player_name: str = Field(min_length=1)
    role_name_ru: str = Field(min_length=1)


class PhaseChangePayload(BaseModel):
    """A game phase transition and its day number."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_phase: Phase
    to_phase: Phase
    day_number: int = Field(ge=0)


class AliveListPayload(BaseModel):
    """Bot's snapshot of the alive players and role composition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alive_player_names: list[str] = Field(min_length=1)
    #: ``name_ru → count`` exactly as written by the bot.
    role_count: dict[str, int] | None = None
    total_players: int | None = Field(default=None, ge=0)


class EndGamePayload(BaseModel):
    """Bot's end-of-game report: winners + losers + duration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: ``[(player_name, role_name_ru), ...]`` — победители (достигли цели,
    #: не обязательно выжили: самоубийца повешен — победил).
    winners: list[tuple[str, str]] = Field(default_factory=list)
    #: ``[(player_name, role_name_ru), ...]`` — остальные участники.
    losers: list[tuple[str, str]] = Field(default_factory=list)
    duration_minutes: int | None = Field(default=None, ge=0)


class _EventModel(BaseModel):
    """Common immutable event fields and timestamp validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: datetime
    game_id: int

    @field_validator("ts")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Reject timestamps without timezone information."""
        if value.tzinfo is None:
            raise ValueError("event timestamp must be timezone-aware")
        return value


class MessageEvent(_EventModel):
    """An observed message event."""

    kind: Literal["message"] = "message"
    payload: MessagePayload


class DeathEvent(_EventModel):
    """A player death event."""

    kind: Literal["death"] = "death"
    payload: DeathPayload


class VoteEvent(_EventModel):
    """A player vote event."""

    kind: Literal["vote"] = "vote"
    payload: VotePayload


class RoleRevealEvent(_EventModel):
    """A public role reveal event."""

    kind: Literal["role_reveal"] = "role_reveal"
    payload: RoleRevealPayload


class PhaseChangeEvent(_EventModel):
    """A game phase transition event."""

    kind: Literal["phase_change"] = "phase_change"
    payload: PhaseChangePayload


class AliveListEvent(_EventModel):
    """Bot reports 'Живые игроки: ... Из них: ... Всего: N'."""

    kind: Literal["alive_list"] = "alive_list"
    payload: AliveListPayload


class EndGameEvent(_EventModel):
    """Bot reports the final standings: 'Игра окончена!'."""

    kind: Literal["end_game"] = "end_game"
    payload: EndGamePayload


AnyEvent: TypeAlias = (
    MessageEvent
    | DeathEvent
    | VoteEvent
    | RoleRevealEvent
    | PhaseChangeEvent
    | AliveListEvent
    | EndGameEvent
)
