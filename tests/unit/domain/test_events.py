"""Tests for domain event payloads and models."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from domain.events import (
    AnyEvent,
    DeathEvent,
    DeathPayload,
    DomainEvent,
    MessageEvent,
    MessagePayload,
    PhaseChangeEvent,
    PhaseChangePayload,
    RoleRevealEvent,
    RoleRevealPayload,
    VoteEvent,
    VotePayload,
)
from domain.phases import Phase


def timestamp() -> datetime:
    """Return a timezone-aware event timestamp."""
    return datetime(2025, 1, 1, tzinfo=timezone.utc)


def make_events() -> tuple[AnyEvent, ...]:
    """Construct one valid instance of each supported event model."""
    return (
        MessageEvent(
            ts=timestamp(),
            game_id=1,
            payload=MessagePayload(author_id=1, author_name="Alice", text="Hello"),
        ),
        DeathEvent(
            ts=timestamp(),
            game_id=1,
            payload=DeathPayload(victim_id=2, victim_name="Bob", cause="lynch"),
        ),
        VoteEvent(
            ts=timestamp(),
            game_id=1,
            payload=VotePayload(
                voter_id=1,
                voter_name="Alice",
                target_id=2,
                target_name="Bob",
            ),
        ),
        RoleRevealEvent(
            ts=timestamp(),
            game_id=1,
            payload=RoleRevealPayload(player_id=1, player_name="Alice", role_name_ru="Role"),
        ),
        PhaseChangeEvent(
            ts=timestamp(),
            game_id=1,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION,
                to_phase=Phase.NIGHT,
                day_number=1,
            ),
        ),
    )


def test_all_payload_and_event_models_construct() -> None:
    """Every payload and event model accepts valid values."""
    message = MessagePayload(author_id=1, author_name="Alice", text="Hello")
    death = DeathPayload(victim_id=2, victim_name="Bob")
    vote = VotePayload(voter_id=1, voter_name="Alice", target_id=None, target_name=None)
    reveal = RoleRevealPayload(player_id=1, player_name="Alice", role_name_ru="Role")
    phase = PhaseChangePayload(from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=1)
    assert message.text == "Hello"
    assert death.cause == "unknown"
    assert vote.target_id is None
    assert reveal.role_name_ru == "Role"
    assert phase.day_number == 1
    assert len(make_events()) == 5


def test_message_payload_rejects_empty_text() -> None:
    """A message body must not be empty."""
    with pytest.raises(ValidationError):
        MessagePayload(author_id=1, author_name="Alice", text="")


def test_death_payload_rejects_unknown_cause() -> None:
    """Death causes are restricted to the declared literal values."""
    with pytest.raises(ValidationError):
        DeathPayload(victim_id=2, victim_name="Bob", cause="magic")  # type: ignore[arg-type]


def test_vote_payload_allows_abstention() -> None:
    """A vote may have no target when the voter abstained."""
    vote = VotePayload(voter_id=1, voter_name="Alice", target_id=None, target_name=None)
    assert vote.target_id is None
    assert vote.target_name is None


def test_vote_payload_allows_partial_target_details() -> None:
    """A target id may be retained when its display name is missing."""
    vote = VotePayload(voter_id=1, voter_name="Alice", target_id=2, target_name=None)
    assert vote.target_id == 2
    assert vote.target_name is None


def test_role_reveal_payload_rejects_empty_role_name() -> None:
    """A role reveal must preserve a non-empty bot role name."""
    with pytest.raises(ValidationError):
        RoleRevealPayload(player_id=1, player_name="Alice", role_name_ru="")


def test_phase_change_payload_allows_same_phase() -> None:
    """Phase equality is allowed by the payload and checked by GameState."""
    payload = PhaseChangePayload(from_phase=Phase.NIGHT, to_phase=Phase.NIGHT, day_number=1)
    assert payload.from_phase == payload.to_phase


def test_event_models_are_frozen() -> None:
    """Event attributes cannot be reassigned."""
    event = make_events()[0]
    with pytest.raises(ValidationError):
        event.game_id = 2


def test_events_reject_naive_timestamps() -> None:
    """Events require timezone-aware timestamps."""
    with pytest.raises(ValidationError):
        MessageEvent(
            ts=datetime(2025, 1, 1),
            game_id=1,
            payload=MessagePayload(author_id=1, author_name="Alice", text="Hello"),
        )


@pytest.mark.parametrize("event", make_events())
def test_event_models_satisfy_domain_event_protocol(event: AnyEvent) -> None:
    """Each supported event structurally implements DomainEvent."""
    assert isinstance(event, DomainEvent)
