"""Tests for event persistence and reconstruction."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from domain.events import (
    AnyEvent,
    DeathEvent,
    DeathPayload,
    MessageEvent,
    MessagePayload,
    PhaseChangeEvent,
    PhaseChangePayload,
    RoleRevealEvent,
    RoleRevealPayload,
    VoteEvent,
    VotePayload,
)
from domain.exceptions import ConfigError
from domain.phases import Phase
from infra.db.repositories.events import EventsRepository
from infra.db.repositories.games import GamesRepository


def create_game(conn: sqlite3.Connection) -> int:
    """Create and return a game id for event foreign keys."""
    return GamesRepository(conn).create_game(chat_id=1, player_count=10)


def timestamp() -> datetime:
    """Return a timezone-aware event timestamp."""
    return datetime(2025, 1, 1, 12, 30, tzinfo=timezone.utc)


def make_message(game_id: int, text: str = "Hello") -> MessageEvent:
    """Create a message event fixture."""
    return MessageEvent(
        ts=timestamp(),
        game_id=game_id,
        payload=MessagePayload(author_id=100, author_name="Alice", text=text),
    )


def test_message_event_round_trips_as_domain_event(conn: sqlite3.Connection) -> None:
    """Message payload JSON reconstructs the equivalent event model."""
    game_id = create_game(conn)
    repository = EventsRepository(conn)
    event = make_message(game_id)
    inserted_id = repository.insert_event(event)
    restored = repository.list_events(game_id)
    assert inserted_id > 0
    assert len(restored) == 1
    assert restored[0].model_dump() == event.model_dump()


def test_death_event_stores_victim_as_actor(conn: sqlite3.Connection) -> None:
    """Death rows store the victim id in actor_id."""
    game_id = create_game(conn)
    event = DeathEvent(
        ts=timestamp(),
        game_id=game_id,
        payload=DeathPayload(victim_id=102, victim_name="Bob", cause="night_kill"),
    )
    EventsRepository(conn).insert_event(event)
    row = conn.execute("SELECT actor_id FROM events").fetchone()
    assert row["actor_id"] == 102


def test_vote_event_stores_target_id(conn: sqlite3.Connection) -> None:
    """Vote rows retain both the voter and target ids."""
    game_id = create_game(conn)
    event = VoteEvent(
        ts=timestamp(),
        game_id=game_id,
        payload=VotePayload(voter_id=100, voter_name="Alice", target_id=102, target_name="Bob"),
    )
    EventsRepository(conn).insert_event(event)
    row = conn.execute("SELECT actor_id, target_id FROM events").fetchone()
    assert row["actor_id"] == 100
    assert row["target_id"] == 102


def test_abstention_vote_stores_null_target(conn: sqlite3.Connection) -> None:
    """Abstention votes store SQL NULL for target_id."""
    game_id = create_game(conn)
    event = VoteEvent(
        ts=timestamp(),
        game_id=game_id,
        payload=VotePayload(voter_id=100, voter_name="Alice", target_id=None, target_name=None),
    )
    EventsRepository(conn).insert_event(event)
    row = conn.execute("SELECT target_id FROM events").fetchone()
    assert row["target_id"] is None


def test_role_reveal_stores_player_as_actor(conn: sqlite3.Connection) -> None:
    """Role reveal rows store the revealed player id in actor_id."""
    game_id = create_game(conn)
    event = RoleRevealEvent(
        ts=timestamp(),
        game_id=game_id,
        payload=RoleRevealPayload(player_id=103, player_name="Carol", role_name_ru="Role"),
    )
    EventsRepository(conn).insert_event(event)
    row = conn.execute("SELECT actor_id FROM events").fetchone()
    assert row["actor_id"] == 103


def test_phase_change_stores_phase_and_day_number(conn: sqlite3.Connection) -> None:
    """Phase transition metadata is extracted into dedicated columns."""
    game_id = create_game(conn)
    event = PhaseChangeEvent(
        ts=timestamp(),
        game_id=game_id,
        payload=PhaseChangePayload(from_phase=Phase.DAY, to_phase=Phase.VOTE, day_number=3),
    )
    repository = EventsRepository(conn)
    repository.insert_event(event)
    row = conn.execute("SELECT phase, day_number FROM events").fetchone()
    assert row["phase"] == "vote"
    assert row["day_number"] == 3
    assert repository.list_events(game_id)[0].model_dump() == event.model_dump()


def test_count_by_kind_groups_event_rows(conn: sqlite3.Connection) -> None:
    """Counts are returned for each persisted event kind."""
    game_id = create_game(conn)
    repository = EventsRepository(conn)
    repository.insert_event(make_message(game_id, "one"))
    repository.insert_event(make_message(game_id, "two"))
    repository.insert_event(
        DeathEvent(
            ts=timestamp(),
            game_id=game_id,
            payload=DeathPayload(victim_id=102, victim_name="Bob"),
        )
    )
    assert repository.count_by_kind(game_id) == {"death": 1, "message": 2}


def test_non_message_event_types_round_trip(conn: sqlite3.Connection) -> None:
    """Death, vote, and role reveal event payloads reconstruct losslessly."""
    game_id = create_game(conn)
    repository = EventsRepository(conn)
    events: list[AnyEvent] = [
        DeathEvent(
            ts=timestamp(),
            game_id=game_id,
            payload=DeathPayload(victim_id=102, victim_name="Bob", cause="lynch"),
        ),
        VoteEvent(
            ts=timestamp(),
            game_id=game_id,
            payload=VotePayload(voter_id=100, voter_name="Alice", target_id=102, target_name="Bob"),
        ),
        RoleRevealEvent(
            ts=timestamp(),
            game_id=game_id,
            payload=RoleRevealPayload(player_id=102, player_name="Bob", role_name_ru="Role"),
        ),
    ]
    for event in events:
        repository.insert_event(event)
    assert [event.model_dump() for event in repository.list_events(game_id)] == [
        event.model_dump() for event in events
    ]


def test_event_insert_preserves_foreign_key_errors(conn: sqlite3.Connection) -> None:
    """Events for absent games are rejected by SQLite foreign keys."""
    event = make_message(999)
    with pytest.raises(sqlite3.IntegrityError):
        EventsRepository(conn).insert_event(event)


def test_unknown_persisted_kind_raises_config_error(conn: sqlite3.Connection) -> None:
    """Unknown row kinds are not silently decoded as domain events."""
    game_id = create_game(conn)
    conn.execute(
        "INSERT INTO events(game_id, kind, payload_json) VALUES (?, ?, ?)",
        (game_id, "future_kind", "{}"),
    )
    with pytest.raises(ConfigError, match="unknown event kind: future_kind"):
        EventsRepository(conn).list_events(game_id)
