"""Tests for the players repository."""

from __future__ import annotations

import sqlite3

import pytest

from domain.exceptions import ConfigError
from infra.db.repositories.players import PlayersRepository


def test_upsert_player_creates_retrievable_row(conn: sqlite3.Connection) -> None:
    """New player details are stored and retrievable."""
    repository = PlayersRepository(conn)
    repository.upsert_player(player_id=100, username="alice", display_name="Alice")
    row = repository.get_player(100)
    assert row is not None
    assert row["id"] == 100
    assert row["username"] == "alice"


def test_upsert_refreshes_username_without_changing_first_seen(
    conn: sqlite3.Connection,
) -> None:
    """Upsert preserves first-seen time and updates username and last-seen."""
    repository = PlayersRepository(conn)
    repository.upsert_player(player_id=100, username="old", display_name="Alice")
    original = repository.get_player(100)
    assert original is not None
    first_seen = original["first_seen_at"]
    conn.execute("UPDATE players SET last_seen_at='2000-01-01 00:00:00' WHERE id=100")
    repository.upsert_player(player_id=100, username="new", display_name="Alice")
    updated = repository.get_player(100)
    assert updated is not None
    assert updated["username"] == "new"
    assert updated["first_seen_at"] == first_seen
    assert updated["last_seen_at"] != "2000-01-01 00:00:00"


def test_get_player_returns_none_for_unknown_id(conn: sqlite3.Connection) -> None:
    """Missing player lookups return None."""
    assert PlayersRepository(conn).get_player(404) is None


def test_list_players_orders_by_last_seen_descending(conn: sqlite3.Connection) -> None:
    """Player lists are ordered from most recently seen to least recent."""
    repository = PlayersRepository(conn)
    repository.upsert_player(player_id=100, username="a", display_name="A")
    repository.upsert_player(player_id=200, username="b", display_name="B")
    conn.execute("UPDATE players SET last_seen_at=? WHERE id=100", ("2025-01-01",))
    conn.execute("UPDATE players SET last_seen_at=? WHERE id=200", ("2025-02-01",))
    assert [row["id"] for row in repository.list_players()] == [200, 100]


def test_update_notes_persists_player_notes(conn: sqlite3.Connection) -> None:
    """Player notes can be updated and read back."""
    repository = PlayersRepository(conn)
    repository.upsert_player(player_id=100, username="alice", display_name="Alice")
    repository.update_notes(player_id=100, notes="careful player")
    row = repository.get_player(100)
    assert row is not None
    assert row["notes"] == "careful player"
    with pytest.raises(ConfigError, match="player 404 not found"):
        repository.update_notes(player_id=404, notes="missing")
