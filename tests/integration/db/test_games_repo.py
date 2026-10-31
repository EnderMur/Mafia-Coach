"""Tests for the games repository."""

from __future__ import annotations

import sqlite3

import pytest

from domain.exceptions import ConfigError
from infra.db.repositories.games import GamesRepository


def test_create_game_returns_id_and_gets_row(conn: sqlite3.Connection) -> None:
    """Created games can be retrieved by their generated id."""
    repository = GamesRepository(conn)
    game_id = repository.create_game(chat_id=-100123, player_count=10)
    row = repository.get_game(game_id)
    assert isinstance(game_id, int)
    assert row["id"] == game_id
    assert row["chat_id"] == -100123
    assert row["player_count"] == 10


def test_finish_game_updates_end_fields(conn: sqlite3.Connection) -> None:
    """Finishing a game stores its end time and faction value."""
    repository = GamesRepository(conn)
    game_id = repository.create_game(chat_id=-100123, player_count=10)
    repository.finish_game(game_id=game_id, winner_faction="town")
    row = repository.get_game(game_id)
    assert row["ended_at"] is not None
    assert row["winner_faction"] == "town"


def test_finish_game_missing_id_raises_config_error(conn: sqlite3.Connection) -> None:
    """Finishing an unknown game reports a configuration error."""
    with pytest.raises(ConfigError, match="game 404 not found"):
        GamesRepository(conn).finish_game(game_id=404, winner_faction="town")


def test_finish_game_rejects_invalid_faction(conn: sqlite3.Connection) -> None:
    """Winner factions must match the domain enum."""
    repository = GamesRepository(conn)
    game_id = repository.create_game(chat_id=1, player_count=2)
    with pytest.raises(ConfigError, match="invalid winner faction"):
        repository.finish_game(game_id=game_id, winner_faction="bogus")


def test_list_games_orders_by_start_time_descending(conn: sqlite3.Connection) -> None:
    """Game lists are ordered from newest to oldest start time."""
    repository = GamesRepository(conn)
    first_id = repository.create_game(chat_id=1, player_count=2)
    second_id = repository.create_game(chat_id=2, player_count=3)
    conn.execute("UPDATE games SET started_at=? WHERE id=?", ("2025-01-01", first_id))
    conn.execute("UPDATE games SET started_at=? WHERE id=?", ("2025-02-01", second_id))
    assert [row["id"] for row in repository.list_games()] == [second_id, first_id]


def test_get_game_missing_id_raises_config_error(conn: sqlite3.Connection) -> None:
    """Retrieving an unknown game reports a configuration error."""
    with pytest.raises(ConfigError, match="game 404 not found"):
        GamesRepository(conn).get_game(404)
