"""Repository for game records."""

from __future__ import annotations

import sqlite3
from typing import cast

from domain.exceptions import ConfigError
from domain.roles import Faction


class GamesRepository:
    """Read and write game records."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the repository with a SQLite connection."""
        self._conn = conn

    def create_game(self, *, chat_id: int, player_count: int) -> int:
        """Create a game and return its database identifier."""
        cursor = self._conn.execute(
            "INSERT INTO games(chat_id, started_at, player_count) VALUES (?, CURRENT_TIMESTAMP, ?)",
            (chat_id, player_count),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def finish_game(self, *, game_id: int, winner_faction: str) -> None:
        """Finish a game with a validated winning faction."""
        try:
            faction = Faction(winner_faction)
        except ValueError as error:
            raise ConfigError(f"invalid winner faction: {winner_faction}") from error

        cursor = self._conn.execute(
            "UPDATE games SET ended_at = CURRENT_TIMESTAMP, winner_faction = ? WHERE id = ?",
            (faction.value, game_id),
        )
        if cursor.rowcount == 0:
            raise ConfigError(f"game {game_id} not found")

    def get_game(self, game_id: int) -> sqlite3.Row:
        """Return a game row or raise when the game does not exist."""
        result = self._conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        row = cast(sqlite3.Row | None, result)
        if row is None:
            raise ConfigError(f"game {game_id} not found")
        return row

    def list_games(self, *, limit: int = 50, offset: int = 0) -> list[sqlite3.Row]:
        """Return recent games ordered by start time descending."""
        cursor = self._conn.execute(
            "SELECT * FROM games ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return list(cursor.fetchall())
