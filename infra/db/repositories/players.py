"""Repository for global player records."""

from __future__ import annotations

import sqlite3
from typing import cast

from domain.exceptions import ConfigError


class PlayersRepository:
    """Read and update global player records."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the repository with a SQLite connection."""
        self._conn = conn

    def upsert_player(self, *, player_id: int, username: str | None, display_name: str) -> None:
        """Insert or refresh a player, storing username as the sole name field."""
        self._conn.execute(
            "INSERT INTO players(id, username, first_seen_at, last_seen_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
            "ON CONFLICT(id) DO UPDATE SET "
            "username = excluded.username, last_seen_at = CURRENT_TIMESTAMP",
            (player_id, username),
        )

    def get_player(self, player_id: int) -> sqlite3.Row | None:
        """Return a player row, or None when the id is not present."""
        row = self._conn.execute("SELECT * FROM players WHERE id = ?", (player_id,)).fetchone()
        return cast(sqlite3.Row | None, row)

    def list_players(self, *, limit: int = 100) -> list[sqlite3.Row]:
        """Return recently seen players ordered by last-seen time."""
        cursor = self._conn.execute(
            "SELECT * FROM players ORDER BY last_seen_at DESC LIMIT ?", (limit,)
        )
        return list(cursor.fetchall())

    def update_notes(self, *, player_id: int, notes: str) -> None:
        """Update a player's notes or raise if the player is not found."""
        cursor = self._conn.execute("UPDATE players SET notes = ? WHERE id = ?", (notes, player_id))
        if cursor.rowcount == 0:
            raise ConfigError(f"player {player_id} not found")

    def delete_player(self, *, player_id: int) -> None:
        """Remove placeholder from players (merge cleanup)."""
        self._conn.execute(
            "DELETE FROM players WHERE id = ? AND id < 0",  # только placeholder
            (player_id,),
        )
