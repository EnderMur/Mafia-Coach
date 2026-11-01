"""Repository for per-game player participation records."""

from __future__ import annotations

import sqlite3
from typing import cast

from domain.exceptions import ConfigError


class GamePlayersRepository:
    """CRUD for game_players. Per-game player participation."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the repository with a SQLite connection."""
        self._conn = conn

    def add_player(
        self,
        *,
        game_id: int,
        player_id: int,
        final_role: str | None = None,
        survived: bool | None = None,
    ) -> int:
        """Insert a game_players row and return its row identifier.

        ``game_id`` and ``player_id`` must already exist in ``games`` and
        ``players`` respectively: the schema declares foreign keys and the
        connection enables ``PRAGMA foreign_keys``, so a missing parent row
        raises :class:`sqlite3.IntegrityError`.

        Args:
            game_id: Identifier of an existing game.
            player_id: Identifier of an existing player.
            final_role: Russian role name (``name_ru`` from ``roles.yaml``) or
                None while the role is still unknown.
            survived: True/False when the outcome is known, None otherwise;
                stored as INTEGER 1/0/NULL.

        Returns:
            The ``game_players.id`` of the inserted row.

        Raises:
            ConfigError: If the player is already registered in this game.
            sqlite3.IntegrityError: If the game or the player does not exist.
        """
        if self.get_player_in_game(game_id=game_id, player_id=player_id) is not None:
            raise ConfigError(f"player {player_id} already registered in game {game_id}")

        cursor = self._conn.execute(
            "INSERT INTO game_players(game_id, player_id, final_role, survived) VALUES (?, ?, ?, ?)",
            (game_id, player_id, final_role, _to_db_survived(survived)),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def set_final_role(
        self,
        *,
        game_id: int,
        player_id: int,
        final_role: str,
        survived: bool | None = None,
    ) -> None:
        """Store a player's final role, optionally updating the survival flag.

        When ``survived`` is None the stored survival value is left untouched.

        Args:
            game_id: Identifier of the game.
            player_id: Identifier of the player.
            final_role: Russian role name (``name_ru`` from ``roles.yaml``).
            survived: True/False to also store the outcome, None to keep it.

        Raises:
            ConfigError: If the player has no participation row in this game.
        """
        if survived is None:
            cursor = self._conn.execute(
                "UPDATE game_players SET final_role = ? WHERE game_id = ? AND player_id = ?",
                (final_role, game_id, player_id),
            )
        else:
            cursor = self._conn.execute(
                "UPDATE game_players SET final_role = ?, survived = ? "
                "WHERE game_id = ? AND player_id = ?",
                (final_role, _to_db_survived(survived), game_id, player_id),
            )
        if cursor.rowcount == 0:
            raise ConfigError(f"player {player_id} not registered in game {game_id}")

    def get_player_in_game(self, *, game_id: int, player_id: int) -> sqlite3.Row | None:
        """Return one participation row, or None when it is absent."""
        row = self._conn.execute(
            "SELECT * FROM game_players WHERE game_id = ? AND player_id = ?",
            (game_id, player_id),
        ).fetchone()
        return cast(sqlite3.Row | None, row)

    def list_game_players(self, game_id: int) -> list[sqlite3.Row]:
        """Return all participation rows of a game ordered by id."""
        cursor = self._conn.execute(
            "SELECT * FROM game_players WHERE game_id = ? ORDER BY id ASC", (game_id,)
        )
        return list(cursor.fetchall())


def _to_db_survived(survived: bool | None) -> int | None:
    """Map an optional survival flag to the INTEGER 1/0/NULL stored in SQLite."""
    if survived is None:
        return None
    return int(survived)
