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
        is_winner: bool | None = None,
    ) -> int:
        """Insert a game_players row and return its row identifier.

        The insert is ``INSERT OR IGNORE`` into a unique ``(game_id,
        player_id)`` pair, so registering the same player in the same game
        twice is a silent no-op that returns the identifier of the existing
        row. An existing row keeps whatever ground truth it already holds: a
        repeated attach or a later backfill never erases a final role and never
        fails the caller.

        ``game_id`` and ``player_id`` must already exist in ``games`` and
        ``players`` respectively: the schema declares foreign keys and the
        connection enables ``PRAGMA foreign_keys``, so a missing parent row
        still raises :class:`sqlite3.IntegrityError` (``OR IGNORE`` never
        suppresses foreign-key violations).

        Args:
            game_id: Identifier of an existing game.
            player_id: Identifier of an existing player.
            final_role: Russian role name (``name_ru`` from ``roles.yaml``) or
                None while the role is still unknown.
            survived: True/False when the outcome is known, None otherwise;
                stored as INTEGER 1/0/NULL.
            is_winner: True/False when the final report named the player,
                None otherwise; stored as INTEGER 1/0/NULL.

        Returns:
            The ``game_players.id`` of the inserted row, or of the existing row
            when the player was already registered in this game.

        Raises:
            sqlite3.IntegrityError: If the game or the player does not exist.
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO game_players"
            "(game_id, player_id, final_role, survived, is_winner) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                game_id,
                player_id,
                final_role,
                _to_db_survived(survived),
                _to_db_optional_flag(is_winner),
            ),
        )
        row = self._conn.execute(
            "SELECT id FROM game_players WHERE game_id = ? AND player_id = ?",
            (game_id, player_id),
        ).fetchone()
        if row is None:  # pragma: no cover — INSERT OR IGNORE always leaves one row
            raise sqlite3.IntegrityError(
                f"game_players row for player {player_id} in game {game_id} is missing"
            )
        return int(row["id"])

    def set_final_role(
        self,
        *,
        game_id: int,
        player_id: int,
        final_role: str,
        survived: bool | None = None,
        is_winner: bool | None = None,
    ) -> None:
        """Store a player's final role and outcome flags.

        ``survived`` and ``is_winner`` use ``COALESCE``: passing None leaves
        the stored value untouched instead of erasing ground truth that was
        written earlier (retro may run more than once for the same game).

        Args:
            game_id: Identifier of the game.
            player_id: Identifier of the player.
            final_role: Russian role name (``name_ru`` from ``roles.yaml``).
            survived: True/False to also store the outcome, None to keep it.
            is_winner: True/False to also store the result, None to keep it.

        Raises:
            ConfigError: If the player has no participation row in this game.
        """
        cursor = self._conn.execute(
            "UPDATE game_players SET final_role = ?, survived = COALESCE(?, survived), "
            "is_winner = COALESCE(?, is_winner) WHERE game_id = ? AND player_id = ?",
            (
                final_role,
                _to_db_survived(survived),
                _to_db_optional_flag(is_winner),
                game_id,
                player_id,
            ),
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
    return _to_db_optional_flag(survived)


def _to_db_optional_flag(flag: bool | None) -> int | None:
    """Map an optional boolean flag to the INTEGER 1/0/NULL stored in SQLite."""
    if flag is None:
        return None
    return int(flag)
