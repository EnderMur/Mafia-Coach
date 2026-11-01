"""Player profile views built on top of the players repository.

The module performs no I/O of its own apart from the repository passed in as an
argument, and it never awaits anything: ``sqlite3`` is synchronous and the
caller decides whether to offload it to a worker thread.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from domain.exceptions import ConfigError
from infra.db.repositories.players import PlayersRepository


class PlayerProfile(BaseModel):
    """Read-only view of a player profile from DB."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: int
    username: str | None
    notes: str | None
    first_seen_at: datetime | None
    last_seen_at: datetime | None


def load_profile(repo: PlayersRepository, player_id: int) -> PlayerProfile | None:
    """Load a player's profile or None if not found.

    Args:
        repo: Players repository bound to an open connection.
        player_id: Telegram user identifier.

    Returns:
        The immutable profile, or None when the player has never been stored.
    """
    row = repo.get_player(player_id)
    if row is None:
        return None
    return PlayerProfile(
        player_id=int(row["id"]),
        username=row["username"],
        notes=row["notes"],
        first_seen_at=_to_datetime(row["first_seen_at"]),
        last_seen_at=_to_datetime(row["last_seen_at"]),
    )


def update_notes(repo: PlayersRepository, *, player_id: int, notes: str) -> None:
    """Update free-form notes for a player.

    Args:
        repo: Players repository bound to an open connection.
        player_id: Telegram user identifier.
        notes: Free-form Russian notes ("блефует", "агрессивный", …).

    Raises:
        ConfigError: If the player is not stored yet (raised by the repository).
    """
    repo.update_notes(player_id=player_id, notes=notes)


def _to_datetime(value: object) -> datetime | None:
    """Convert a raw SQLite timestamp column into a naive ``datetime``.

    Raises:
        ConfigError: If the stored value is not a parseable timestamp.
    """
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as error:
        raise ConfigError(f"invalid timestamp in players table: {value!r}") from error
