"""Tests for the game_players repository."""

from __future__ import annotations

import sqlite3

import pytest

from domain.exceptions import ConfigError
from infra.db.repositories.game_players import GamePlayersRepository
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository

GAME_CHAT_ID = -1001234567890


def create_participation_repository(
    conn: sqlite3.Connection,
) -> tuple[int, GamePlayersRepository]:
    """Create the FK parents (one game, one player) and the repository under test."""
    game_id = GamesRepository(conn).create_game(chat_id=GAME_CHAT_ID, player_count=3)
    PlayersRepository(conn).upsert_player(player_id=100, username="alice", display_name="Alice")
    return game_id, GamePlayersRepository(conn)


def test_add_player_creates_retrievable_row(conn: sqlite3.Connection) -> None:
    """A new participation row is stored with empty role and survival."""
    game_id, repository = create_participation_repository(conn)
    row_id = repository.add_player(game_id=game_id, player_id=100)

    row = repository.get_player_in_game(game_id=game_id, player_id=100)
    assert row is not None
    assert row["id"] == row_id
    assert row["game_id"] == game_id
    assert row["player_id"] == 100
    assert row["final_role"] is None
    assert row["survived"] is None


def test_add_player_stores_russian_final_role(conn: sqlite3.Connection) -> None:
    """The final role is stored verbatim as the configured Russian name."""
    game_id, repository = create_participation_repository(conn)
    repository.add_player(game_id=game_id, player_id=100, final_role="Мафия")

    row = repository.get_player_in_game(game_id=game_id, player_id=100)
    assert row is not None
    assert row["final_role"] == "Мафия"


def test_add_player_stores_survived_true_as_integer_one(conn: sqlite3.Connection) -> None:
    """A surviving player is stored as INTEGER 1."""
    game_id, repository = create_participation_repository(conn)
    repository.add_player(game_id=game_id, player_id=100, survived=True)

    stored = conn.execute(
        "SELECT survived FROM game_players WHERE game_id = ? AND player_id = ?",
        (game_id, 100),
    ).fetchone()
    assert stored is not None
    assert stored["survived"] == 1
    assert isinstance(stored["survived"], int)


def test_add_player_stores_survived_false_as_integer_zero(conn: sqlite3.Connection) -> None:
    """A dead player is stored as INTEGER 0."""
    game_id, repository = create_participation_repository(conn)
    repository.add_player(game_id=game_id, player_id=100, survived=False)

    stored = conn.execute(
        "SELECT survived FROM game_players WHERE game_id = ? AND player_id = ?",
        (game_id, 100),
    ).fetchone()
    assert stored is not None
    assert stored["survived"] == 0


def test_add_player_stores_unknown_survival_as_null(conn: sqlite3.Connection) -> None:
    """An unknown survival flag is stored as SQL NULL."""
    game_id, repository = create_participation_repository(conn)
    repository.add_player(game_id=game_id, player_id=100, survived=None)

    stored = conn.execute(
        "SELECT survived FROM game_players WHERE game_id = ? AND player_id = ?",
        (game_id, 100),
    ).fetchone()
    assert stored is not None
    assert stored["survived"] is None


def test_add_player_insert_or_ignore(conn: sqlite3.Connection) -> None:
    """13. Re-registering a player is a silent no-op that keeps one row and its id."""
    game_id, repository = create_participation_repository(conn)

    first_id = repository.add_player(game_id=game_id, player_id=100, final_role="Мафия")
    second_id = repository.add_player(game_id=game_id, player_id=100)

    assert second_id == first_id
    rows = repository.list_game_players(game_id)
    assert len(rows) == 1
    assert rows[0]["id"] == first_id
    # The second call must not erase ground truth stored by the first one.
    assert rows[0]["final_role"] == "Мафия"


def test_set_final_role_updates_existing_row(conn: sqlite3.Connection) -> None:
    """The final role of an existing participation row is replaced."""
    game_id, repository = create_participation_repository(conn)
    repository.add_player(game_id=game_id, player_id=100)
    repository.set_final_role(game_id=game_id, player_id=100, final_role="Дон")

    row = repository.get_player_in_game(game_id=game_id, player_id=100)
    assert row is not None
    assert row["final_role"] == "Дон"


def test_set_final_role_updates_survival_when_given(conn: sqlite3.Connection) -> None:
    """Passing ``survived`` also refreshes the stored survival flag."""
    game_id, repository = create_participation_repository(conn)
    repository.add_player(game_id=game_id, player_id=100, final_role="Мирный житель", survived=True)
    repository.set_final_role(game_id=game_id, player_id=100, final_role="Дон", survived=False)

    row = repository.get_player_in_game(game_id=game_id, player_id=100)
    assert row is not None
    assert row["final_role"] == "Дон"
    assert row["survived"] == 0


def test_set_final_role_keeps_survival_when_omitted(conn: sqlite3.Connection) -> None:
    """Omitting ``survived`` leaves the stored survival flag untouched."""
    game_id, repository = create_participation_repository(conn)
    repository.add_player(game_id=game_id, player_id=100, survived=True)
    repository.set_final_role(game_id=game_id, player_id=100, final_role="Дон")

    row = repository.get_player_in_game(game_id=game_id, player_id=100)
    assert row is not None
    assert row["survived"] == 1


def test_set_final_role_without_row_raises_config_error(conn: sqlite3.Connection) -> None:
    """Updating a missing participation row reports a configuration error."""
    game_id, repository = create_participation_repository(conn)

    with pytest.raises(ConfigError, match="player 100 not registered in game"):
        repository.set_final_role(game_id=game_id, player_id=100, final_role="Мафия")


def test_get_player_in_game_returns_none_for_unknown_row(conn: sqlite3.Connection) -> None:
    """A missing participation row is reported as None."""
    game_id, repository = create_participation_repository(conn)
    assert repository.get_player_in_game(game_id=game_id, player_id=100) is None


def test_list_game_players_returns_rows_ordered_by_id(conn: sqlite3.Connection) -> None:
    """Participation rows come back in insertion order."""
    game_id, repository = create_participation_repository(conn)
    players_repository = PlayersRepository(conn)
    for player_id in (200, 300):
        players_repository.upsert_player(
            player_id=player_id, username=None, display_name=f"Player {player_id}"
        )
    repository.add_player(game_id=game_id, player_id=300, final_role="Мафия")
    repository.add_player(game_id=game_id, player_id=100, final_role="Дон")
    repository.add_player(game_id=game_id, player_id=200, final_role="Мирный житель")

    rows = repository.list_game_players(game_id)
    assert [row["player_id"] for row in rows] == [300, 100, 200]
    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)


def test_list_game_players_is_scoped_to_one_game(conn: sqlite3.Connection) -> None:
    """Rows of another game are never returned."""
    game_id, repository = create_participation_repository(conn)
    other_game_id = GamesRepository(conn).create_game(chat_id=-100999, player_count=1)
    repository.add_player(game_id=game_id, player_id=100, final_role="Мафия")
    repository.add_player(game_id=other_game_id, player_id=100, final_role="Дон")

    rows = repository.list_game_players(game_id)
    assert [row["final_role"] for row in rows] == ["Мафия"]


def test_add_player_with_unknown_game_id_raises_integrity_error(
    conn: sqlite3.Connection,
) -> None:
    """Foreign keys reject participation rows of a game that does not exist."""
    _, repository = create_participation_repository(conn)

    with pytest.raises(sqlite3.IntegrityError):
        repository.add_player(game_id=404, player_id=100)


def test_add_player_with_unknown_player_id_raises_integrity_error(
    conn: sqlite3.Connection,
) -> None:
    """Foreign keys reject participation rows of a player that does not exist."""
    game_id, repository = create_participation_repository(conn)

    with pytest.raises(sqlite3.IntegrityError):
        repository.add_player(game_id=game_id, player_id=999)


def test_set_final_role_with_is_winner(conn: sqlite3.Connection) -> None:
    """11. The winner flag is stored as INTEGER 1/0 next to the final role."""
    game_id, repository = create_participation_repository(conn)
    repository.add_player(game_id=game_id, player_id=100)
    repository.set_final_role(
        game_id=game_id,
        player_id=100,
        final_role="Мафия",
        is_winner=True,
    )

    row = repository.get_player_in_game(game_id=game_id, player_id=100)
    assert row is not None
    assert row["final_role"] == "Мафия"
    assert row["is_winner"] == 1
    assert isinstance(row["is_winner"], int)


def test_set_final_role_coalesce_preserves(conn: sqlite3.Connection) -> None:
    """12. Omitting survived/is_winner keeps the ground truth already stored."""
    game_id, repository = create_participation_repository(conn)
    repository.add_player(
        game_id=game_id,
        player_id=100,
        final_role="Мирный житель",
        survived=False,
        is_winner=False,
    )

    repository.set_final_role(game_id=game_id, player_id=100, final_role="Дон")

    row = repository.get_player_in_game(game_id=game_id, player_id=100)
    assert row is not None
    assert row["final_role"] == "Дон"
    assert row["survived"] == 0
    assert row["is_winner"] == 0


def test_add_player_reports_legacy_duplicate_rows(conn: sqlite3.Connection) -> None:
    """A legacy table holding duplicate pairs fails loudly instead of adding more."""
    game_id, repository = create_participation_repository(conn)
    for _ in range(2):
        conn.execute(
            "INSERT INTO game_players(game_id, player_id) VALUES (?, ?)",
            (game_id, 100),
        )

    with pytest.raises(sqlite3.IntegrityError, match="duplicate"):
        repository.add_player(game_id=game_id, player_id=100)
