"""End-to-end tests for the analyzer → predictions → retro cycle (C4)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from domain.roles import RoleRegistry, load_roles
from features.memory.retro import retro_analyze
from infra.db.connection import connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.game_players import GamePlayersRepository
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository

ROLES_PATH = Path(__file__).resolve().parents[2] / "configs" / "roles.yaml"
CHAT_ID = -1001111111111


@pytest.fixture(name="registry")
def registry_fixture() -> RoleRegistry:
    """Load the configured role definitions."""
    return load_roles(ROLES_PATH)


@pytest.fixture(name="conn")
def conn_fixture() -> Iterator[sqlite3.Connection]:
    """Provide a migrated in-memory SQLite database."""
    connection = connect(":memory:")
    apply_migrations(connection)
    try:
        yield connection
    finally:
        connection.close()


def stored_correct(conn: sqlite3.Connection, game_id: int) -> list[int | None]:
    """Return the stored correctness labels of one game in insertion order."""
    rows = conn.execute(
        "SELECT correct FROM predictions WHERE game_id = ? ORDER BY id", (game_id,)
    ).fetchall()
    return [row["correct"] for row in rows]


def prepare_full_game(conn: sqlite3.Connection) -> int:
    """Create a finished three-player game with labelled predictions."""
    games_repo = GamesRepository(conn)
    players_repo = PlayersRepository(conn)
    game_players_repo = GamePlayersRepository(conn)
    predictions_repo = PredictionsRepository(conn)

    game_id = games_repo.create_game(chat_id=CHAT_ID, player_count=3)
    final_roles = {101: "Мафия", 102: "Дон", 103: "Мирный житель"}
    predictions = {101: "Мафия", 102: "Дон", 103: "Маньяк"}

    for player_id, final_role in final_roles.items():
        players_repo.upsert_player(
            player_id=player_id, username=None, display_name=f"Player {player_id}"
        )
        game_players_repo.add_player(
            game_id=game_id, player_id=player_id, final_role=final_role, survived=True
        )
    for player_id, predicted_role in predictions.items():
        predictions_repo.insert_prediction(
            game_id=game_id,
            player_id=player_id,
            predicted_role=predicted_role,
            confidence=0.6,
            reasoning="e2e prediction",
        )
    games_repo.finish_game(game_id=game_id, winner_faction="mafia")
    return game_id


def test_e2e_full_retro_cycle(conn: sqlite3.Connection, registry: RoleRegistry) -> None:
    """Predictions are labelled, aggregated, and reported in one pass."""
    game_id = prepare_full_game(conn)

    report = retro_analyze(
        game_id=game_id,
        registry=registry,
        predictions_repo=PredictionsRepository(conn),
        game_players_repo=GamePlayersRepository(conn),
        games_repo=GamesRepository(conn),
    )

    assert report.game_id == game_id
    assert report.total_predictions == 3
    assert report.correct == 2
    assert report.wrong == 1
    assert report.unknown == 0
    assert set(report.by_role) == {"Мафия", "Дон", "Маньяк"}
    assert report.by_role["Мафия"].correct == 1
    assert report.by_role["Дон"].correct == 1
    assert report.by_role["Маньяк"].correct == 0
    assert set(stored_correct(conn, game_id)) == {0, 1}


def test_e2e_retro_no_final_roles(conn: sqlite3.Connection, registry: RoleRegistry) -> None:
    """A finished game without final roles leaves every label untouched."""
    games_repo = GamesRepository(conn)
    players_repo = PlayersRepository(conn)
    game_players_repo = GamePlayersRepository(conn)
    predictions_repo = PredictionsRepository(conn)

    game_id = games_repo.create_game(chat_id=CHAT_ID, player_count=2)
    for player_id in (201, 202):
        players_repo.upsert_player(
            player_id=player_id, username=None, display_name=f"Player {player_id}"
        )
        game_players_repo.add_player(game_id=game_id, player_id=player_id)
    for player_id in (201, 202):
        predictions_repo.insert_prediction(
            game_id=game_id,
            player_id=player_id,
            predicted_role="Мафия",
            confidence=0.4,
            reasoning="e2e prediction without ground truth",
        )

    report = retro_analyze(
        game_id=game_id,
        registry=registry,
        predictions_repo=predictions_repo,
        game_players_repo=game_players_repo,
        games_repo=games_repo,
    )

    assert report.total_predictions == 2
    assert report.unknown == 2
    assert report.correct == 0
    assert report.wrong == 0
    assert report.by_role == {}
    assert stored_correct(conn, game_id) == [None, None]
