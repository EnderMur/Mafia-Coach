"""Tests for the end-of-game retrospective over stored predictions."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from domain.exceptions import ConfigError
from domain.roles import RoleRegistry, load_roles
from features.memory.retro import GameRetroReport, RoleAccuracy, retro_analyze
from infra.db.repositories.game_players import GamePlayersRepository
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository

ROLES_PATH = Path(__file__).resolve().parents[4] / "configs" / "roles.yaml"


@dataclass(frozen=True)
class RetroFixture:
    """Repositories and identifiers of one prepared game."""

    game_id: int
    registry: RoleRegistry
    games_repo: GamesRepository
    predictions_repo: PredictionsRepository
    game_players_repo: GamePlayersRepository


@pytest.fixture(name="registry")
def registry_fixture() -> RoleRegistry:
    """Load the configured role definitions."""
    return load_roles(ROLES_PATH)


def prepare_game(
    conn: sqlite3.Connection,
    registry: RoleRegistry,
    *,
    roles: dict[int, str | None],
    predictions: list[tuple[int, str]],
    extra_player_ids: tuple[int, ...] = (),
) -> RetroFixture:
    """Create a game with players, their final roles, and predictions.

    ``extra_player_ids`` are stored in ``players`` only: they never get a
    ``game_players`` row, which models a prediction about an unregistered player.
    """
    games_repo = GamesRepository(conn)
    players_repo = PlayersRepository(conn)
    game_players_repo = GamePlayersRepository(conn)
    predictions_repo = PredictionsRepository(conn)

    game_id = games_repo.create_game(chat_id=-1007777777777, player_count=len(roles))
    for player_id, final_role in roles.items():
        players_repo.upsert_player(
            player_id=player_id, username=None, display_name=f"Player {player_id}"
        )
        game_players_repo.add_player(game_id=game_id, player_id=player_id, final_role=final_role)

    for player_id in extra_player_ids:
        players_repo.upsert_player(
            player_id=player_id, username=None, display_name=f"Player {player_id}"
        )

    for player_id, predicted_role in predictions:
        predictions_repo.insert_prediction(
            game_id=game_id,
            player_id=player_id,
            predicted_role=predicted_role,
            confidence=0.5,
            reasoning="проверка ретро",
        )

    return RetroFixture(
        game_id=game_id,
        registry=registry,
        games_repo=games_repo,
        predictions_repo=predictions_repo,
        game_players_repo=game_players_repo,
    )


def run_retro(fixture: RetroFixture) -> GameRetroReport:
    """Run the retrospective with the games repository attached."""
    return retro_analyze(
        game_id=fixture.game_id,
        registry=fixture.registry,
        predictions_repo=fixture.predictions_repo,
        game_players_repo=fixture.game_players_repo,
        games_repo=fixture.games_repo,
    )


def stored_correct(conn: sqlite3.Connection, game_id: int) -> list[int | None]:
    """Return the stored correctness labels of one game in insertion order."""
    rows = conn.execute(
        "SELECT correct FROM predictions WHERE game_id = ? ORDER BY id", (game_id,)
    ).fetchall()
    return [row["correct"] for row in rows]


def test_empty_game_produces_zero_report(conn: sqlite3.Connection, registry: RoleRegistry) -> None:
    """A game without predictions yields an all-zero report."""
    fixture = prepare_game(conn, registry, roles={101: "Мафия", 102: "Дон"}, predictions=[])

    report = run_retro(fixture)

    assert isinstance(report, GameRetroReport)
    assert report.game_id == fixture.game_id
    assert report.total_predictions == 0
    assert report.correct == 0
    assert report.wrong == 0
    assert report.unknown == 0
    assert report.by_role == {}


def test_correct_prediction_is_counted(conn: sqlite3.Connection, registry: RoleRegistry) -> None:
    """A prediction equal to the final role is counted as correct."""
    fixture = prepare_game(
        conn,
        registry,
        roles={101: "Мафия"},
        predictions=[(101, "Мафия")],
    )

    report = run_retro(fixture)

    assert report.total_predictions == 1
    assert report.correct == 1
    assert report.wrong == 0
    assert report.unknown == 0
    assert stored_correct(conn, fixture.game_id) == [1]


def test_wrong_prediction_is_counted(conn: sqlite3.Connection, registry: RoleRegistry) -> None:
    """A prediction that differs from the final role is counted as wrong."""
    fixture = prepare_game(
        conn,
        registry,
        roles={101: "Дон"},
        predictions=[(101, "Мирный житель")],
    )

    report = run_retro(fixture)

    assert report.total_predictions == 1
    assert report.correct == 0
    assert report.wrong == 1
    assert report.unknown == 0
    assert stored_correct(conn, fixture.game_id) == [0]


def test_prediction_without_final_role_stays_unlabelled(
    conn: sqlite3.Connection, registry: RoleRegistry
) -> None:
    """Unknown final roles are counted as unknown and never written back."""
    fixture = prepare_game(
        conn,
        registry,
        roles={101: None},
        predictions=[(101, "Маньяк")],
    )

    report = run_retro(fixture)

    assert report.total_predictions == 1
    assert report.correct == 0
    assert report.wrong == 0
    assert report.unknown == 1
    assert report.by_role == {}
    assert stored_correct(conn, fixture.game_id) == [None]


def test_mixed_game_aggregates_counts(conn: sqlite3.Connection, registry: RoleRegistry) -> None:
    """Two correct and one wrong prediction are counted separately."""
    fixture = prepare_game(
        conn,
        registry,
        roles={101: "Мафия", 102: "Дон", 103: "Мирный житель"},
        predictions=[(101, "Мафия"), (102, "Дон"), (103, "Мафия")],
    )

    report = run_retro(fixture)

    assert report.total_predictions == 3
    assert report.correct == 2
    assert report.wrong == 1
    assert report.unknown == 0
    assert stored_correct(conn, fixture.game_id) == [1, 1, 0]


def test_by_role_is_grouped_by_predicted_role(
    conn: sqlite3.Connection, registry: RoleRegistry
) -> None:
    """Per-role stats aggregate over the predicted role, not the final one."""
    fixture = prepare_game(
        conn,
        registry,
        roles={101: "Мафия", 102: "Дон", 103: "Мирный житель"},
        predictions=[(101, "Мафия"), (102, "Мафия"), (103, "Мафия")],
    )

    report = run_retro(fixture)

    assert set(report.by_role) == {"Мафия"}
    mafia_stats = report.by_role["Мафия"]
    assert isinstance(mafia_stats, RoleAccuracy)
    assert mafia_stats.total == 3
    assert mafia_stats.correct == 1
    assert report.correct == 1
    assert report.wrong == 2


def test_by_role_splits_multiple_predicted_roles(
    conn: sqlite3.Connection, registry: RoleRegistry
) -> None:
    """Different predicted roles are reported as separate entries."""
    fixture = prepare_game(
        conn,
        registry,
        roles={101: "Мафия", 102: "Дон", 103: "Мирный житель"},
        predictions=[(101, "Мафия"), (102, "Дон"), (103, "Мафия")],
    )

    report = run_retro(fixture)

    assert report.by_role["Мафия"] == RoleAccuracy(total=2, correct=1)
    assert report.by_role["Дон"] == RoleAccuracy(total=1, correct=1)
    assert report.wrong == 1


def test_prediction_without_participation_row_is_unknown(
    conn: sqlite3.Connection, registry: RoleRegistry
) -> None:
    """A predicted player missing from game_players counts as unknown."""
    fixture = prepare_game(
        conn,
        registry,
        roles={101: "Мафия"},
        predictions=[(101, "Мафия"), (999, "Дон")],
        extra_player_ids=(999,),
    )

    report = run_retro(fixture)

    assert report.total_predictions == 2
    assert report.correct == 1
    assert report.wrong == 0
    assert report.unknown == 1
    assert stored_correct(conn, fixture.game_id) == [1, None]


def test_retro_without_games_repository_does_not_validate_game(
    conn: sqlite3.Connection, registry: RoleRegistry
) -> None:
    """Without ``games_repo`` the report is built from whatever predictions exist."""
    report = retro_analyze(
        game_id=404,
        registry=registry,
        predictions_repo=PredictionsRepository(conn),
        game_players_repo=GamePlayersRepository(conn),
    )

    assert report.game_id == 404
    assert report.total_predictions == 0
    assert report.unknown == 0


def test_missing_game_raises_config_error(conn: sqlite3.Connection, registry: RoleRegistry) -> None:
    """A game that is absent from the games table is rejected."""
    with pytest.raises(ConfigError, match="game 404 not found"):
        retro_analyze(
            game_id=404,
            registry=registry,
            predictions_repo=PredictionsRepository(conn),
            game_players_repo=GamePlayersRepository(conn),
            games_repo=GamesRepository(conn),
        )


def test_retro_is_idempotent(conn: sqlite3.Connection, registry: RoleRegistry) -> None:
    """Running the retrospective twice keeps the same report and does not duplicate rows."""
    fixture = prepare_game(
        conn,
        registry,
        roles={101: "Мафия", 102: "Дон"},
        predictions=[(101, "Мафия"), (102, "Мирный житель")],
    )

    first = run_retro(fixture)
    second = run_retro(fixture)

    assert first == second
    assert second.correct == 1
    assert second.wrong == 1
    rows = conn.execute(
        "SELECT COUNT(*) AS total FROM predictions WHERE game_id = ?", (fixture.game_id,)
    ).fetchone()
    assert rows is not None
    assert rows["total"] == 2
    assert stored_correct(conn, fixture.game_id) == [1, 0]
