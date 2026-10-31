"""Tests for the predictions repository."""

from __future__ import annotations

import sqlite3

import pytest

from domain.exceptions import ConfigError
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository


def create_prediction_repository(conn: sqlite3.Connection) -> tuple[int, PredictionsRepository]:
    """Create FK dependencies and return a prediction repository."""
    game_id = GamesRepository(conn).create_game(chat_id=1, player_count=1)
    PlayersRepository(conn).upsert_player(player_id=100, username="alice", display_name="Alice")
    return game_id, PredictionsRepository(conn)


def test_insert_prediction_and_list(conn: sqlite3.Connection) -> None:
    """Predictions are inserted and returned with their stored values."""
    game_id, repository = create_prediction_repository(conn)
    prediction_id = repository.insert_prediction(
        game_id=game_id,
        player_id=100,
        predicted_role="civilian",
        confidence=0.7,
        reasoning="test reasoning",
    )
    rows = repository.list_predictions(game_id)
    assert len(rows) == 1
    assert rows[0]["id"] == prediction_id
    assert rows[0]["player_id"] == 100
    assert rows[0]["predicted_role"] == "civilian"
    assert rows[0]["confidence"] == pytest.approx(0.7)
    assert rows[0]["reasoning"] == "test reasoning"


def test_mark_prediction_correct_as_one(conn: sqlite3.Connection) -> None:
    """A true correctness result is stored as integer one."""
    game_id, repository = create_prediction_repository(conn)
    prediction_id = repository.insert_prediction(
        game_id=game_id,
        player_id=100,
        predicted_role="civilian",
        confidence=0.7,
        reasoning="test",
    )
    repository.mark_correct(prediction_id=prediction_id, correct=True)
    assert repository.list_predictions(game_id)[0]["correct"] == 1


def test_mark_prediction_incorrect_as_zero(conn: sqlite3.Connection) -> None:
    """A false correctness result is stored as integer zero."""
    game_id, repository = create_prediction_repository(conn)
    prediction_id = repository.insert_prediction(
        game_id=game_id,
        player_id=100,
        predicted_role="civilian",
        confidence=0.7,
        reasoning="test",
    )
    repository.mark_correct(prediction_id=prediction_id, correct=False)
    assert repository.list_predictions(game_id)[0]["correct"] == 0


def test_mark_missing_prediction_raises_config_error(conn: sqlite3.Connection) -> None:
    """Updating a missing prediction reports a configuration error."""
    with pytest.raises(ConfigError, match="prediction 404 not found"):
        PredictionsRepository(conn).mark_correct(prediction_id=404, correct=True)
