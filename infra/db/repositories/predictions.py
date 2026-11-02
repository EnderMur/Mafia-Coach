"""Repository for model predictions."""

from __future__ import annotations

import sqlite3

from domain.exceptions import ConfigError


class PredictionsRepository:
    """Persist predictions and attach eventual correctness labels."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the repository with a SQLite connection."""
        self._conn = conn

    def insert_prediction(
        self,
        *,
        game_id: int,
        player_id: int,
        predicted_role: str,
        confidence: float,
        reasoning: str,
    ) -> int:
        """Deprecated compatibility alias for the current prediction upsert."""
        return self.upsert_prediction(
            game_id=game_id,
            player_id=player_id,
            predicted_role=predicted_role,
            confidence=confidence,
            reasoning=reasoning,
        )

    def upsert_prediction(
        self,
        *,
        game_id: int,
        player_id: int,
        predicted_role: str,
        confidence: float,
        reasoning: str,
    ) -> int:
        """Insert or update the current prediction and return its stable id."""
        cursor = self._conn.execute(
            "INSERT INTO predictions (game_id, ts, player_id, predicted_role, confidence, reasoning) "
            "VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?) "
            "ON CONFLICT(game_id, player_id) DO UPDATE SET "
            "ts = CURRENT_TIMESTAMP, predicted_role = excluded.predicted_role, "
            "confidence = excluded.confidence, reasoning = excluded.reasoning RETURNING id",
            (game_id, player_id, predicted_role, confidence, reasoning),
        )
        row = cursor.fetchone()
        assert row is not None
        return int(row[0])

    def insert_history(
        self,
        *,
        game_id: int,
        player_id: int,
        predicted_role: str,
        confidence: float,
        reasoning: str,
    ) -> int:
        """Append a prediction to history and return its id."""
        cursor = self._conn.execute(
            "INSERT INTO predictions_history "
            "(game_id, ts, player_id, predicted_role, confidence, reasoning) "
            "VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)",
            (game_id, player_id, predicted_role, confidence, reasoning),
        )
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    def list_predictions(self, game_id: int) -> list[sqlite3.Row]:
        """Return a game's predictions in timestamp and insertion order."""
        cursor = self._conn.execute(
            "SELECT * FROM predictions WHERE game_id = ? ORDER BY ts ASC, id ASC",
            (game_id,),
        )
        return list(cursor.fetchall())

    def mark_correct(self, *, prediction_id: int, correct: bool) -> None:
        """Set the correctness label or raise if the prediction is missing."""
        cursor = self._conn.execute(
            "UPDATE predictions SET correct = ? WHERE id = ?",
            (int(correct), prediction_id),
        )
        if cursor.rowcount == 0:
            raise ConfigError(f"prediction {prediction_id} not found")
