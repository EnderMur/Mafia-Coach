"""Offline JSONL export tests using the real migrations and in-memory SQLite."""

from __future__ import annotations

import importlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from domain.events import MessageEvent, MessagePayload, VoteEvent, VotePayload
from infra.db.connection import connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.events import EventsRepository
from infra.db.repositories.game_players import GamePlayersRepository
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository

# scripts is a CLI namespace, not an installed package. Match the existing CLI
# tests and avoid assigning a standalone script a second mypy module name.
export_dataset = importlib.import_module("scripts.export_dataset")


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """Create an empty, fully migrated in-memory database."""
    connection = connect(":memory:")
    apply_migrations(connection)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def populated_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Seed one game, two events, one prediction, and one game player."""
    game_id = GamesRepository(conn).create_game(chat_id=-100, player_count=2)
    players = PlayersRepository(conn)
    players.upsert_player(player_id=101, username="alice", display_name="Алиса")
    players.upsert_player(player_id=102, username="boris", display_name="Борис")
    events = EventsRepository(conn)
    events.insert_event(
        MessageEvent(
            ts=datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
            game_id=game_id,
            payload=MessagePayload(author_id=101, author_name="Алиса", text="Я мирный игрок."),
        )
    )
    events.insert_event(
        VoteEvent(
            ts=datetime(2026, 10, 31, 12, 1, tzinfo=UTC),
            game_id=game_id,
            payload=VotePayload(
                voter_id=101,
                voter_name="Алиса",
                target_id=102,
                target_name="Борис",
            ),
        )
    )
    predictions = PredictionsRepository(conn)
    prediction_id = predictions.insert_prediction(
        game_id=game_id,
        player_id=101,
        predicted_role="Мафия",
        confidence=0.8,
        reasoning="Игрок уводит обсуждение от голосования.",
    )
    predictions.mark_correct(prediction_id=prediction_id, correct=True)
    GamePlayersRepository(conn).add_player(
        game_id=game_id,
        player_id=101,
        final_role="Мафия",
        survived=False,
        is_winner=False,
    )
    return conn


def test_export_raw_creates_file(populated_conn: sqlite3.Connection, tmp_path: Path) -> None:
    """Raw output has exactly four records, excluding parent game/player rows."""
    out = tmp_path / "raw.jsonl"

    assert export_dataset._export_raw(populated_conn, out) == 4

    assert out.is_file()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    records = [json.loads(line) for line in lines]
    assert Counter(record["type"] for record in records) == {
        "event": 2,
        "prediction": 1,
        "game_player": 1,
    }
    assert [record["type"] for record in records] == [
        "event",
        "event",
        "prediction",
        "game_player",
    ]


def test_export_raw_event_format(populated_conn: sqlite3.Connection, tmp_path: Path) -> None:
    """Event metadata stays flat and payload_json becomes a nested object."""
    out = tmp_path / "events.jsonl"
    export_dataset._export_raw(populated_conn, out)

    event = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert event == {
        "type": "event",
        "id": 1,
        "game_id": 1,
        "ts": "2026-10-31T12:00:00+00:00",
        "kind": "message",
        "actor_id": 101,
        "target_id": None,
        "phase": None,
        "day_number": None,
        "payload": {"author_id": 101, "author_name": "Алиса", "text": "Я мирный игрок."},
    }
    assert "Игрок уводит обсуждение" in out.read_text(encoding="utf-8")


def test_export_raw_prediction_format(populated_conn: sqlite3.Connection, tmp_path: Path) -> None:
    """Predictions retain their reasoning and label; game players retain outcomes."""
    out = tmp_path / "predictions.jsonl"
    export_dataset._export_raw(populated_conn, out)

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    prediction = next(record for record in records if record["type"] == "prediction")
    stored_ts = populated_conn.execute("SELECT ts FROM predictions WHERE id = 1").fetchone()[0]
    assert prediction == {
        "type": "prediction",
        "id": 1,
        "game_id": 1,
        "ts": stored_ts,
        "player_id": 101,
        "predicted_role": "Мафия",
        "confidence": 0.8,
        "reasoning": "Игрок уводит обсуждение от голосования.",
        "correct": 1,
    }
    game_player = next(record for record in records if record["type"] == "game_player")
    assert game_player == {
        "type": "game_player",
        "id": 1,
        "game_id": 1,
        "player_id": 101,
        "final_role": "Мафия",
        "survived": 0,
        "is_winner": 0,
    }


def test_export_chatml_creates_messages(populated_conn: sqlite3.Connection, tmp_path: Path) -> None:
    """One history row produces three ChatML messages and valid assistant JSON."""
    PredictionsRepository(populated_conn).insert_history(
        game_id=1,
        player_id=101,
        predicted_role="Мафия",
        confidence=0.8,
        reasoning="Игрок уводит обсуждение от голосования.",
    )
    out = tmp_path / "chatml.jsonl"

    assert export_dataset._export_chatml(populated_conn, out) == 1

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["game_id"] == 1
    stored_ts = populated_conn.execute(
        "SELECT ts FROM predictions_history WHERE id = 1"
    ).fetchone()[0]
    assert record["ts"] == stored_ts
    messages = record["messages"]
    assert len(messages) == 3
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert "аналитик-тренер" in messages[0]["content"]
    assert "Мафия" in messages[0]["content"]
    assert "prediction for player 101" in messages[1]["content"]
    assistant = json.loads(messages[2]["content"])
    assert assistant == {
        "analysis": [
            {
                "player_id": 101,
                "player_name": "player_101",
                "suspected_role": "Мафия",
                "confidence": 0.8,
                "reasons": [],
            }
        ],
        "game_read": "",
        "reasoning": "Игрок уводит обсуждение от голосования.",
    }


def test_export_chatml_missing_predictions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty history exports an empty file and the CLI succeeds."""
    out = tmp_path / "nested" / "empty.jsonl"

    assert export_dataset.main(["--db", ":memory:", "--out", str(out), "--format", "chatml"]) == 0

    assert out.is_file()
    assert out.read_text(encoding="utf-8") == ""
    assert f"Exported 0 records to {out} (format=chatml)" in capsys.readouterr().out
