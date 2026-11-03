"""Offline JSONL export tests using the real migrations and in-memory SQLite."""

from __future__ import annotations

import importlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from domain.events import (
    AliveListEvent,
    AliveListPayload,
    DeathEvent,
    DeathPayload,
    EndGameEvent,
    EndGamePayload,
    MessageEvent,
    MessagePayload,
    PhaseChangeEvent,
    PhaseChangePayload,
    RoleRevealEvent,
    RoleRevealPayload,
    VoteEvent,
    VotePayload,
)
from domain.phases import Phase
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


def _copy_db(conn: sqlite3.Connection, path: Path) -> None:
    """Copy a populated in-memory database for CLI tests."""
    with closing(sqlite3.connect(path)) as target:
        conn.backup(target)


def test_export_raw_anonymize(
    populated_conn: sqlite3.Connection, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI consistently replaces both IDs and names across all raw rows."""
    db, out = tmp_path / "mafia.db", tmp_path / "raw.jsonl"
    _copy_db(populated_conn, db)

    assert (
        export_dataset.main(["--db", str(db), "--out", str(out), "--format", "raw", "--anonymize"])
        == 0
    )

    message, vote, prediction, game_player = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
    ]
    assert message["actor_id"] == "player_1"
    assert message["target_id"] is None
    assert message["payload"] == {
        "author_id": "player_1",
        "author_name": "player_1",
        "text": "Я мирный игрок.",
    }
    assert vote["actor_id"] == "player_1"
    assert vote["target_id"] == "player_2"
    assert vote["payload"] == {
        "voter_id": "player_1",
        "voter_name": "player_1",
        "target_id": "player_2",
        "target_name": "player_2",
    }
    assert prediction["player_id"] == game_player["player_id"] == "player_1"
    assert isinstance(prediction["player_id"], str)
    assert "PII" in capsys.readouterr().err
    # Pseudonymization must not overwrite identities in the database used by the CLI.
    with closing(sqlite3.connect(db)) as source:
        original = json.loads(
            source.execute("SELECT payload_json FROM events WHERE id = 1").fetchone()[0]
        )
    assert original["author_id"] == 101
    assert original["author_name"] == "Алиса"


def test_export_raw_no_anonymize_default(
    populated_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Without the opt-in flag the original names and integer IDs are retained."""
    db, out = tmp_path / "mafia.db", tmp_path / "raw.jsonl"
    _copy_db(populated_conn, db)

    assert export_dataset.main(["--db", str(db), "--out", str(out), "--format", "raw"]) == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert records[0]["actor_id"] == 101
    assert records[0]["payload"]["author_id"] == 101
    assert records[0]["payload"]["author_name"] == "Алиса"
    assert records[1]["payload"]["target_name"] == "Борис"
    assert records[2]["player_id"] == records[3]["player_id"] == 101


def test_export_chatml_anonymize(
    populated_conn: sqlite3.Connection, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ChatML user text and assistant JSON use the same stable aliases."""
    predictions = PredictionsRepository(populated_conn)
    for player_id in (101, 102, 101):
        predictions.insert_history(
            game_id=1,
            player_id=player_id,
            predicted_role="Мафия",
            confidence=0.8,
            reasoning="Игрок подозрителен.",
        )
    # History timestamps have one-second precision, so equal timestamps are common.
    populated_conn.execute("UPDATE predictions_history SET ts = '2026-10-31T12:00:00+00:00'")
    db, out = tmp_path / "mafia.db", tmp_path / "chatml.jsonl"
    _copy_db(populated_conn, db)

    assert (
        export_dataset.main(
            ["--db", str(db), "--out", str(out), "--format", "chatml", "--anonymize"]
        )
        == 0
    )

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    for record, alias in zip(records, ("player_1", "player_2", "player_1"), strict=True):
        messages = record["messages"]
        assert messages[1]["content"] == (
            f"Game {record['game_id']}, prediction for {alias} at {record['ts']}."
        )
        assistant = json.loads(messages[2]["content"])
        assert assistant["analysis"][0]["player_id"] == int(alias.removeprefix("player_"))
        assert isinstance(assistant["analysis"][0]["player_id"], int)
        assert assistant["analysis"][0]["player_name"] == alias
        assert "101" not in messages[2]["content"]
        assert "102" not in messages[2]["content"]
    assert "PII" in capsys.readouterr().err


def test_export_raw_anonymizes_all_event_types(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """Name-only lists, votes, deaths, and reveals do not leak player identities."""
    game_id = GamesRepository(conn).create_game(chat_id=-100, player_count=3)
    events = EventsRepository(conn)
    ts = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    events.insert_event(
        AliveListEvent(
            ts=ts,
            game_id=game_id,
            payload=AliveListPayload(
                alive_player_names=["Алиса", "Борис", "Призрак"],
                role_count={"Мафия": 1},
                total_players=3,
            ),
        )
    )
    events.insert_event(
        VoteEvent(
            ts=ts,
            game_id=game_id,
            payload=VotePayload(
                voter_id=101, voter_name="Алиса", target_id=None, target_name="Призрак"
            ),
        )
    )
    for player_id, name in ((101, "Алиса"), (102, "Борис")):
        events.insert_event(
            MessageEvent(
                ts=ts,
                game_id=game_id,
                payload=MessagePayload(author_id=player_id, author_name=name, text="Привет!"),
            )
        )
    events.insert_event(
        VoteEvent(
            ts=ts,
            game_id=game_id,
            payload=VotePayload(voter_id=101, voter_name="Алиса", target_id=102, target_name=None),
        )
    )
    events.insert_event(
        DeathEvent(
            ts=ts,
            game_id=game_id,
            payload=DeathPayload(victim_id=101, victim_name="Алиса", cause="lynch"),
        )
    )
    events.insert_event(
        RoleRevealEvent(
            ts=ts,
            game_id=game_id,
            payload=RoleRevealPayload(player_id=102, player_name="Борис", role_name_ru="Мафия"),
        )
    )
    events.insert_event(
        EndGameEvent(
            ts=ts,
            game_id=game_id,
            payload=EndGamePayload(
                winners=[("Алиса", "Мафия")],
                losers=[("Борис", "Мирный"), ("Призрак", "Мирный")],
                duration_minutes=15,
            ),
        )
    )
    events.insert_event(
        PhaseChangeEvent(
            ts=ts,
            game_id=game_id,
            payload=PhaseChangePayload(from_phase=Phase.DAY, to_phase=Phase.VOTE, day_number=1),
        )
    )
    out = tmp_path / "all-events.jsonl"

    assert export_dataset._export_raw(conn, out, anonymize=True) == 9

    contents = out.read_text(encoding="utf-8")
    assert all(name not in contents for name in ("Алиса", "Борис", "Призрак"))
    alive, vote, alice, boris, partial_vote, death, reveal, ending, phase = [
        json.loads(line) for line in contents.splitlines()
    ]
    assert alive["payload"]["alive_player_names"] == ["player_1", "player_2", "player_3"]
    assert alive["payload"]["role_count"] == {"Мафия": 1}
    assert vote["actor_id"] == vote["payload"]["voter_id"] == "player_1"
    assert vote["target_id"] is vote["payload"]["target_id"] is None
    assert vote["payload"]["target_name"] == "player_3"
    assert alice["payload"]["author_name"] == alice["payload"]["author_id"] == "player_1"
    assert boris["payload"]["author_name"] == boris["payload"]["author_id"] == "player_2"
    assert partial_vote["target_id"] == partial_vote["payload"]["target_id"] == "player_2"
    assert partial_vote["payload"]["target_name"] is None
    assert death["actor_id"] == death["payload"]["victim_id"] == "player_1"
    assert death["payload"]["victim_name"] == "player_1"
    assert reveal["actor_id"] == reveal["payload"]["player_id"] == "player_2"
    assert reveal["payload"]["player_name"] == "player_2"
    assert ending["payload"]["winners"] == [["player_1", "Мафия"]]
    assert ending["payload"]["losers"] == [["player_2", "Мирный"], ["player_3", "Мирный"]]
    assert phase["payload"] == {"from_phase": "day", "to_phase": "vote", "day_number": 1}


def test_export_raw_does_not_conflate_ambiguous_names(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """An unresolved duplicate name is distinct from both ID-bearing players."""
    game_id = GamesRepository(conn).create_game(chat_id=-100, player_count=2)
    events = EventsRepository(conn)
    ts = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    events.insert_event(
        AliveListEvent(
            ts=ts,
            game_id=game_id,
            payload=AliveListPayload(alive_player_names=["Одинаковое имя"]),
        )
    )
    for player_id in (101, 102):
        events.insert_event(
            MessageEvent(
                ts=ts,
                game_id=game_id,
                payload=MessagePayload(
                    author_id=player_id, author_name="Одинаковое имя", text="Привет!"
                ),
            )
        )
    out = tmp_path / "ambiguous.jsonl"

    assert export_dataset._export_raw(conn, out, anonymize=True) == 3

    alive, first, second = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
    ]
    assert alive["payload"]["alive_player_names"] == ["player_1"]
    assert first["payload"]["author_name"] == first["actor_id"] == "player_2"
    assert second["payload"]["author_name"] == second["actor_id"] == "player_3"


def test_export_raw_anonymize_ids_without_events(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """Predictions and game-player rows receive aliases even without any events."""
    game_id = GamesRepository(conn).create_game(chat_id=-100, player_count=2)
    for player_id in (101, 102):
        PlayersRepository(conn).upsert_player(
            player_id=player_id, username=f"user{player_id}", display_name="Test"
        )
    PredictionsRepository(conn).insert_prediction(
        game_id=game_id, player_id=102, predicted_role="Мафия", confidence=0.8, reasoning=""
    )
    GamePlayersRepository(conn).add_player(game_id=game_id, player_id=101)
    GamePlayersRepository(conn).add_player(game_id=game_id, player_id=102)
    conn.execute("INSERT INTO game_players(game_id, player_id) VALUES (?, NULL)", (game_id,))
    out = tmp_path / "without-events.jsonl"

    assert export_dataset._export_raw(conn, out, anonymize=True) == 4

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["player_id"] for row in records] == ["player_1", "player_2", "player_1", None]


def test_export_raw_anonymize_uses_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent write cannot change the events between the index and export passes."""
    db, out = tmp_path / "mafia.db", tmp_path / "snapshot.jsonl"
    reader = connect(db)
    apply_migrations(reader)
    game_id = GamesRepository(reader).create_game(chat_id=-100, player_count=2)
    ts = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    EventsRepository(reader).insert_event(
        MessageEvent(
            ts=ts,
            game_id=game_id,
            payload=MessagePayload(author_id=101, author_name="Алиса", text="Привет!"),
        )
    )
    writer = connect(db)
    original_collect = export_dataset._collect_aliases

    def collect_then_write(connection: sqlite3.Connection) -> object:
        aliases = original_collect(connection)
        EventsRepository(writer).insert_event(
            MessageEvent(
                ts=ts,
                game_id=game_id,
                payload=MessagePayload(author_id=102, author_name="Борис", text="Привет!"),
            )
        )
        return aliases

    monkeypatch.setattr(export_dataset, "_collect_aliases", collect_then_write)
    try:
        assert export_dataset._export_raw(reader, out, anonymize=True) == 1
        assert not reader.in_transaction
        assert writer.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
        record = json.loads(out.read_text(encoding="utf-8"))
        assert record["payload"]["author_name"] == "player_1"
    finally:
        reader.close()
        writer.close()


def test_export_raw_anonymize_preserves_caller_transaction(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """A pre-existing caller transaction is neither committed nor rolled back."""
    conn.execute("BEGIN")
    game_id = GamesRepository(conn).create_game(chat_id=-100, player_count=1)
    EventsRepository(conn).insert_event(
        MessageEvent(
            ts=datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
            game_id=game_id,
            payload=MessagePayload(author_id=101, author_name="Алиса", text="Привет!"),
        )
    )

    assert export_dataset._export_raw(conn, tmp_path / "in-transaction.jsonl", anonymize=True) == 1
    assert conn.in_transaction
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


@pytest.mark.parametrize(
    "payload",
    [
        "not an object",
        {"author_id": "101", "author_name": "PrivateName"},
        {"author_id": 101, "author_name": ["PrivateName"]},
        {"alive_player_names": "PrivateName"},
        {"winners": [["PrivateName", "Role"], "bad entry"]},
        {"winners": [["PrivateName", "Role", "unexpected"]]},
    ],
)
def test_export_raw_anonymize_rejects_malformed_identity_data(
    conn: sqlite3.Connection, tmp_path: Path, payload: object
) -> None:
    """Invalid identity fields are not silently copied into an anonymous export."""
    game_id = GamesRepository(conn).create_game(chat_id=-100, player_count=1)
    conn.execute(
        "INSERT INTO events(game_id, ts, kind, actor_id, payload_json) VALUES (?, ?, ?, ?, ?)",
        (game_id, "2026-10-31T12:00:00+00:00", "message", 101, json.dumps(payload)),
    )

    with pytest.raises((TypeError, ValueError)) as error:
        export_dataset._export_raw(conn, tmp_path / "bad.jsonl", anonymize=True)
    assert "PrivateName" not in str(error.value)
    assert not conn.in_transaction
