"""End-to-end tests for the SFT dataset preparation CLI."""

from __future__ import annotations

import importlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from domain.events import (
    AliveListEvent,
    AliveListPayload,
    MessageEvent,
    MessagePayload,
    PhaseChangeEvent,
    PhaseChangePayload,
)
from domain.phases import Phase
from features.finetune.models import PredictionRow
from infra.db.connection import connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.events import EventsRepository
from infra.db.repositories.game_players import GamePlayersRepository
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository

# scripts is a CLI namespace, not an installed package. Match the existing CLI
# tests and avoid assigning a standalone script a second mypy module name.
prepare = importlib.import_module("scripts.prepare_finetune_dataset")

ROLES_PATH = Path(__file__).parents[3] / "configs" / "roles.yaml"
BASE_TS = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """Create an empty, fully migrated in-memory database."""
    connection = connect(":memory:")
    apply_migrations(connection)
    try:
        yield connection
    finally:
        connection.close()


def seed_game(conn: sqlite3.Connection) -> int:
    """Seed one game: two players, alive list, phase, message, prediction."""
    game_id = GamesRepository(conn).create_game(chat_id=-100, player_count=2)
    PlayersRepository(conn).upsert_player(player_id=101, username="alice", display_name="Алиса")
    PlayersRepository(conn).upsert_player(player_id=102, username="boris", display_name="Борис")
    events = EventsRepository(conn)
    events.insert_event(
        AliveListEvent(
            ts=BASE_TS,
            game_id=game_id,
            payload=AliveListPayload(
                alive_player_names=["user_101", "user_102"],
                role_count={"Мафия": 1, "Мирный житель": 1},
                total_players=2,
            ),
        )
    )
    events.insert_event(
        PhaseChangeEvent(
            ts=BASE_TS,
            game_id=game_id,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
            ),
        )
    )
    events.insert_event(
        MessageEvent(
            ts=BASE_TS,
            game_id=game_id,
            payload=MessagePayload(author_id=101, author_name="Алиса", text="Я мирный игрок."),
        )
    )
    return game_id


def seed_prediction_with_ground_truth(conn: sqlite3.Connection, game_id: int) -> int:
    """Insert a correct prediction and its final role."""
    predictions = PredictionsRepository(conn)
    prediction_id = predictions.upsert_prediction(
        game_id=game_id,
        player_id=101,
        predicted_role="Мафия",
        confidence=0.8,
        reasoning="Игрок уводит обсуждение от голосования.",
        reason="Уклончивые ответы.",
    )
    predictions.mark_correct(prediction_id=prediction_id, correct=True)
    GamePlayersRepository(conn).add_player(
        game_id=game_id,
        player_id=101,
        final_role="Мафия",
        survived=False,
        is_winner=False,
    )
    return prediction_id


def _registry():
    from domain.roles import load_roles

    return load_roles(ROLES_PATH)


def _jinja_env():
    import jinja2

    return jinja2.Environment(
        loader=jinja2.FileSystemLoader("infra/llm/prompts"),
        autoescape=False,
    )


def _run(conn: sqlite3.Connection, out: Path, **kwargs) -> tuple[int, Counter[str]]:
    """Run the dataset preparation with default dependencies."""
    kwargs.setdefault("registry", _registry())
    kwargs.setdefault("club_patterns", [])
    kwargs.setdefault("jinja_env", _jinja_env())
    kwargs.setdefault("max_messages", 200)
    return prepare.prepare_dataset(conn=conn, out=out, **kwargs)


def test_prepare_current_creates_one_record(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """One labeled prediction with ground truth yields one JSONL record."""
    game_id = seed_game(conn)
    seed_prediction_with_ground_truth(conn, game_id)
    out = tmp_path / "sft.jsonl"

    records, skipped = _run(conn, out, source="current")

    assert records == 1
    assert skipped == Counter()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["game_id"] == game_id
    assert record["correct"] is True
    assert record["ground_truth_role"] == "Мафия"
    assert record["predicted_role"] == "Мафия"
    assert len(record["messages"]) == 3
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
    assistant = json.loads(record["messages"][2]["content"])
    assert assistant["analysis"][0]["suspected_role"] == "Мафия"
    assert assistant["analysis"][0]["confidence"] == 0.8
    assert assistant["analysis"][0]["reason"] == "Уклончивые ответы."
    assert assistant["reasoning"] == "Игрок уводит обсуждение от голосования."


def test_prepare_skips_predictions_without_ground_truth(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """Predictions lacking a final role are skipped with a counter."""
    game_id = seed_game(conn)
    PredictionsRepository(conn).upsert_prediction(
        game_id=game_id,
        player_id=101,
        predicted_role="Мафия",
        confidence=0.8,
        reasoning="",
    )
    out = tmp_path / "sft.jsonl"

    records, skipped = _run(conn, out, source="current")

    assert records == 0
    assert skipped["no_ground_truth"] == 1


def test_prepare_only_correct_filter(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """--only-correct drops wrong predictions even with ground truth."""
    game_id = seed_game(conn)
    predictions = PredictionsRepository(conn)
    prediction_id = predictions.upsert_prediction(
        game_id=game_id,
        player_id=101,
        predicted_role="Мирный житель",
        confidence=0.7,
        reasoning="",
    )
    predictions.mark_correct(prediction_id=prediction_id, correct=False)
    GamePlayersRepository(conn).add_player(game_id=game_id, player_id=101, final_role="Мафия")
    out = tmp_path / "sft.jsonl"

    records, skipped = _run(conn, out, source="current", only_correct=True)
    assert records == 0
    assert skipped["incorrect"] == 1


def test_prepare_wrong_prediction_is_corrected(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """A wrong prediction with ground truth becomes a corrected example."""
    game_id = seed_game(conn)
    predictions = PredictionsRepository(conn)
    prediction_id = predictions.upsert_prediction(
        game_id=game_id,
        player_id=101,
        predicted_role="Мирный житель",
        confidence=0.7,
        reasoning="Вася вёл себя мирно.",
    )
    predictions.mark_correct(prediction_id=prediction_id, correct=False)
    GamePlayersRepository(conn).add_player(game_id=game_id, player_id=101, final_role="Мафия")
    out = tmp_path / "sft.jsonl"

    records, _ = _run(conn, out, source="current")

    assert records == 1
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assistant = json.loads(record["messages"][2]["content"])
    assert assistant["analysis"][0]["suspected_role"] == "Мафия"
    assert assistant["reasoning"] is None


def test_prepare_skips_null_role_predictions(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """Null-role predictions (silent players) are skipped."""
    game_id = seed_game(conn)
    PredictionsRepository(conn).upsert_prediction(
        game_id=game_id,
        player_id=101,
        predicted_role=None,
        confidence=0.0,
        reasoning="",
    )
    GamePlayersRepository(conn).add_player(game_id=game_id, player_id=101, final_role="Мафия")
    out = tmp_path / "sft.jsonl"

    records, skipped = _run(conn, out, source="current")
    assert records == 0
    assert skipped["no_role_predicted"] == 1


def test_prepare_history_source(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """The history source uses predictions_history and dedupes repeats."""
    game_id = seed_game(conn)
    GamePlayersRepository(conn).add_player(game_id=game_id, player_id=101, final_role="Мафия")
    predictions = PredictionsRepository(conn)
    for _ in range(3):
        predictions.insert_history(
            game_id=game_id,
            player_id=101,
            predicted_role="Мафия",
            confidence=0.8,
            reasoning="",
        )
    out = tmp_path / "sft.jsonl"

    records, skipped = _run(conn, out, source="history")
    assert records == 1
    assert skipped["duplicate_history"] == 2


def test_prepare_anonymize(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """Anonymized records use player_N aliases for ids and names."""
    game_id = seed_game(conn)
    seed_prediction_with_ground_truth(conn, game_id)
    out = tmp_path / "sft.jsonl"

    records, _ = _run(conn, out, source="current", anonymize=True)

    assert records == 1
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert record["player_id"] == 1
    assistant = json.loads(record["messages"][2]["content"])
    assert assistant["analysis"][0]["player_id"] == 1
    assert assistant["analysis"][0]["player_name"] == "player_1"
    assert "user_101" not in record["messages"][1]["content"]


def test_prepare_cli_main(
    conn: sqlite3.Connection, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI entry point writes the file and reports the count."""
    game_id = seed_game(conn)
    seed_prediction_with_ground_truth(conn, game_id)
    db = tmp_path / "mafia.db"
    with sqlite3.connect(db) as source:
        conn.backup(source)
    out = tmp_path / "nested" / "sft.jsonl"

    exit_code = prepare.main(
        [
            "--db",
            str(db),
            "--out",
            str(out),
            "--source",
            "current",
        ]
    )

    assert exit_code == 0
    assert out.is_file()
    assert "Prepared 1 examples" in capsys.readouterr().out


def test_prediction_row_parses_sqlite_row(conn: sqlite3.Connection) -> None:
    """PredictionRow.from_sqlite_row handles naive CURRENT_TIMESTAMP values."""
    game_id = GamesRepository(conn).create_game(chat_id=-100, player_count=1)
    PlayersRepository(conn).upsert_player(player_id=101, username="alice", display_name="Алиса")
    prediction_id = PredictionsRepository(conn).upsert_prediction(
        game_id=game_id,
        player_id=101,
        predicted_role="Мафия",
        confidence=0.8,
        reasoning="",
    )
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    parsed = PredictionRow.from_sqlite_row(row)
    assert parsed.player_id == 101
    assert parsed.predicted_role == "Мафия"
    assert parsed.ts.tzinfo is not None
    assert parsed.correct is None


def test_prepare_split_writes_three_files(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """--split writes three files with whole games in exactly one split."""
    game_ids = []
    for _ in range(6):
        game_id = seed_game(conn)
        seed_prediction_with_ground_truth(conn, game_id)
        game_ids.append(game_id)
    out = tmp_path / "sft.jsonl"

    records, skipped = _run(conn, out, source="current", split=True)

    assert records == 6
    assert skipped == Counter()
    train = out.with_name("sft_train.jsonl")
    val = out.with_name("sft_val.jsonl")
    test = out.with_name("sft_test.jsonl")
    assert train.is_file() and val.is_file() and test.is_file()
    # Every game appears in exactly one split.
    seen: set[int] = set()
    for path in (train, val, test):
        for line in path.read_text(encoding="utf-8").splitlines():
            game_id = json.loads(line)["game_id"]
            assert game_id not in seen
            seen.add(game_id)
    assert seen == set(game_ids)


def test_prepare_split_is_deterministic(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """The same seed yields the same split assignment."""
    game_ids = []
    for _ in range(8):
        game_id = seed_game(conn)
        seed_prediction_with_ground_truth(conn, game_id)
        game_ids.append(game_id)

    first = tmp_path / "first" / "sft.jsonl"
    second = tmp_path / "second" / "sft.jsonl"
    _run(conn, first, source="current", split=True)
    _run(conn, second, source="current", split=True)

    for name in ("train", "val", "test"):
        lines_first = [
            json.loads(line)["game_id"]
            for line in first.with_name(f"sft_{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        lines_second = [
            json.loads(line)["game_id"]
            for line in second.with_name(f"sft_{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert sorted(lines_first) == sorted(lines_second)


def test_prepare_split_respects_ratios(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """test_ratio/val_ratio are fractions of games, not examples."""
    for _ in range(20):
        game_id = seed_game(conn)
        seed_prediction_with_ground_truth(conn, game_id)
    out = tmp_path / "sft.jsonl"

    records, _ = _run(conn, out, source="current", split=True, test_ratio=0.5, val_ratio=0.25)

    assert records == 20
    counts = {
        name: len(out.with_name(f"sft_{name}.jsonl").read_text(encoding="utf-8").splitlines())
        for name in ("train", "val", "test")
    }
    assert counts == {"train": 5, "val": 5, "test": 10}


def test_prepare_split_rejects_bad_ratios(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """Ratios that cannot form a split raise ValueError."""
    game_id = seed_game(conn)
    seed_prediction_with_ground_truth(conn, game_id)
    out = tmp_path / "sft.jsonl"

    with pytest.raises(ValueError):
        _run(conn, out, source="current", split=True, test_ratio=0.7, val_ratio=0.4)


def test_prepare_split_cli_main(
    conn: sqlite3.Connection, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI --split flag writes three files and reports the count."""
    for _ in range(4):
        game_id = seed_game(conn)
        seed_prediction_with_ground_truth(conn, game_id)
    db = tmp_path / "mafia.db"
    with sqlite3.connect(db) as source:
        conn.backup(source)
    out = tmp_path / "sft.jsonl"

    exit_code = prepare.main(
        [
            "--db",
            str(db),
            "--out",
            str(out),
            "--source",
            "current",
            "--split",
            "--test-ratio",
            "0.5",
            "--val-ratio",
            "0.25",
        ]
    )

    assert exit_code == 0
    assert out.with_name("sft_train.jsonl").is_file()
    assert out.with_name("sft_val.jsonl").is_file()
    assert out.with_name("sft_test.jsonl").is_file()
    output = capsys.readouterr().out
    assert "Prepared 4 examples" in output
