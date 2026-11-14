"""Проверки гигиены, контрактов и разделения mock-данных."""

import json
import sqlite3
from pathlib import Path

import pytest

from infra.llm.schemas import AnalysisResponse
from scripts.finetune.prepare_dataset import build_record, check_balance, prepare
from scripts.finetune.train_lora import assess, validate_dataset


def answer(role):
    return {
        "analysis": [
            {
                "player_id": 1,
                "player_name": "Аня",
                "suspected_role": role,
                "confidence": 0.5,
                "reason": "Есть реплика.",
                "reasons": ["Есть реплика."],
                "pattern_ids": [],
                "pattern_descriptions": [],
            }
        ],
        "game_read": "Игра продолжается.",
    }


def make_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE events (id INTEGER, game_id INTEGER, ts TEXT, kind TEXT, payload_json TEXT)"
        )
        for game in range(1, 9):
            payload = {
                "author_id": 1,
                "author_name": "Аня",
                "text": "Я голосую против.",
                "final_role": "SECRET_ROLE",
                "private": {"token": "SECRET_NESTED"},
            }
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                (game * 10, game, "2026-01-01T00:00:00+00:00", "message", json.dumps(payload)),
            )
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                (
                    game * 10 + 1,
                    game,
                    "2026-01-01T00:01:00+00:00",
                    "role_reveal",
                    json.dumps(
                        {
                            "player_id": 1,
                            "player_name": "Аня",
                            "role_name_ru": "Сержант",
                            "final_role": "PRIVATE_REVEAL_FIELD",
                        }
                    ),
                ),
            )
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                (
                    game * 10 + 2,
                    game,
                    "2026-01-01T00:02:00+00:00",
                    "message",
                    json.dumps({"author_id": 1, "author_name": "Аня", "text": "FUTURE_SECRET"}),
                ),
            )
        conn.execute(
            "CREATE TABLE game_players (game_id INTEGER, player_id INTEGER, final_role TEXT)"
        )
        conn.execute("INSERT INTO game_players VALUES (1,1,'PRIVATE_TRUTH')")


def fixture_data(tmp_path):
    db = tmp_path / "test.db"
    make_db(db)
    communities = ["а", "а", "а", "а", "б", "б", "в", "в"]
    records = []
    for i, community in enumerate(communities, 1):
        records.append(
            {
                "db": str(db),
                "game_id": i,
                "cutoff_event_id": i * 10,
                "community": community,
                "format": "формат1" if i % 2 else "формат2",
                "outcome": "победа" if i % 2 else "поражение",
                "reply_type": "атака" if i % 2 else "защита",
                "reviewed": True,
                "response": answer("Доктор" if i % 2 else "Сержант"),
            }
        )
    config = {
        "dataset_dir": str(tmp_path / "dataset"),
        "train_communities": ["а"],
        "val_communities": ["б"],
        "test_communities": ["в"],
        "max_class_fraction": 0.7,
    }
    return db, config, records


def test_public_allowlist_and_schema(tmp_path):
    db, _, annotations = fixture_data(tmp_path)
    with sqlite3.connect(db) as conn:
        rec = build_record(conn, annotations[0])
    inputs = json.dumps(rec["messages"][:2], ensure_ascii=False)
    for secret in (
        "SECRET_ROLE",
        "SECRET_NESTED",
        "PRIVATE_TRUTH",
        "PRIVATE_REVEAL_FIELD",
        "FUTURE_SECRET",
        "final_role",
    ):
        assert secret not in inputs
    assert "Я голосую против." in inputs
    assert (
        AnalysisResponse.model_validate_json(rec["messages"][2]["content"])
        .analysis[0]
        .suspected_role
        == "Доктор"
    )
    assert set(rec) == {"messages"}


def test_message_from_dead_player_is_skipped_before_apply(tmp_path):
    db = tmp_path / "dead-player.db"
    make_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM events WHERE game_id=1")
        events = [
            (
                10,
                "2026-01-01T00:00:00+00:00",
                "message",
                {"author_id": 1, "author_name": "Аня", "text": "Первая реплика."},
            ),
            (
                11,
                "2026-01-01T00:01:00+00:00",
                "death",
                {"victim_id": 1, "victim_name": "Аня", "cause": "lynch"},
            ),
            (
                12,
                "2026-01-01T00:02:00+00:00",
                "message",
                {
                    "author_id": 1,
                    "author_name": "Аня",
                    "text": "__Результат проверки: мирный.__",
                },
            ),
        ]
        conn.executemany(
            "INSERT INTO events VALUES (?, 1, ?, ?, ?)",
            [
                (event_id, ts, kind, json.dumps(payload, ensure_ascii=False))
                for event_id, ts, kind, payload in events
            ],
        )
        record = build_record(
            conn,
            {"game_id": 1, "cutoff_event_id": 12, "response": answer("Доктор")},
        )

    input_text = json.dumps(record["messages"][:-1], ensure_ascii=False)
    assert "Первая реплика." in input_text
    assert "Результат проверки" not in input_text


def _build_record_after_alive_list_name_change(tmp_path, name, player_id, game_id):
    db = tmp_path / f"name-change-{game_id}.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE events (id INTEGER, game_id INTEGER, ts TEXT, kind TEXT, payload_json TEXT)"
        )
        events = [
            (
                1495,
                "2026-01-01T00:00:00+00:00",
                "message",
                {"author_id": player_id, "author_name": name, "text": "Первая реплика."},
            ),
            (
                1496,
                "2026-01-01T00:01:00+00:00",
                "message",
                {
                    "author_id": player_id,
                    "author_name": f"user_{player_id}",
                    "text": "Вторая реплика.",
                },
            ),
            (
                1506,
                "2026-01-01T00:02:00+00:00",
                "alive_list",
                {"alive_player_names": [name]},
            ),
            (
                1520,
                "2026-01-01T00:03:00+00:00",
                "death",
                {"victim_id": player_id, "victim_name": name, "cause": "lynch"},
            ),
        ]
        conn.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            [
                (event_id, game_id, ts, kind, json.dumps(payload, ensure_ascii=False))
                for event_id, ts, kind, payload in events
            ],
        )
        response = {
            "analysis": [
                {
                    "player_id": player_id,
                    "player_name": name,
                    "suspected_role": "Доктор",
                    "confidence": 0.5,
                    "reason": "Есть реплики.",
                    "reasons": ["Есть реплики."],
                    "pattern_ids": [],
                    "pattern_descriptions": [],
                }
            ],
            "game_read": "Игра продолжается.",
        }
        return build_record(
            conn,
            {"game_id": game_id, "cutoff_event_id": 1520, "response": response},
        )


def test_name_change_to_user_prefix_does_not_mark_player_dead(tmp_path):
    record = _build_record_after_alive_list_name_change(tmp_path, "Kizy", 6319004559, game_id=12)
    assert (
        AnalysisResponse.model_validate_json(record["messages"][-1]["content"])
        .analysis[0]
        .player_name
        == "Kizy"
    )


def test_real_name_with_user_prefix_is_preserved(tmp_path):
    name = "user_клубный_ник"
    record = _build_record_after_alive_list_name_change(tmp_path, name, 80001, game_id=15)
    prediction = AnalysisResponse.model_validate_json(record["messages"][-1]["content"]).analysis[0]
    assert prediction.player_name == name


@pytest.mark.parametrize(
    ("name", "player_id", "game_id"),
    [("Анна", 1743930626, 13), ("bl1ndii", 6335673414, 14)],
)
def test_same_for_other_players(tmp_path, name, player_id, game_id):
    record = _build_record_after_alive_list_name_change(tmp_path, name, player_id, game_id)
    prediction = AnalysisResponse.model_validate_json(record["messages"][-1]["content"]).analysis[0]
    assert prediction.player_name == name


def test_nonmonotonic_event_rows_are_sorted_by_timestamp_then_id(tmp_path):
    db = tmp_path / "out-of-order.db"
    make_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM events WHERE game_id=1")
        events = [
            (10, "2026-01-01T00:02:00+00:00", "Поздняя реплика."),
            (11, "2026-01-01T00:01:00+00:00", "Ранняя реплика."),
            (12, "2026-01-01T00:01:00+00:00", "Реплика с тем же временем."),
        ]
        conn.executemany(
            "INSERT INTO events VALUES (?, 1, ?, 'message', ?)",
            [
                (
                    event_id,
                    ts,
                    json.dumps(
                        {"author_id": 1, "author_name": "Аня", "text": text},
                        ensure_ascii=False,
                    ),
                )
                for event_id, ts, text in events
            ],
        )
        record = build_record(
            conn,
            {"game_id": 1, "cutoff_event_id": 12, "response": answer("Доктор")},
        )

    prompt_messages = json.dumps(record["messages"][:-1], ensure_ascii=False)
    first = prompt_messages.index("Ранняя реплика.")
    tied = prompt_messages.index("Реплика с тем же временем.")
    last = prompt_messages.index("Поздняя реплика.")
    assert first < tied < last


def test_mock_game_builds_a_valid_training_record(tmp_path):
    db, _, annotations = fixture_data(tmp_path)
    with sqlite3.connect(db) as conn:
        record = build_record(conn, annotations[0])

    assert len(record["messages"]) == 3
    assert record["messages"][0]["role"] == "system"
    assert record["messages"][1]["role"] == "user"
    assert record["messages"][2]["role"] == "assistant"
    AnalysisResponse.model_validate_json(record["messages"][2]["content"])


def test_prepare_splits_balance_and_validate(tmp_path):
    _, config, annotations = fixture_data(tmp_path)
    report = prepare(config, annotations)
    assert [report["splits"][s]["games"] for s in ("train", "val", "test")] == [4, 2, 2]
    assert len(validate_dataset(config)["train"]) == 4
    path = Path(config["dataset_dir"]) / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Хеш"):
        validate_dataset(config)


def test_unbalanced_rejected_and_old_data_deleted(tmp_path):
    _, config, annotations = fixture_data(tmp_path)
    prepare(config, annotations)
    for annotation in annotations:
        annotation["reply_type"] = "атака"
    with pytest.raises(ValueError, match="Дисбаланс"):
        prepare(config, annotations)
    assert not (Path(config["dataset_dir"]) / "train.jsonl").exists()
    with pytest.raises(ValueError):
        check_balance({"roles": {"а": 71, "б": 29}}, 0.7)
    check_balance({"roles": {"а": 70, "б": 30}}, 0.7)


def test_reject_split_leakage_and_bad_annotation(tmp_path):
    _, config, annotations = fixture_data(tmp_path)
    config["val_communities"] = ["а"]
    with pytest.raises(ValueError, match="Сообщество"):
        prepare(config, annotations)
    config["val_communities"] = ["б"]
    annotations[0]["reviewed"] = False
    with pytest.raises(ValueError, match="проверена"):
        prepare(config, annotations)


def test_loss_gate():
    cfg = {"max_val_loss": 2, "max_val_train_gap": 0.5, "max_val_loss_increase": 0.2}
    assert not assess(
        {
            "validation_history": [1.9, 1.5],
            "validation_completed": True,
            "training_steps": 1,
            "train_loss": 1.2,
            "test_loss": 1.6,
        },
        cfg,
    )
    assert assess(
        {
            "validation_history": [1.9, 3.0],
            "validation_completed": True,
            "training_steps": 1,
            "train_loss": 1.0,
            "test_loss": 2.0,
        },
        cfg,
    )
    assert assess(
        {
            "validation_history": [1.9, float("nan")],
            "validation_completed": True,
            "training_steps": 1,
            "train_loss": 1.0,
            "test_loss": 2.0,
        },
        cfg,
    )


def test_train_balancing_without_copying_and_no_eval_sampling(tmp_path):
    _, config, annotations = fixture_data(tmp_path)
    # Дополнительные снимки разных партий доминирующего класса.
    db = annotations[0]["db"]
    with sqlite3.connect(db) as conn:
        for game in range(9, 13):
            payload = {"author_id": 1, "author_name": "Аня", "text": "Реплика"}
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                (game * 10, game, "2026-01-01T00:00:00+00:00", "message", json.dumps(payload)),
            )
            annotations.append({**annotations[0], "game_id": game, "cutoff_event_id": game * 10})
    report = prepare(config, annotations)
    assert report["train_removed"] > 0
    assert report["splits"]["val"]["examples"] == 2
    assert report["splits"]["test"]["examples"] == 2
    counts = report["splits"]["train"]["distribution"]["roles"]
    assert max(counts.values()) / sum(counts.values()) <= 0.7


def test_public_reveal_visible_without_private_truth(tmp_path):
    db, _, annotations = fixture_data(tmp_path)
    annotations[0]["cutoff_event_id"] += 1
    annotations[0]["response"] = answer("Сержант")
    with sqlite3.connect(db) as conn:
        rec = build_record(conn, annotations[0])
        prompt = rec["messages"][1]["content"]
        assert "Раскрытые роли" in prompt
        assert "player_id=1 (Аня): Сержант" in prompt
        assert "PRIVATE_REVEAL_FIELD" not in prompt
        assert "PRIVATE_TRUTH" not in prompt
        assert "FUTURE_SECRET" not in prompt
        annotations[0]["response"] = answer("Доктор")
        with pytest.raises(ValueError, match="раскрытию"):
            build_record(conn, annotations[0])


def test_rejects_invalid_schema(tmp_path):
    db, _, annotations = fixture_data(tmp_path)
    annotations[0]["response"]["analysis"][0]["confidence"] = 5
    with sqlite3.connect(db) as conn, pytest.raises(ValueError):
        build_record(conn, annotations[0])


def test_quality_rejects_missing_validation_or_training_steps():
    cfg = {"max_val_loss": 2, "max_val_train_gap": 0.5, "max_val_loss_increase": 0.2}
    metrics = {"validation_history": [1.9, 1.5], "train_loss": 1.2, "test_loss": 1.6}
    assert assess(metrics, cfg)
    metrics.update(validation_completed=True, training_steps=0)
    assert assess(metrics, cfg)
    metrics["training_steps"] = 1
    assert not assess(metrics, cfg)
    metrics["validation_history"] = []
    assert assess(metrics, cfg)


def test_all_prepared_examples_match_contract(tmp_path):
    _, config, annotations = fixture_data(tmp_path)
    prepare(config, annotations)
    for name in ("train", "val", "test"):
        path = Path(config["dataset_dir"]) / f"{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows
        for row in rows:
            AnalysisResponse.model_validate_json(row["messages"][-1]["content"])


def test_train_removes_old_ready_status_when_validation_unavailable(tmp_path, monkeypatch):
    from scripts.finetune.template import JINJA, OLLAMA
    from scripts.finetune.train_lora import train

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    quality = adapter / "quality.json"
    quality.write_text('{"ready_for_use": true}', encoding="utf-8")
    config = {
        "adapter_dir": str(adapter),
        "chat_template": str(JINJA),
        "ollama_template": str(OLLAMA),
    }

    def no_validation(config):
        raise ValueError("Нет проверенной validation-выборки")

    monkeypatch.setattr("scripts.finetune.train_lora.validate_dataset", no_validation)
    with pytest.raises(ValueError, match="validation"):
        train(config)
    assert not quality.exists()
