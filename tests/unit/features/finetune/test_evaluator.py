"""Unit tests for the fine-tune evaluation metrics."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from domain.roles import load_roles
from features.finetune.evaluator import (
    compute_report,
    evaluate_records,
    extract_json,
    parse_prediction,
    render_table,
)
from features.finetune.models import DatasetRecord, Message

ROLES_PATH = Path(__file__).parents[4] / "configs" / "roles.yaml"


@pytest.fixture
def registry():
    return load_roles(ROLES_PATH)


def _make_record(
    player_id: int = 101,
    ground_truth: str = "Мафия",
    game_id: int = 1,
) -> DatasetRecord:
    return DatasetRecord(
        messages=[
            Message(role="system", content="system prompt"),
            Message(role="user", content="user context"),
            Message(role="assistant", content="assistant response"),
        ],
        game_id=game_id,
        prediction_id=1,
        player_id=player_id,
        ts=datetime(2026, 1, 1, tzinfo=UTC),
        correct=True,
        ground_truth_role=ground_truth,
    )


def _model_output(player_id: int, role: str | None, confidence: float = 0.8) -> str:
    payload: dict[str, Any] = {
        "analysis": [
            {
                "player_id": player_id,
                "player_name": "alice",
                "suspected_role": role,
                "confidence": confidence,
                "reason": "test",
                "reasons": ["test"],
            }
        ],
        "game_read": "test",
    }
    return json.dumps(payload, ensure_ascii=False)


# --- extract_json ---------------------------------------------------------


def test_extract_json_plain_object() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_block() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_with_surrounding_text() -> None:
    raw = 'Here is the result:\n{"analysis": []}\nThanks!'
    assert extract_json(raw) == {"analysis": []}


def test_extract_json_invalid() -> None:
    assert extract_json("not json at all") is None


def test_extract_json_non_dict() -> None:
    assert extract_json("[1, 2, 3]") is None


# --- parse_prediction ------------------------------------------------------


def test_parse_prediction_matching_player() -> None:
    raw = _model_output(101, "Мафия", 0.9)
    role, confidence = parse_prediction(raw, 101)
    assert role == "Мафия"
    assert confidence == 0.9


def test_parse_prediction_non_matching_player() -> None:
    raw = _model_output(102, "Мирный житель")
    role, confidence = parse_prediction(raw, 101)
    assert role is None
    assert confidence == 0.0


def test_parse_prediction_invalid_json() -> None:
    role, confidence = parse_prediction("not json", 101)
    assert role is None
    assert confidence == 0.0


def test_parse_prediction_null_role() -> None:
    raw = _model_output(101, None, 0.0)
    role, confidence = parse_prediction(raw, 101)
    assert role is None
    assert confidence == 0.0


# --- evaluate_records -----------------------------------------------------


def test_evaluate_records_correct_prediction(registry) -> None:
    records = [_make_record(ground_truth="Мафия")]
    predict_fn = lambda record: _model_output(record.player_id, "Мафия", 0.9)
    samples = evaluate_records(records, predict_fn=predict_fn, registry=registry)
    assert len(samples) == 1
    assert samples[0].exact_correct is True
    assert samples[0].faction_correct is True


def test_evaluate_records_wrong_role(registry) -> None:
    records = [_make_record(ground_truth="Мафия")]
    predict_fn = lambda record: _model_output(record.player_id, "Мирный житель", 0.7)
    samples = evaluate_records(records, predict_fn=predict_fn, registry=registry)
    assert samples[0].exact_correct is False
    assert samples[0].faction_correct is False


def test_evaluate_records_wrong_faction_same_category(registry) -> None:
    # "Дон" (don) and "Мафия" are both mafia faction → faction_correct is True
    # even though exact_correct is False.
    records = [_make_record(ground_truth="Дон")]
    predict_fn = lambda record: _model_output(record.player_id, "Мафия", 0.6)
    samples = evaluate_records(records, predict_fn=predict_fn, registry=registry)
    assert samples[0].exact_correct is False
    assert samples[0].faction_correct is True


def test_evaluate_records_null_prediction(registry) -> None:
    records = [_make_record(ground_truth="Мафия")]
    predict_fn = lambda record: _model_output(record.player_id, None, 0.0)
    samples = evaluate_records(records, predict_fn=predict_fn, registry=registry)
    assert samples[0].exact_correct is False
    assert samples[0].faction_correct is False


# --- compute_report -------------------------------------------------------


def test_compute_report_basic(registry) -> None:
    records = [
        _make_record(player_id=101, ground_truth="Мафия", game_id=1),
        _make_record(player_id=102, ground_truth="Мирный житель", game_id=2),
    ]
    predict_fn = lambda record: _model_output(record.player_id, record.ground_truth_role, 0.8)
    samples = evaluate_records(records, predict_fn=predict_fn, registry=registry)
    report = compute_report(samples, base_model="qwen/qwen3.5-9b", adapter=None, registry=registry)
    assert report.examples == 2
    assert report.accuracy_exact == 1.0
    assert "mafia" in report.accuracy_by_faction
    assert "town" in report.accuracy_by_faction
    assert report.avg_confidence_correct == 0.8
    assert report.avg_confidence_incorrect == 0.0


def test_compute_report_empty(registry) -> None:
    report = compute_report([], base_model="qwen/qwen3.5-9b", adapter=None, registry=registry)
    assert report.examples == 0
    assert report.accuracy_exact == 0.0


def test_compute_report_mixed_results(registry) -> None:
    records = [
        _make_record(player_id=101, ground_truth="Мафия", game_id=1),
        _make_record(player_id=102, ground_truth="Мирный житель", game_id=2),
        _make_record(player_id=103, ground_truth="Дон", game_id=3),
    ]
    predictions = {
        101: ("Мафия", 0.9),
        102: ("Мафия", 0.4),
        103: ("Мирный житель", 0.3),
    }
    predict_fn = lambda record: _model_output(
        record.player_id,
        predictions[record.player_id][0],
        predictions[record.player_id][1],
    )
    samples = evaluate_records(records, predict_fn=predict_fn, registry=registry)
    report = compute_report(samples, base_model="test", adapter=None, registry=registry)
    assert report.examples == 3
    assert report.accuracy_exact == 1 / 3
    # 1 correct, 2 incorrect
    assert report.avg_confidence_correct == 0.9
    assert report.avg_confidence_incorrect == (0.4 + 0.3) / 2


# --- render_table ---------------------------------------------------------


def test_render_table_base_only(registry) -> None:
    records = [_make_record(ground_truth="Мафия")]
    predict_fn = lambda record: _model_output(record.player_id, "Мафия", 0.9)
    samples = evaluate_records(records, predict_fn=predict_fn, registry=registry)
    report = compute_report(samples, base_model="qwen/qwen3.5-9b", adapter=None, registry=registry)
    table = render_table(report, None)
    assert "Base" in table
    assert "Accuracy (exact)" in table
    assert "100.0%" in table


def test_render_table_with_adapter(registry) -> None:
    records = [_make_record(ground_truth="Мафия")]
    base_predict = lambda record: _model_output(record.player_id, "Мафия", 0.9)
    adapter_predict = lambda record: _model_output(record.player_id, "Мирный житель", 0.5)
    base_samples = evaluate_records(records, predict_fn=base_predict, registry=registry)
    adapter_samples = evaluate_records(records, predict_fn=adapter_predict, registry=registry)
    base_report = compute_report(
        base_samples,
        base_model="qwen/qwen3.5-9b",
        adapter=None,
        registry=registry,
    )
    adapter_report = compute_report(
        adapter_samples,
        base_model="qwen/qwen3.5-9b",
        adapter="data/adapters/test",
        registry=registry,
    )
    table = render_table(base_report, adapter_report)
    assert "Base" in table
    assert "Adapter" in table
    assert "100.0%" in table
    assert "0.0%" in table
