"""Tests for the evaluate_model CLI script."""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from features.finetune.models import DatasetRecord, Message

evaluate = importlib.import_module("scripts.evaluate_model")


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


def _write_dataset(path: Path, records: list[DatasetRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")


def _model_output(player_id: int, role: str, confidence: float = 0.8) -> str:
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


def test_load_dataset_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    _write_dataset(path, [_make_record(), _make_record(player_id=102, game_id=2)])
    records = evaluate._load_dataset(path)
    assert len(records) == 2


def test_load_dataset_max_examples(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    _write_dataset(path, [_make_record(game_id=i) for i in range(10)])
    records = evaluate._load_dataset(path, max_examples=3)
    assert len(records) == 3


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    from domain.exceptions import TrainingError

    with pytest.raises(TrainingError):
        evaluate._load_dataset(tmp_path / "missing.jsonl")


def test_load_dataset_empty_file(tmp_path: Path) -> None:
    from domain.exceptions import TrainingError

    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(TrainingError):
        evaluate._load_dataset(path)


def test_main_missing_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI reports an error when the dataset is missing."""
    exit_code = evaluate.main(
        [
            "--dataset",
            str(tmp_path / "missing.jsonl"),
            "--base-model",
            "qwen/qwen3.5-9b",
        ]
    )
    assert exit_code == 1


def test_main_evaluates_base_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI evaluates the base model and writes a JSON report."""
    dataset_path = tmp_path / "sft_test.jsonl"
    report_path = tmp_path / "report.json"
    _write_dataset(dataset_path, [_make_record()])

    # Monkeypatch the heavy model loader to avoid importing torch/transformers.
    def fake_predict_fn(base_model: str, adapter: str | None, max_new_tokens: int):
        def predict(record: DatasetRecord) -> str:
            return _model_output(record.player_id, "Мафия", 0.9)

        return predict

    monkeypatch.setattr(evaluate, "_build_predict_fn", fake_predict_fn)

    exit_code = evaluate.main(
        [
            "--dataset",
            str(dataset_path),
            "--base-model",
            "qwen/qwen3.5-9b",
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Base" in output
    assert "100.0%" in output

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["base"]["examples"] == 1
    assert report["base"]["accuracy_exact"] == 1.0
    assert report["adapter"] is None


def test_main_evaluates_base_and_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI evaluates both base and adapter when --adapter is given."""
    dataset_path = tmp_path / "sft_test.jsonl"
    report_path = tmp_path / "report.json"
    _write_dataset(dataset_path, [_make_record()])

    def fake_predict_fn(base_model: str, adapter: str | None, max_new_tokens: int):
        def predict(record: DatasetRecord) -> str:
            role = "Мирный житель" if adapter else "Мафия"
            return _model_output(record.player_id, role, 0.8)

        return predict

    monkeypatch.setattr(evaluate, "_build_predict_fn", fake_predict_fn)

    exit_code = evaluate.main(
        [
            "--dataset",
            str(dataset_path),
            "--base-model",
            "qwen/qwen3.5-9b",
            "--adapter",
            "data/adapters/test",
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Base" in output
    assert "Adapter" in output

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["base"]["accuracy_exact"] == 1.0
    assert report["adapter"]["accuracy_exact"] == 0.0
    assert report["adapter"]["adapter"] == "data/adapters/test"
