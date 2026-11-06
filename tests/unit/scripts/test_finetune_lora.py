"""Tests for the fine-tune CLI: argument parsing and error paths."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from domain.exceptions import TrainingError
from features.finetune.models import TrainingOptions

# scripts is a CLI namespace, not an installed package. Match the existing CLI
# tests and avoid assigning a standalone script a second mypy module name.
finetune_lora = importlib.import_module("scripts.finetune_lora")

ROOT = Path(__file__).resolve().parents[3]


def make_config_file(tmp_path: Path, **overrides) -> Path:
    """Write a valid settings file with a finetune section."""
    import yaml

    example = ROOT / "configs" / "settings.example.yaml"
    settings = yaml.safe_load(example.read_text(encoding="utf-8"))
    settings["telegram"].update(
        {
            "api_id": 12345,
            "api_hash": "0123456789abcdef0123456789abcdef",
            "chat_id": -100123,
            "bot_username": "@FakeMafiaBot",
            "bot_id": 42,
        }
    )
    settings["db"]["path"] = str(tmp_path / "mafia.db")
    settings["roles_file"] = str(ROOT / "configs" / "roles.yaml")
    settings["club_patterns_file"] = str(tmp_path / "patterns.yaml")
    settings["logging"]["dir"] = str(tmp_path / "logs")
    for key, value in overrides.items():
        settings.setdefault("finetune", {})[key] = value
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        yaml.safe_dump(settings, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_cli_missing_dataset_returns_error(tmp_path: Path) -> None:
    """A missing dataset file is reported as a CLI error."""
    config = make_config_file(tmp_path)
    exit_code = finetune_lora.main(
        [
            "--config",
            str(config),
            "--dataset",
            str(tmp_path / "missing.jsonl"),
        ]
    )
    assert exit_code == 1


def test_cli_parses_overrides(tmp_path: Path) -> None:
    """CLI overrides replace config values in the resolved options."""
    config = make_config_file(tmp_path)
    args = finetune_lora._build_arg_namespace(
        [
            "--base-model",
            "custom/model",
            "--method",
            "lora",
            "--backend",
            "peft",
            "--epochs",
            "5",
        ]
    )
    from app.config import load_config

    options = finetune_lora._resolve_options(args, load_config(config))
    assert options.base_model == "custom/model"
    assert options.method == "lora"
    assert options.backend == "peft"
    assert options.epochs == 5


def test_resolve_options_defaults(tmp_path: Path) -> None:
    """Without overrides the config section values are used."""
    config = make_config_file(tmp_path)
    args = finetune_lora._build_arg_namespace([])
    from app.config import load_config

    options = finetune_lora._resolve_options(args, load_config(config))
    assert options.base_model == "qwen/qwen3.5-9b"
    assert options.method == "qlora"
    assert options.backend == "unsloth"
    assert options.rank == 16
    assert options.alpha == 32


def test_timestamped_dir(tmp_path: Path) -> None:
    """The adapter directory name matches YYYY-MM-DD_HH-MM."""
    base = str(tmp_path / "adapters")
    result = finetune_lora._timestamped_dir(base)
    assert result.parent == tmp_path / "adapters"
    assert len(result.name) == 16
    assert result.name[4] == "-"
    assert result.name[7] == "-"
    assert result.name[10] == "_"


def test_train_lora_missing_dataset(tmp_path: Path) -> None:
    """train_lora raises TrainingError for a missing dataset."""
    from features.finetune.trainer import train_lora

    options = TrainingOptions(base_model="m", rank=8, alpha=16)
    with pytest.raises(TrainingError, match="датасет не найден"):
        train_lora(
            options=options,
            dataset_path=tmp_path / "missing.jsonl",
            output_dir=tmp_path / "out",
        )


def test_train_lora_empty_dataset(tmp_path: Path) -> None:
    """train_lora raises TrainingError for an empty dataset."""
    from features.finetune.trainer import train_lora

    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("", encoding="utf-8")
    options = TrainingOptions(base_model="m", rank=8, alpha=16)
    with pytest.raises(TrainingError, match="датасет пуст"):
        train_lora(
            options=options,
            dataset_path=dataset,
            output_dir=tmp_path / "out",
        )


def test_train_lora_missing_dependency(tmp_path: Path) -> None:
    """train_lora raises a helpful TrainingError when datasets is missing."""
    from features.finetune.trainer import train_lora

    dataset = tmp_path / "one.jsonl"
    dataset.write_text(
        '{"messages": [{"role": "user", "content": "x"}]}\n',
        encoding="utf-8",
    )
    options = TrainingOptions(base_model="m")
    import sys

    original = sys.modules.get("datasets")
    sys.modules["datasets"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(TrainingError, match="datasets"):
            train_lora(
                options=options,
                dataset_path=dataset,
                output_dir=tmp_path / "out",
            )
    finally:
        if original is not None:
            sys.modules["datasets"] = original
        else:
            del sys.modules["datasets"]
