"""Tests for the finetune configuration section."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.config import AppConfig, FinetuneSection, load_config
from domain.exceptions import ConfigError
from tests.unit.app.conftest import create_settings_file


def test_finetune_section_defaults() -> None:
    """A minimal FinetuneSection gets sensible defaults."""
    section = FinetuneSection(base_model="qwen/qwen3.5-9b")
    assert section.enabled is False
    assert section.method == "qlora"
    assert section.backend == "unsloth"
    assert section.rank == 16
    assert section.alpha == 32
    assert section.dropout == 0.05
    assert section.epochs == 3
    assert section.batch_size == 2
    assert section.gradient_accumulation_steps == 8
    assert section.max_seq_length == 4096
    assert section.output_dir == "data/adapters"
    assert section.dataset_path == "data/datasets/sft_train.jsonl"
    assert section.max_examples is None


def test_app_config_has_finetune_section(tmp_path: Path) -> None:
    """The example settings load with a valid finetune section."""
    config_path = create_settings_file(tmp_path)
    config = load_config(config_path)
    assert isinstance(config, AppConfig)
    assert isinstance(config.finetune, FinetuneSection)
    assert config.finetune.base_model == "qwen/qwen3.5-9b"


def test_finetune_section_custom_values(tmp_path: Path) -> None:
    """Custom finetune values are preserved."""
    config_path = create_settings_file(tmp_path)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["finetune"].update(
        {
            "base_model": "custom/model",
            "method": "lora",
            "backend": "peft",
            "rank": 32,
            "alpha": 64,
            "epochs": 5,
        }
    )
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

    config = load_config(config_path)
    assert config.finetune.base_model == "custom/model"
    assert config.finetune.method == "lora"
    assert config.finetune.backend == "peft"
    assert config.finetune.rank == 32
    assert config.finetune.alpha == 64
    assert config.finetune.epochs == 5


def test_finetune_section_invalid_method_rejected(tmp_path: Path) -> None:
    """An unknown method fails validation."""
    config_path = create_settings_file(tmp_path)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["finetune"]["method"] = "invalid"
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_finetune_section_invalid_backend_rejected(tmp_path: Path) -> None:
    """An unknown backend fails validation."""
    config_path = create_settings_file(tmp_path)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["finetune"]["backend"] = "unknown"
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_finetune_section_invalid_rank_rejected(tmp_path: Path) -> None:
    """A non-positive rank fails validation."""
    config_path = create_settings_file(tmp_path)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["finetune"]["rank"] = 0
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_finetune_section_invalid_learning_rate_rejected(tmp_path: Path) -> None:
    """A non-positive learning rate fails validation."""
    config_path = create_settings_file(tmp_path)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["finetune"]["learning_rate"] = 0.0
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_old_config_without_finetune_section(tmp_path: Path) -> None:
    """A settings file without a finetune section still loads."""
    config_path = create_settings_file(tmp_path)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del settings["finetune"]
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

    config = load_config(config_path)
    assert config.finetune.base_model == "qwen/qwen3.5-9b"
    assert config.finetune.method == "qlora"
