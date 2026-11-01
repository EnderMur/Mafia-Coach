"""Tests for YAML loading and validation of application settings."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.config import AppConfig, load_config
from domain.exceptions import ConfigError
from tests.unit.app.conftest import create_settings_file


def test_load_valid_config(tmp_path: Path) -> None:
    """A valid copy of the example settings becomes an AppConfig."""
    config = load_config(create_settings_file(tmp_path, chat_id=-100123))

    assert isinstance(config, AppConfig)
    assert config.telegram.chat_id == -100123
    assert config.telegram.bot_id == 42
    assert config.llm.trigger_mode == "queue"
    assert config.db.path == str(tmp_path / "mafia.db")


def test_missing_file_raises_config_error_with_path(tmp_path: Path) -> None:
    """A missing settings file is reported with its full path."""
    missing_path = tmp_path / "missing-settings.yaml"

    with pytest.raises(ConfigError, match=str(missing_path)):
        load_config(missing_path)


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    """Malformed YAML is translated to the domain configuration error."""
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("telegram: [", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(config_path)


def test_missing_required_field_raises_config_error(tmp_path: Path) -> None:
    """Required nested settings are validated by Pydantic."""
    config_path = create_settings_file(tmp_path)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del settings["telegram"]["api_id"]
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ConfigError, match="api_id"):
        load_config(config_path)


def test_bot_identifier_is_required(tmp_path: Path) -> None:
    """At least one of bot_username and bot_id must be configured."""
    config_path = create_settings_file(tmp_path)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["telegram"]["bot_username"] = None
    settings["telegram"]["bot_id"] = None
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ConfigError, match="bot_username or telegram.bot_id required"):
        load_config(config_path)


def test_top_level_extra_field_is_forbidden(tmp_path: Path) -> None:
    """Unexpected root keys fail validation instead of being silently ignored."""
    config_path = create_settings_file(tmp_path)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["unexpected"] = True
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ConfigError, match="unexpected"):
        load_config(config_path)


def test_nested_invalid_type_raises_config_error(tmp_path: Path) -> None:
    """A non-integer configured chat id is rejected."""
    config_path = create_settings_file(tmp_path)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["telegram"]["chat_id"] = "not-int"
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ConfigError, match="chat_id"):
        load_config(config_path)
