"""Validated application settings loaded from a YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from domain.exceptions import ConfigError


class TelegramSection(BaseModel):
    """Telegram credentials and target-chat configuration."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    session_file: str = Field(min_length=1)
    api_id: int = Field(gt=0)
    api_hash: str = Field(min_length=16)
    chat_id: int | None = None
    bot_username: str | None = None
    bot_id: int | None = Field(default=None, gt=0)
    read_only: bool = True
    ignore_own_messages: bool = True

    @model_validator(mode="after")
    def validate_bot_identifier(self) -> Self:
        """Require at least one configured identifier for the Mafia game bot."""
        has_username = self.bot_username is not None and bool(self.bot_username.strip())
        if not has_username and self.bot_id is None:
            raise ValueError("telegram.bot_username or telegram.bot_id required")
        if not self.session_file.strip():
            raise ValueError("telegram.session_file must not be blank")
        if not self.api_hash.strip():
            raise ValueError("telegram.api_hash must not be blank")
        return self


class LLMSection(BaseModel):
    """Connection, generation, and message-trigger settings for the LLM."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0, default=0.3)
    top_p: float = Field(ge=0.0, le=1.0, default=0.9)
    max_tokens: int = Field(ge=1, default=2048)
    request_timeout_sec: float = Field(gt=0, default=120.0)
    connect_timeout_sec: float = Field(gt=0, default=5.0)
    max_retries: int = Field(ge=1, le=3, default=3)
    retry_backoff_sec: float = Field(gt=0, default=2.0)
    stream_reasoning: bool = True
    context_window_messages: int = Field(ge=1, default=200)
    summarize_after_messages: int = Field(ge=1, default=300)
    trigger_mode: Literal["every_message", "queue", "batch_debounce"] = "queue"
    batch_debounce_sec: float = Field(ge=0.0, default=3.0)
    batch_max_messages: int = Field(ge=1, default=10)
    reasoning_max_chars: int = Field(ge=100, default=4096)


class DBSection(BaseModel):
    """SQLite location and migration startup policy."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    path: str = Field(min_length=1)
    apply_migrations_on_start: bool = True


class AnalysisSection(BaseModel):
    """Presentation thresholds for analysis results."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    min_confidence_to_display: float = Field(ge=0.0, le=1.0, default=0.15)


class UISection(BaseModel):
    """Optional output-window settings."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    stats_window: bool = True
    reasoning_window: bool = True
    language: str = "ru"


class LoggingSection(BaseModel):
    """Loguru sink settings."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    level: str = "INFO"
    dir: str = "logs"
    rotation: str = "10 MB"
    retention_days: int = Field(ge=1, default=30)


class AppConfig(BaseModel):
    """Immutable root configuration loaded from ``configs/settings.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    telegram: TelegramSection
    llm: LLMSection
    db: DBSection
    roles_file: str = Field(min_length=1)
    analysis: AnalysisSection = Field(default_factory=AnalysisSection)
    ui: UISection = Field(default_factory=UISection)
    logging: LoggingSection = Field(default_factory=LoggingSection)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an :class:`AppConfig` from a UTF-8 YAML file.

    Args:
        path: Settings file location.

    Returns:
        A validated, immutable application configuration.

    Raises:
        ConfigError: If the file cannot be read, its YAML is invalid, or the
            parsed settings do not satisfy the configuration schema.
    """
    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigError(f"config file not found: {config_path}") from error
    except (OSError, UnicodeError) as error:
        raise ConfigError(f"cannot read config {config_path}: {error}") from error

    try:
        raw_config = yaml.safe_load(raw_text)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error

    try:
        return AppConfig.model_validate(raw_config)
    except ValidationError as error:
        raise ConfigError(f"invalid config {config_path}: {error}") from error
