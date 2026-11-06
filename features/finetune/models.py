"""Immutable models for the fine-tune dataset and training options."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Bump when the JSONL line format changes in a breaking way.
FORMAT_VERSION: str = "1.0"


class Message(BaseModel):
    """One chat message in the OpenAI messages (ChatML) format."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class DatasetRecord(BaseModel):
    """One JSONL line of the SFT dataset: a full chat example + metadata.

    The ``messages`` list is ``[system, user, assistant]``: the system prompt
    comes from ``system.jinja``, the user message is the reconstructed game
    context at the prediction moment, and the assistant message is the correct
    JSON response (the model's own answer when it was right, or the ground
    truth role when it was wrong but final roles are known).

    Metadata fields are ignored by TRL/Unsloth (they read ``messages``) but are
    useful for debugging and for Axolotl/LLaMA-Factory-style tooling.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: str = FORMAT_VERSION
    messages: list[Message] = Field(min_length=3)
    game_id: int
    prediction_id: int
    player_id: int
    ts: datetime
    source: Literal["current", "history"] = "current"
    correct: bool
    ground_truth_role: str
    predicted_role: str | None = None
    confidence: float | None = None


class PredictionRow(BaseModel):
    """A prediction row from the ``predictions`` or ``predictions_history`` table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    game_id: int
    ts: datetime
    player_id: int
    predicted_role: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str | None = None
    reason: str = ""
    pattern_ids: list[str] = Field(default_factory=list)
    pattern_descriptions: list[str] = Field(default_factory=list)
    correct: bool | None = None

    @classmethod
    def from_sqlite_row(cls, row: sqlite3.Row) -> PredictionRow:
        """Parse a prediction table row, tolerating naive SQLite timestamps."""
        ts_raw = row["ts"]
        if isinstance(ts_raw, str):
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        elif isinstance(ts_raw, datetime):
            ts = ts_raw if ts_raw.tzinfo is not None else ts_raw.replace(tzinfo=UTC)
        else:  # pragma: no cover — defensive: unexpected ts type
            raise TypeError(f"unexpected prediction ts type: {type(ts_raw).__name__}")
        pattern_ids = _parse_json_list(row["pattern_ids"])
        pattern_descriptions = _parse_json_list(row["pattern_descriptions"])
        correct_raw = row["correct"]
        return cls(
            id=int(row["id"]),
            game_id=int(row["game_id"]),
            ts=ts,
            player_id=int(row["player_id"]),
            predicted_role=row["predicted_role"],
            confidence=float(row["confidence"]),
            reasoning=row["reasoning"],
            reason=row["reason"] or "",
            pattern_ids=pattern_ids,
            pattern_descriptions=pattern_descriptions,
            correct=None if correct_raw is None else bool(correct_raw),
        )


class TrainingOptions(BaseModel):
    """Resolved hyperparameters for one LoRA/QLoRA training run.

    Built from :class:`app.config.FinetuneSection` plus CLI overrides by the
    ``scripts/finetune_lora.py`` entry point. Kept dependency-free so the
    trainer module never imports ``app/``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_model: str = Field(min_length=1)
    method: Literal["lora", "qlora"] = "qlora"
    backend: Literal["unsloth", "peft"] = "unsloth"
    rank: int = Field(ge=1, le=256, default=16)
    alpha: int = Field(ge=1, le=512, default=32)
    dropout: float = Field(ge=0.0, le=1.0, default=0.05)
    learning_rate: float = Field(gt=0.0, default=2e-4)
    epochs: int = Field(ge=1, le=100, default=3)
    batch_size: int = Field(ge=1, le=128, default=2)
    gradient_accumulation_steps: int = Field(ge=1, le=1024, default=8)
    max_seq_length: int = Field(ge=256, le=32768, default=4096)
    warmup_ratio: float = Field(ge=0.0, le=1.0, default=0.03)
    weight_decay: float = Field(ge=0.0, le=1.0, default=0.0)


def _parse_json_list(raw: Any) -> list[str]:
    """Decode a JSON-text column into a list of strings, defaulting to empty."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]
