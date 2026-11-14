"""Общие проверки и форматы оффлайн-пайплайна."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from infra.llm.schemas import AnalysisResponse


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Конфиг должен быть словарём")
    return config


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or [m.get("role") for m in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("Ожидаются сообщения system, user, assistant")
    for message in messages:
        if set(message) != {"role", "content"} or not isinstance(message["content"], str):
            raise ValueError("Недопустимые поля сообщения")
        if any(token in message["content"] for token in ("<|im_start|>", "<|im_end|>")):
            raise ValueError("Специальные токены в содержимом сообщения")
    response = AnalysisResponse.model_validate_json(messages[-1]["content"])
    for prediction in response.analysis:
        if prediction.suspected_role is None and (
            prediction.confidence != 0.0 or prediction.reason != "нет данных"
        ):
            raise ValueError("Для null-роли требуются confidence=0 и reason=нет данных")
