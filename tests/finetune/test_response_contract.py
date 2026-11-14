"""Контракт целевых SFT-ответов и его отрицательные случаи."""

import json

import pytest
from pydantic import ValidationError

from infra.llm.schemas import AnalysisResponse
from scripts.finetune.common import validate_messages


def record(role="Доктор", confidence=0.5, reason="Реплика игрока."):
    response = {
        "analysis": [
            {
                "player_id": 1,
                "player_name": "Аня",
                "suspected_role": role,
                "confidence": confidence,
                "reason": reason,
                "reasons": [reason],
                "pattern_ids": [],
                "pattern_descriptions": [],
            }
        ],
        "game_read": "Игра идёт.",
    }
    return [
        {"role": "system", "content": "системное сообщение"},
        {"role": "user", "content": "данные"},
        {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
    ]


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_schema_valid_and_immutable(confidence):
    messages = record(confidence=confidence)
    validate_messages(messages)
    response = AnalysisResponse.model_validate_json(messages[-1]["content"])
    with pytest.raises(ValidationError):
        response.game_read = "другая"
    with pytest.raises(ValidationError):
        response.analysis[0].confidence = 0.3


def test_null_role_with_no_data_valid():
    validate_messages(record(role=None, confidence=0.0, reason="нет данных"))


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_out_of_range_confidence_rejected(confidence):
    with pytest.raises(ValidationError):
        validate_messages(record(confidence=confidence))


@pytest.mark.parametrize("scope", ["response", "prediction"])
def test_extra_field_rejected(scope):
    messages = record()
    data = json.loads(messages[-1]["content"])
    (data if scope == "response" else data["analysis"][0])["hidden"] = "секрет"
    messages[-1]["content"] = json.dumps(data, ensure_ascii=False)
    with pytest.raises(ValidationError):
        validate_messages(messages)


@pytest.mark.parametrize("confidence,reason", [(0.5, "нет данных"), (0.0, "неизвестно")])
def test_null_role_invariants_rejected(confidence, reason):
    with pytest.raises((ValidationError, ValueError)):
        validate_messages(record(role=None, confidence=confidence, reason=reason))


def test_null_with_pattern_exception_rejected_by_dataset_policy():
    # Существующая схема допускает исключение для непустых pattern_ids;
    # строгий контракт датасета не допускает его, не меняя модель проекта.
    messages = record(role=None, confidence=0.0, reason="другая причина")
    data = json.loads(messages[-1]["content"])
    data["analysis"][0]["pattern_ids"] = ["#1"]
    data["analysis"][0]["pattern_descriptions"] = ["паттерн"]
    messages[-1]["content"] = json.dumps(data, ensure_ascii=False)
    AnalysisResponse.model_validate(data)
    with pytest.raises(ValueError, match="Для null-роли"):
        validate_messages(messages)
