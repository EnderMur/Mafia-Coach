"""Tests for LLM response schemas and context validation."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from infra.llm.schemas import (
    AlternativePrediction,
    AnalysisResponse,
    PlayerPrediction,
    _has_strong_signal,
    validate_response,
)


def valid_prediction(**overrides: Any) -> dict[str, Any]:
    """Return a valid raw prediction mapping."""
    prediction: dict[str, Any] = {
        "player_id": 100,
        "player_name": "Player A",
        "suspected_role": "Role A",
        "confidence": 0.6,
        "reason": "Краткое объяснение роли.",
        "alternative": {"role": "Role B", "confidence": 0.3},
        "reasons": ["reason one"],
        "pattern_ids": [],
        "pattern_descriptions": [],
    }
    prediction.update(overrides)
    return prediction


def response_json(analysis: list[dict[str, Any]]) -> str:
    """Serialize predictions within a valid response envelope."""
    return json.dumps(
        {"analysis": analysis, "game_read": "A concise read.", "reasoning": "Reasoned."}
    )


def test_valid_analysis_response_parses_all_fields() -> None:
    """A valid response parses and dumps all declared fields."""
    response = AnalysisResponse.model_validate_json(response_json([valid_prediction()]))
    dumped = response.model_dump()
    assert set(dumped) == {"analysis", "game_read", "reasoning"}
    assert set(dumped["analysis"][0]) == {
        "player_id",
        "player_name",
        "suspected_role",
        "confidence",
        "reason",
        "alternative",
        "reasons",
        "pattern_ids",
        "pattern_descriptions",
    }


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_player_confidence_out_of_bounds_is_rejected(confidence: float) -> None:
    """Direct PlayerPrediction construction enforces confidence bounds."""
    with pytest.raises(ValidationError):
        PlayerPrediction.model_validate(valid_prediction(confidence=confidence))


def test_player_confidence_upper_boundary_is_valid() -> None:
    """Confidence equal to one is a valid boundary value."""
    prediction = PlayerPrediction.model_validate(valid_prediction(confidence=1.0, alternative=None))
    assert prediction.confidence == 1.0


def test_alternative_confidence_out_of_bounds_is_rejected() -> None:
    """Alternative confidence uses the same inclusive bounds."""
    with pytest.raises(ValidationError):
        AlternativePrediction(role="Role B", confidence=1.01)


def test_confidence_sum_over_one_is_rejected_by_model() -> None:
    """Direct model validation rejects an overfull confidence pair."""
    with pytest.raises(ValidationError, match=r"confidence \+ alternative"):
        PlayerPrediction.model_validate(
            valid_prediction(
                confidence=0.7,
                alternative={"role": "Role B", "confidence": 0.4},
            )
        )


def test_null_role_requires_zero_confidence() -> None:
    """suspected_role = null is only valid with confidence 0.0."""
    with pytest.raises(ValidationError, match=r"confidence must be 0.0"):
        PlayerPrediction.model_validate(valid_prediction(suspected_role=None, confidence=0.5))


def test_null_role_requires_no_data_reason() -> None:
    """suspected_role = null is only valid with reason "нет данных"."""
    with pytest.raises(ValidationError, match=r'"нет данных"'):
        PlayerPrediction.model_validate(
            valid_prediction(
                suspected_role=None,
                confidence=0.0,
                reason="неизвестно",
            )
        )


def test_null_role_with_zero_confidence_and_no_data_is_valid() -> None:
    """suspected_role = null with confidence 0.0 and reason "нет данных" passes."""
    prediction = PlayerPrediction.model_validate(
        valid_prediction(suspected_role=None, confidence=0.0, reason="нет данных")
    )
    assert prediction.suspected_role is None
    assert prediction.confidence == 0.0
    assert prediction.reason == "нет данных"


def test_pattern_lists_must_have_equal_length() -> None:
    """pattern_ids and pattern_descriptions must stay in sync."""
    with pytest.raises(ValidationError, match=r"same length"):
        PlayerPrediction.model_validate(
            valid_prediction(pattern_ids=["#2"], pattern_descriptions=[])
        )


def test_null_role_with_messages_is_normalized_not_dropped() -> None:
    """FIX-LOG-ANALYSIS C: null + messages/votes → composition slot, not a drop."""
    raw = response_json(
        [valid_prediction(suspected_role=None, confidence=0.0, reason="нет данных")]
    )
    result, warnings = validate_response(
        raw,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert len(result.analysis) == 1
    prediction = result.analysis[0]
    assert prediction.suspected_role in {"Role A", "Role B"}
    assert 0.3 <= prediction.confidence <= 0.5
    assert prediction.reason == "распределение состава бота"
    assert not any("prediction dropped" in warning for warning in warnings)
    assert any("100" in warning and "normalized" in warning for warning in warnings)


def test_null_role_kept_for_silent_player() -> None:
    """A silent player (no messages, no votes) may receive a null role."""
    raw = response_json(
        [valid_prediction(suspected_role=None, confidence=0.0, reason="нет данных")]
    )
    result, warnings = validate_response(
        raw,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
        silent_player_ids=frozenset({100}),
    )
    assert len(result.analysis) == 1
    assert result.analysis[0].suspected_role is None
    assert result.analysis[0].confidence == 0.0
    assert result.analysis[0].reason == "нет данных"
    assert warnings == []


def test_null_role_not_dropped_as_placeholder_reasons() -> None:
    """A null role carries "нет данных" in ``reason`` and survives the placeholder drop."""
    raw = response_json(
        [
            valid_prediction(
                suspected_role=None,
                confidence=0.0,
                reason="нет данных",
                reasons=["нет данных"],
            )
        ]
    )
    result, warnings = validate_response(
        raw,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
        silent_player_ids=frozenset({100}),
    )
    assert len(result.analysis) == 1
    assert warnings == []


def test_empty_reasons_are_rejected() -> None:
    """At least one explanation reason is required."""
    with pytest.raises(ValidationError):
        PlayerPrediction.model_validate(valid_prediction(reasons=[]))


def test_six_reasons_are_rejected() -> None:
    """No more than five explanation reasons are accepted."""
    with pytest.raises(ValidationError):
        PlayerPrediction.model_validate(valid_prediction(reasons=["r"] * 6))


def test_empty_game_read_is_rejected() -> None:
    """The game summary must be non-empty."""
    raw = json.loads(response_json([]))
    raw["game_read"] = ""
    with pytest.raises(ValidationError):
        AnalysisResponse.model_validate(raw)


def test_player_prediction_is_frozen() -> None:
    """Prediction fields cannot be reassigned after validation."""
    prediction = PlayerPrediction.model_validate(valid_prediction())
    with pytest.raises(ValidationError):
        prediction.player_name = "Changed"


def test_extra_fields_are_forbidden() -> None:
    """Response objects reject fields outside the public schema."""
    raw = json.loads(response_json([]))
    raw["unexpected"] = True
    with pytest.raises(ValidationError):
        AnalysisResponse.model_validate(raw)


def test_invalid_json_raises_pydantic_validation_error() -> None:
    """Invalid JSON is surfaced through Pydantic's validation error type."""
    with pytest.raises(ValidationError):
        validate_response(
            "not valid JSON",
            known_player_ids=frozenset(),
            role_names_ru=frozenset(),
        )


def test_context_validation_keeps_known_prediction_without_warnings() -> None:
    """Known player ids and role names pass contextual validation."""
    raw = response_json([valid_prediction()])
    result, warnings = validate_response(
        raw,
        known_player_ids=frozenset({100, 101}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis[0].player_id == 100
    assert result.analysis[0].suspected_role == "Role A"
    assert warnings == []


def test_unknown_player_prediction_is_dropped_with_warning() -> None:
    """Predictions for players outside the active game are removed."""
    raw = response_json([valid_prediction(player_id=999)])
    result, warnings = validate_response(
        raw,
        known_player_ids=frozenset({100, 101}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis == []
    assert any("player 999 not in current game" in warning for warning in warnings)


def test_role_alias_is_replaced_by_canonical_name() -> None:
    """A role alias is replaced by the resolver's canonical name."""
    result, warnings = validate_response(
        response_json([valid_prediction(suspected_role="alias-a")]),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
        role_alias_resolver=lambda value: {"alias-a": "Role A"}.get(value),
    )
    assert result.analysis[0].suspected_role == "Role A"
    assert warnings == []


def test_alternative_role_alias_is_replaced_by_canonical_name() -> None:
    """An alternative role alias resolves without dropping its prediction."""
    result, warnings = validate_response(
        response_json([valid_prediction(alternative={"role": "alias-b", "confidence": 0.3})]),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
        role_alias_resolver=lambda value: {"alias-b": "Role B"}.get(value),
    )
    assert result.analysis[0].alternative is not None
    assert result.analysis[0].alternative.role == "Role B"
    assert warnings == []


def test_unresolved_primary_role_drops_prediction() -> None:
    """An unknown primary role drops that prediction and reports a warning."""
    result, warnings = validate_response(
        response_json([valid_prediction(suspected_role="Unknown", alternative=None)]),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis == []
    assert any("prediction dropped" in warning for warning in warnings)


def test_unknown_alternative_is_dropped_without_dropping_prediction() -> None:
    """An unknown alternative is removed while retaining the primary result."""
    result, warnings = validate_response(
        response_json([valid_prediction(alternative={"role": "Unknown", "confidence": 0.3})]),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert len(result.analysis) == 1
    assert result.analysis[0].alternative is None
    assert any("alternative dropped" in warning for warning in warnings)


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps([]),
        json.dumps({"analysis": {}, "game_read": "ok", "reasoning": "ok"}),
        json.dumps({"analysis": [None], "game_read": "ok", "reasoning": "ok"}),
    ],
)
def test_invalid_response_shapes_raise_validation_error(raw: str) -> None:
    """Malformed response and prediction structures are schema errors."""
    with pytest.raises(ValidationError):
        validate_response(
            raw,
            known_player_ids=frozenset({100}),
            role_names_ru=frozenset({"Role A", "Role B"}),
        )


def test_missing_confidence_raises_validation_error() -> None:
    """Confidence remains a required field after soft preprocessing."""
    prediction = valid_prediction()
    prediction.pop("confidence")
    with pytest.raises(ValidationError):
        validate_response(
            response_json([prediction]),
            known_player_ids=frozenset({100}),
            role_names_ru=frozenset({"Role A", "Role B"}),
        )


def test_non_numeric_confidence_raises_validation_error() -> None:
    """Non-numeric confidence is left for Pydantic schema validation."""
    with pytest.raises(ValidationError):
        validate_response(
            response_json([valid_prediction(confidence="high")]),
            known_player_ids=frozenset({100}),
            role_names_ru=frozenset({"Role A", "Role B"}),
        )


def test_nan_confidence_raises_validation_error() -> None:
    """NaN cannot be clamped and remains invalid under the schema."""
    raw = json.dumps(
        {
            "analysis": [valid_prediction(confidence=float("nan"), alternative=None)],
            "game_read": "ok",
            "reasoning": "ok",
        }
    )
    with pytest.raises(ValidationError):
        validate_response(
            raw,
            known_player_ids=frozenset({100}),
            role_names_ru=frozenset({"Role A", "Role B"}),
        )


def test_context_validation_clamps_alternative_confidence() -> None:
    """Alternative confidence outside the interval is clamped with a warning."""
    result, warnings = validate_response(
        response_json(
            [
                valid_prediction(
                    confidence=0.0,
                    alternative={"role": "Role B", "confidence": 1.5},
                )
            ]
        ),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis[0].alternative is not None
    assert result.analysis[0].alternative.confidence == 1.0
    assert any("alternative confidence" in warning for warning in warnings)


def test_context_validation_clamps_out_of_range_confidence() -> None:
    """Context parsing clamps numeric confidence and reports the adjustment."""
    result, warnings = validate_response(
        response_json(
            [
                valid_prediction(
                    confidence=1.5,
                    alternative=None,
                    reasons=["голосовал против раскрытого дона"],
                )
            ]
        ),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis[0].confidence == 1.0
    assert any("clamped to [0, 1]" in warning for warning in warnings)


def test_context_validation_normalizes_overfull_pair() -> None:
    """Overfull primary and alternative confidence values are normalized."""
    result, warnings = validate_response(
        response_json(
            [
                valid_prediction(
                    confidence=0.8,
                    alternative={"role": "Role B", "confidence": 0.6},
                )
            ]
        ),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    prediction = result.analysis[0]
    assert prediction.alternative is not None
    assert prediction.confidence + prediction.alternative.confidence == pytest.approx(1.0)
    assert any("normalized" in warning for warning in warnings)


@pytest.mark.parametrize(
    "reasons",
    [
        ["нет данных"],
        ["  НЕТ ДАННЫХ  ", " N/A ", "None"],
        ["неизвестно", "не знаю", "непонятно"],
        ["нет аргументов", "недостаточно данных"],
        ["   "],
    ],
)
def test_drop_placeholder_reasons(reasons: list[str]) -> None:
    """Predictions with no genuine reason are removed and reported."""
    result, warnings = validate_response(
        response_json([valid_prediction(reasons=reasons)]),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis == []
    assert warnings == ["dropped prediction for player 100: placeholder reasons"]


@pytest.mark.parametrize(
    "reasons", [["голосовал против комиссара"], ["нет данных", "голосовал против комиссара"]]
)
def test_keep_real_reasons(reasons: list[str]) -> None:
    """A substantive reason preserves the prediction, including a mixed list."""
    result, warnings = validate_response(
        response_json([valid_prediction(reasons=reasons)]),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert len(result.analysis) == 1
    assert result.analysis[0].reasons == reasons
    assert warnings == []


def test_clamp_confidence_without_strong_signal() -> None:
    """B3. A confident role backed only by "молчит" is clamped to 0.5."""
    result, warnings = validate_response(
        response_json([valid_prediction(confidence=0.95, alternative=None, reasons=["молчит"])]),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis[0].confidence == 0.5
    assert any(
        "clamped confidence for player 100" in warning and "no strong signal" in warning
        for warning in warnings
    )


def test_keep_strong_signal_confidence() -> None:
    """B3. A concrete argument keeps the model's confidence untouched."""
    result, warnings = validate_response(
        response_json(
            [
                valid_prediction(
                    confidence=0.9,
                    alternative=None,
                    reasons=["голосовал против раскрытого дона трижды"],
                )
            ]
        ),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis[0].confidence == 0.9
    assert warnings == []


def test_has_strong_signal_classifies_reasons() -> None:
    """B3. Only a detailed reason counts as evidence."""
    assert not _has_strong_signal(["молчит"])
    assert not _has_strong_signal(["молчит", "пассивен", "нет данных"])
    assert _has_strong_signal(["голосовал против раскрытого дона"])


@pytest.mark.parametrize("reasoning", ["", None])
def test_response_allows_empty_or_null_reasoning(reasoning: str | None) -> None:
    response = AnalysisResponse.model_validate(
        {"analysis": [], "game_read": "Обзор", "reasoning": reasoning}
    )
    assert response.reasoning == reasoning


def test_response_allows_missing_reasoning() -> None:
    response = AnalysisResponse.model_validate({"analysis": [], "game_read": "Обзор"})
    assert response.reasoning is None


def test_llm_fixture_validates_and_contains_reasoning() -> None:
    from pathlib import Path

    fixtures = list(Path("tests/fixtures/llm").glob("*.json"))
    assert fixtures
    for fixture in fixtures:
        response = AnalysisResponse.model_validate_json(fixture.read_text(encoding="utf-8"))
        assert response.reasoning


def test_invalid_primary_uses_canonical_alternative_and_logs_info() -> None:
    from loguru import logger

    logs: list[str] = []
    sink = logger.add(lambda message: logs.append(str(message)), level="INFO")
    try:
        result, warnings = validate_response(
            response_json(
                [
                    valid_prediction(
                        suspected_role="not a role",
                        confidence=0.4,
                        alternative={"role": "alias-b", "confidence": 0.35},
                    )
                ]
            ),
            known_player_ids=frozenset({100}),
            role_names_ru=frozenset({"Role A", "Role B"}),
            role_alias_resolver=lambda value: {"alias-b": "Role B"}.get(value),
        )
    finally:
        logger.remove(sink)
    assert len(result.analysis) == 1
    assert result.analysis[0].suspected_role == "Role B"
    assert result.analysis[0].confidence == pytest.approx(0.35)
    assert result.analysis[0].alternative is None
    assert not warnings
    assert any("100" in message and "Role B" in message for message in logs)


@pytest.mark.parametrize("alternative", [None, {"role": "invalid", "confidence": 0.2}])
def test_invalid_primary_and_unusable_alternative_are_dropped(
    alternative: dict[str, Any] | None,
) -> None:
    result, warnings = validate_response(
        response_json([valid_prediction(suspected_role="invalid", alternative=alternative)]),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis == []
    assert any("prediction dropped" in warning for warning in warnings)


def test_unknown_placeholder_primary_uses_valid_alternative() -> None:
    result, warnings = validate_response(
        response_json(
            [
                valid_prediction(
                    suspected_role="Неизвестно", alternative={"role": "Role B", "confidence": 0.3}
                )
            ]
        ),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis[0].suspected_role == "Role B"
    assert result.analysis[0].alternative is None
    assert warnings == []


def test_unknown_placeholder_without_valid_alternative_remains_silent() -> None:
    result, warnings = validate_response(
        response_json([valid_prediction(suspected_role="Неизвестно", alternative=None)]),
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis == [] and warnings == []


def test_pattern_ids_one_vs_descriptions_two_is_normalized() -> None:
    """FIX-LOG-ANALYSIS B: 1 vs 2 must not fail AnalysisResponse."""
    raw = response_json(
        [valid_prediction(pattern_ids=["#2"], pattern_descriptions=["desc a", "desc b"])]
    )
    result, warnings = validate_response(
        raw,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert len(result.analysis) == 1
    prediction = result.analysis[0]
    assert prediction.pattern_ids == ["#2"]
    assert prediction.pattern_descriptions == ["desc a"]
    assert len(prediction.pattern_ids) == len(prediction.pattern_descriptions)
    assert any("100" in warning and "1" in warning and "2" in warning for warning in warnings)


@pytest.mark.parametrize(
    ("pattern_ids", "pattern_descriptions"),
    [(["#2", "#7"], []), ([], ["desc a", "desc b"]), (None, ["desc a"])],
)
def test_empty_or_invalid_pattern_list_forces_both_empty(
    pattern_ids: object, pattern_descriptions: object
) -> None:
    """If either pattern list is empty (or not a list), both become empty."""
    raw = response_json(
        [valid_prediction(pattern_ids=pattern_ids, pattern_descriptions=pattern_descriptions)]
    )
    result, _warnings = validate_response(
        raw,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
    )
    assert result.analysis[0].pattern_ids == []
    assert result.analysis[0].pattern_descriptions == []


def test_model_validator_still_rejects_mismatched_pattern_lists() -> None:
    """The equality invariant stays in the model; only raw data is normalized."""
    raw = json.loads(
        response_json([valid_prediction(pattern_ids=["#2"], pattern_descriptions=["a", "b"])])
    )
    with pytest.raises(ValidationError, match="same length"):
        AnalysisResponse.model_validate(raw)


def test_pattern_alignment_also_applies_to_null_role() -> None:
    """PR #54 null handling still clears pattern lists after alignment."""
    raw = response_json(
        [
            valid_prediction(
                suspected_role=None,
                confidence=0.0,
                reason="нет данных",
                pattern_ids=["#2"],
                pattern_descriptions=["a", "b"],
            )
        ]
    )
    result, _warnings = validate_response(
        raw,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
        silent_player_ids=frozenset({100}),
    )
    assert len(result.analysis) == 1
    assert result.analysis[0].suspected_role is None
    assert result.analysis[0].pattern_ids == []
    assert result.analysis[0].pattern_descriptions == []


def test_null_role_without_messages_or_votes_is_not_dropped() -> None:
    """FIX-LOG-ANALYSIS C: null + no data → role=null, 0.0, 'нет данных'."""
    raw = response_json([valid_prediction(suspected_role=None, confidence=0.42, reason="не знаю")])
    result, warnings = validate_response(
        raw,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({"Role A", "Role B"}),
        silent_player_ids=frozenset({100}),
    )
    assert len(result.analysis) == 1
    prediction = result.analysis[0]
    assert prediction.suspected_role is None
    assert prediction.confidence == 0.0
    assert prediction.reason == "нет данных"
    assert prediction.pattern_ids == []
    assert prediction.pattern_descriptions == []
    assert warnings == []
