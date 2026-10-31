"""Pydantic schemas and contextual validation for LLM responses."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Callable, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class AlternativePrediction(BaseModel):
    """An alternative role hypothesis with a confidence score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class PlayerPrediction(BaseModel):
    """Role hypotheses and reasons for one player."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: int
    player_name: str = Field(min_length=1)
    suspected_role: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    alternative: AlternativePrediction | None = None
    reasons: list[str] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def _conf_sum_le_one(self) -> PlayerPrediction:
        """Ensure the primary and alternative confidence total is at most one."""
        if self.alternative is not None:
            total = self.confidence + self.alternative.confidence
            if total > 1.0 + 1e-9:
                raise ValueError(f"confidence + alternative.confidence > 1.0: {total}")
        return self


class AnalysisResponse(BaseModel):
    """The structured final response returned by the language model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis: list[PlayerPrediction]
    game_read: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)


def validate_response(
    raw: str,
    *,
    known_player_ids: frozenset[int],
    role_names_ru: frozenset[str],
    role_alias_resolver: Callable[[str], str | None] | None = None,
) -> tuple[AnalysisResponse, list[str]]:
    """Parse an LLM response and validate its predictions against game context.

    Schema violations raise Pydantic's ValidationError. Contextual mismatches are
    softened into warnings and dropped predictions or alternatives.
    """
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        try:
            AnalysisResponse.model_validate_json(raw)
        except ValidationError as validation_error:
            raise validation_error from error
        raise error
    warnings: list[str] = []
    normalized_data = deepcopy(parsed)
    _normalize_confidence_values(normalized_data, warnings)
    response = AnalysisResponse.model_validate(normalized_data)

    validated_predictions: list[PlayerPrediction] = []
    for prediction in response.analysis:
        if prediction.player_id not in known_player_ids:
            warnings.append(f"player {prediction.player_id} not in current game")
            continue

        canonical_role = _resolve_role(
            prediction.suspected_role, role_names_ru, role_alias_resolver
        )
        if canonical_role is None:
            warnings.append(
                f"unknown suspected role {prediction.suspected_role!r} for player "
                f"{prediction.player_id}; prediction dropped"
            )
            continue

        updated = prediction
        if canonical_role != prediction.suspected_role:
            updated = updated.model_copy(update={"suspected_role": canonical_role})

        alternative = updated.alternative
        if alternative is not None:
            canonical_alternative = _resolve_role(
                alternative.role, role_names_ru, role_alias_resolver
            )
            if canonical_alternative is None:
                warnings.append(
                    f"unknown alternative role {alternative.role!r} for player "
                    f"{prediction.player_id}; alternative dropped"
                )
                updated = updated.model_copy(update={"alternative": None})
            elif canonical_alternative != alternative.role:
                resolved_alternative = alternative.model_copy(
                    update={"role": canonical_alternative}
                )
                updated = updated.model_copy(update={"alternative": resolved_alternative})

        validated_predictions.append(updated)

    return response.model_copy(update={"analysis": validated_predictions}), warnings


def _resolve_role(
    role_name: str,
    role_names_ru: frozenset[str],
    role_alias_resolver: Callable[[str], str | None] | None,
) -> str | None:
    """Return a canonical role name or None when no known role matches."""
    if role_name in role_names_ru:
        return role_name
    if role_alias_resolver is not None:
        return role_alias_resolver(role_name)
    return None


def _normalize_confidence_values(data: Any, warnings: list[str]) -> None:
    """Clamp numeric confidence values and normalize overfull role pairs."""
    if not isinstance(data, dict):
        return
    predictions = data.get("analysis")
    if not isinstance(predictions, list):
        return

    for item in predictions:
        if not isinstance(item, dict):
            continue
        player_id = item.get("player_id", "unknown")
        confidence = _clamp_value(item.get("confidence"))
        if confidence is not None:
            value, changed = confidence
            item["confidence"] = value
            if changed:
                warnings.append(f"confidence for player {player_id} clamped to [0, 1]")

        alternative = item.get("alternative")
        if not isinstance(alternative, dict):
            continue
        alternative_confidence = _clamp_value(alternative.get("confidence"))
        if alternative_confidence is not None:
            value, changed = alternative_confidence
            alternative["confidence"] = value
            if changed:
                warnings.append(f"alternative confidence for player {player_id} clamped to [0, 1]")

        primary_value = item.get("confidence")
        alternative_value = alternative.get("confidence")
        if _is_number(primary_value) and _is_number(alternative_value):
            total = float(primary_value) + float(alternative_value)
            if total > 1.0 + 1e-9:
                item["confidence"] = float(primary_value) / total
                alternative["confidence"] = float(alternative_value) / total
                warnings.append(
                    f"confidence values normalized for player {player_id}: "
                    "their sum exceeded 1.0"
                )


def _clamp_value(value: Any) -> tuple[float, bool] | None:
    """Clamp numeric values to the confidence interval."""
    if not _is_number(value):
        return None
    number = float(value)
    if number != number:
        return None
    clamped = min(1.0, max(0.0, number))
    return clamped, clamped != number


def _is_number(value: Any) -> TypeGuard[int | float]:
    """Return whether a value can be treated as a numeric confidence."""
    return isinstance(value, (int, float))
