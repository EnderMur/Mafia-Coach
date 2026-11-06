"""Evaluation of a model on the SFT dataset: metrics and report.

The evaluation logic is dependency-free: model inference is injected as a
``predict_fn`` callable, so this module can be unit-tested without torch,
transformers, or peft installed. The CLI script in ``scripts/evaluate_model.py``
builds the predict function with lazy heavy imports.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from domain.roles import Faction, RoleRegistry
from features.finetune.models import DatasetRecord


@dataclass(frozen=True)
class EvalSample:
    """One evaluated example: model prediction vs ground truth."""

    record: DatasetRecord
    predicted_role: str | None
    confidence: float
    exact_correct: bool
    faction_correct: bool


class EvaluationReport(BaseModel):
    """Aggregated evaluation metrics for one model configuration."""

    model_config = ConfigDict(frozen=True)

    base_model: str
    adapter: str | None = None
    examples: int
    accuracy_exact: float
    accuracy_by_faction: dict[str, float]
    avg_confidence_correct: float
    avg_confidence_incorrect: float


def extract_json(raw: str) -> dict[str, Any] | None:
    """Parse the first JSON object from model output, tolerating code fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, TypeError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def parse_prediction(raw: str, player_id: int) -> tuple[str | None, float]:
    """Extract ``(suspected_role, confidence)`` for ``player_id`` from output."""
    payload = extract_json(raw)
    if payload is None:
        return None, 0.0
    analysis = payload.get("analysis")
    if not isinstance(analysis, list):
        return None, 0.0
    for entry in analysis:
        if not isinstance(entry, dict):
            continue
        try:
            entry_player_id = int(entry.get("player_id", -1))
        except (TypeError, ValueError):
            continue
        if entry_player_id != player_id:
            continue
        suspected = entry.get("suspected_role")
        confidence = float(entry.get("confidence", 0.0))
        return suspected, confidence
    return None, 0.0


def _canonical_name(role_name: str | None, registry: RoleRegistry) -> str | None:
    """Resolve a role name to its canonical Russian form via the registry."""
    if role_name is None:
        return None
    role = registry.resolve_alias(role_name)
    return role.name_ru if role is not None else None


def _faction_of(role_name: str | None, registry: RoleRegistry) -> Faction | None:
    """Resolve a role name to its faction via the registry."""
    if role_name is None:
        return None
    role = registry.resolve_alias(role_name)
    return role.faction if role is not None else None


def evaluate_records(
    records: list[DatasetRecord],
    *,
    predict_fn: Callable[[DatasetRecord], str],
    registry: RoleRegistry,
) -> list[EvalSample]:
    """Run ``predict_fn`` on every record and compare with ground truth."""
    samples: list[EvalSample] = []
    for record in records:
        raw = predict_fn(record)
        predicted_role, confidence = parse_prediction(raw, record.player_id)
        truth_canonical = _canonical_name(record.ground_truth_role, registry)
        pred_canonical = _canonical_name(predicted_role, registry)
        exact_correct = pred_canonical is not None and pred_canonical == truth_canonical
        truth_faction = _faction_of(record.ground_truth_role, registry)
        pred_faction = _faction_of(predicted_role, registry)
        faction_correct = (
            pred_faction is not None and truth_faction is not None and pred_faction == truth_faction
        )
        samples.append(
            EvalSample(
                record=record,
                predicted_role=predicted_role,
                confidence=confidence,
                exact_correct=exact_correct,
                faction_correct=faction_correct,
            )
        )
    return samples


def compute_report(
    samples: list[EvalSample],
    *,
    base_model: str,
    adapter: str | None,
    registry: RoleRegistry,
) -> EvaluationReport:
    """Aggregate per-sample results into an :class:`EvaluationReport`."""
    n = len(samples)
    if n == 0:
        return EvaluationReport(
            base_model=base_model,
            adapter=adapter,
            examples=0,
            accuracy_exact=0.0,
            accuracy_by_faction={},
            avg_confidence_correct=0.0,
            avg_confidence_incorrect=0.0,
        )
    correct = [s for s in samples if s.exact_correct]
    incorrect = [s for s in samples if not s.exact_correct]
    by_faction: dict[str, list[bool]] = {}
    for sample in samples:
        faction = _faction_of(sample.record.ground_truth_role, registry)
        if faction is None:
            continue
        by_faction.setdefault(faction.value, []).append(sample.faction_correct)
    accuracy_by_faction = {faction: sum(vals) / len(vals) for faction, vals in by_faction.items()}
    return EvaluationReport(
        base_model=base_model,
        adapter=adapter,
        examples=n,
        accuracy_exact=len(correct) / n,
        accuracy_by_faction=accuracy_by_faction,
        avg_confidence_correct=(
            sum(s.confidence for s in correct) / len(correct) if correct else 0.0
        ),
        avg_confidence_incorrect=(
            sum(s.confidence for s in incorrect) / len(incorrect) if incorrect else 0.0
        ),
    )


def render_table(base: EvaluationReport, adapter: EvaluationReport | None) -> str:
    """Render a comparative metrics table as plain text."""

    def pct(value: float) -> str:
        return f"{value:.1%}"

    def conf(value: float) -> str:
        return f"{value:.3f}"

    def cell(value: str | None) -> str:
        return value if value is not None else "—"

    lines: list[str] = []
    lines.append(f"{'Metric':<30} {'Base':>12} {'Adapter':>12}")
    lines.append("-" * 54)
    lines.append(
        f"{'Examples':<30} {base.examples:>12} "
        f"{cell(str(adapter.examples) if adapter else None):>12}"
    )
    lines.append(
        f"{'Accuracy (exact)':<30} {pct(base.accuracy_exact):>12} "
        f"{cell(pct(adapter.accuracy_exact) if adapter else None):>12}"
    )
    factions = sorted(
        set(base.accuracy_by_faction.keys())
        | (set(adapter.accuracy_by_faction.keys()) if adapter else set())
    )
    lines.append(f"{'Accuracy by faction:':<30}")
    for faction in factions:
        b = base.accuracy_by_faction.get(faction, 0.0)
        a = adapter.accuracy_by_faction.get(faction, 0.0) if adapter else None
        lines.append(
            f"{'  ' + faction:<30} {pct(b):>12} {cell(pct(a) if a is not None else None):>12}"
        )
    lines.append(
        f"{'Avg confidence (correct)':<30} {conf(base.avg_confidence_correct):>12} "
        f"{cell(conf(adapter.avg_confidence_correct) if adapter else None):>12}"
    )
    lines.append(
        f"{'Avg confidence (wrong)':<30} {conf(base.avg_confidence_incorrect):>12} "
        f"{cell(conf(adapter.avg_confidence_incorrect) if adapter else None):>12}"
    )
    return "\n".join(lines)
