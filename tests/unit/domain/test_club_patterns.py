"""Tests for club-pattern configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from domain.club_patterns import (
    CONFIDENCE_TIERS,
    ClubPattern,
    ClubPatterns,
    load_club_patterns,
)
from domain.exceptions import ConfigError

PATTERNS_PATH = Path(__file__).parents[3] / "configs" / "club_patterns.example.yaml"


def make_pattern(**overrides: Any) -> dict[str, Any]:
    """Create a minimal valid club-pattern mapping for configuration tests."""
    pattern: dict[str, Any] = {
        "id": "#9",
        "trigger": "claim_investigation",
        "conclusion_role": "Мафия",
        "confidence_tier": 0.8,
        "description": "Тестовое описание паттерна.",
        "requires_confirmation": False,
    }
    pattern.update(overrides)
    return pattern


def write_config(tmp_path: Path, patterns: list[dict[str, Any]]) -> Path:
    """Write a club-patterns config fixture into a temporary directory."""
    path = tmp_path / "club_patterns.yaml"
    path.write_text(json.dumps({"patterns": patterns}, ensure_ascii=False), encoding="utf-8")
    return path


def load_real_patterns() -> ClubPatterns:
    """Load the repository's real club-patterns configuration."""
    return load_club_patterns(PATTERNS_PATH)


def test_example_config_contains_one_pattern() -> None:
    """The example contains one valid, documented pattern."""
    patterns = load_real_patterns()
    assert len(patterns) == 1
    pattern = patterns.patterns[0]
    assert pattern.id == "#1"
    assert pattern.confidence_tier == 0.8
    assert pattern.requires_confirmation is True
    assert pattern.conclusion_role == "Комиссар Каттани"


def test_real_pattern_confidence_tiers_are_valid() -> None:
    """Every tier belongs to the documented set {0.9, 0.8, 0.6}."""
    assert all(
        pattern.confidence_tier in CONFIDENCE_TIERS for pattern in load_real_patterns().patterns
    )


def test_pattern_is_frozen() -> None:
    """Pattern attributes cannot be reassigned after construction."""
    pattern = load_real_patterns().patterns[0]
    with pytest.raises(ValidationError):
        pattern.description = "Изменение"


def test_invalid_confidence_tier_rejected(tmp_path: Path) -> None:
    """Confidence values outside the documented tiers are rejected."""
    path = write_config(tmp_path, [make_pattern(confidence_tier=0.5)])
    with pytest.raises(ConfigError):
        load_club_patterns(path)


def test_duplicate_pattern_id_rejected(tmp_path: Path) -> None:
    """Duplicate pattern ids invalidate a configuration."""
    path = write_config(tmp_path, [make_pattern(), make_pattern(id="#9", description="Другой")])
    with pytest.raises(ConfigError, match="unique"):
        load_club_patterns(path)


def test_yaml_and_missing_file_errors_are_wrapped(tmp_path: Path) -> None:
    """Missing files and malformed YAML become contextual config errors."""
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError, match="not found"):
        load_club_patterns(missing)
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("patterns: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_club_patterns(malformed)


def test_registry_get_by_id_and_unknown_id() -> None:
    """Registry id lookup returns patterns and reports missing ids."""
    registry = load_real_patterns()
    pattern = registry.patterns[0]
    assert registry.get(pattern.id) == pattern
    with pytest.raises(ConfigError, match="unknown club pattern"):
        registry.get("#999")


def test_by_tier_groups_patterns_correctly() -> None:
    """by_tier returns exactly the patterns of the requested tier."""
    registry = ClubPatterns.model_validate(
        {
            "patterns": [
                make_pattern(id=f"#{i}", confidence_tier=tier)
                for i, tier in enumerate((0.9, 0.8, 0.6), start=1)
            ]
        }
    )
    assert {p.id for p in registry.by_tier(0.9)} == {"#1"}
    assert {p.id for p in registry.by_tier(0.8)} == {"#2"}
    assert {p.id for p in registry.by_tier(0.6)} == {"#3"}
    assert registry.by_tier(0.5) == ()


def test_config_rejects_empty_patterns_and_extra_fields(tmp_path: Path) -> None:
    """Empty pattern lists and unknown configuration fields are rejected."""
    empty_path = write_config(tmp_path, [])
    with pytest.raises(ConfigError):
        load_club_patterns(empty_path)
    extra_path = tmp_path / "extra.yaml"
    extra_path.write_text(
        json.dumps({"patterns": [make_pattern()], "unexpected": True}), encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_club_patterns(extra_path)


def test_pattern_model_rejects_unknown_fields() -> None:
    """Direct ClubPattern construction rejects undeclared fields."""
    with pytest.raises(ValidationError):
        ClubPattern.model_validate(
            {
                "id": "#9",
                "trigger": "claim_investigation",
                "conclusion_role": "Мафия",
                "confidence_tier": 0.8,
                "description": "x",
                "unknown_field": True,
            }
        )


def test_example_config_is_flat_schema() -> None:
    """The example YAML conforms to the flat pattern schema."""
    registry = load_club_patterns(PATTERNS_PATH)
    assert len(registry) == 1
    assert registry.all_ids() == frozenset({"#1"})
    for pattern in registry.patterns:
        assert isinstance(pattern.trigger, str) and pattern.trigger
        assert isinstance(pattern.conclusion_role, str) and pattern.conclusion_role
        assert isinstance(pattern.description, str) and pattern.description
        assert isinstance(pattern.requires_confirmation, bool)
        assert pattern.confidence_tier in CONFIDENCE_TIERS
