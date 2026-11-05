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

PATTERNS_PATH = Path(__file__).parents[3] / "configs" / "club_patterns.yaml"


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


def test_load_real_config_has_7_patterns() -> None:
    """The real configuration contains the contracted 7 club patterns."""
    assert len(load_real_patterns()) == 7


def test_real_pattern_ids_are_hash1_to_hash7() -> None:
    """Every configured identifier is #1..#7 in order."""
    patterns = load_real_patterns()
    assert [pattern.id for pattern in patterns.patterns] == [
        "#1",
        "#2",
        "#3",
        "#4",
        "#5",
        "#6",
        "#7",
    ]


def test_real_pattern_tiers_match_contract() -> None:
    """#2/#7 are 0.9, #1/#5 are 0.8, #3/#4/#6 are 0.6."""
    patterns = load_real_patterns()
    tiers = {pattern.id: pattern.confidence_tier for pattern in patterns.patterns}
    assert tiers["#2"] == 0.9
    assert tiers["#7"] == 0.9
    assert tiers["#1"] == 0.8
    assert tiers["#5"] == 0.8
    assert tiers["#3"] == 0.6
    assert tiers["#4"] == 0.6
    assert tiers["#6"] == 0.6


def test_real_patterns_require_confirmation_flags() -> None:
    """#1 and #5 require confirmation; the rest do not."""
    patterns = load_real_patterns()
    flags = {pattern.id: pattern.requires_confirmation for pattern in patterns.patterns}
    assert flags["#1"] is True
    assert flags["#5"] is True
    assert flags["#2"] is False
    assert flags["#3"] is False
    assert flags["#4"] is False
    assert flags["#6"] is False
    assert flags["#7"] is False


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
    registry = load_real_patterns()
    assert {p.id for p in registry.by_tier(0.9)} == {"#2", "#7"}
    assert {p.id for p in registry.by_tier(0.8)} == {"#1", "#5"}
    assert {p.id for p in registry.by_tier(0.6)} == {"#3", "#4", "#6"}
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
