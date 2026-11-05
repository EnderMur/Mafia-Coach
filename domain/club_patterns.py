"""Club pattern registry and configuration loading.

Pure domain module: no I/O beyond reading the YAML config file, no imports
from ``infra/``, ``features/``, or ``app/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from domain.exceptions import ConfigError

#: Allowed confidence tiers for club patterns.
CONFIDENCE_TIERS: frozenset[float] = frozenset({0.9, 0.8, 0.6})


class ClubPattern(BaseModel):
    """One documented club pattern with a confidence tier.

    ``id`` is a short pattern identifier such as ``"#1"`` through ``"#7"``.
    ``trigger`` names the kind of observable behaviour (for example
    ``claim_investigation`` or ``bot_investigation_result``).
    ``conclusion_role`` is the role the pattern points to (canonical
    ``name_ru`` from ``configs/roles.yaml``).
    ``confidence_tier`` is one of ``0.9``, ``0.8``, ``0.6``.
    ``requires_confirmation`` marks patterns that need a voting result
    before they can be applied (for example #1 and #5).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^#[1-9][0-9]*$")
    trigger: str = Field(min_length=1)
    conclusion_role: str = Field(min_length=1)
    confidence_tier: float = Field(ge=0.0, le=1.0)
    description: str = Field(min_length=1)
    requires_confirmation: bool = False

    @field_validator("confidence_tier")
    @classmethod
    def validate_confidence_tier(cls, value: float) -> float:
        """Reject confidence values outside the documented tiers."""
        if value not in CONFIDENCE_TIERS:
            raise ValueError(
                f"confidence_tier must be one of {sorted(CONFIDENCE_TIERS)}, got {value}"
            )
        return value


class ClubPatterns(BaseModel):
    """Validated, immutable collection of club patterns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    patterns: list[ClubPattern] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        """Reject duplicate pattern identifiers."""
        ids = [pattern.id for pattern in self.patterns]
        if len(ids) != len(set(ids)):
            raise ValueError("pattern ids must be unique")
        return self

    def get(self, pattern_id: str) -> ClubPattern:
        """Return a pattern by id, raising when it is not registered."""
        for pattern in self.patterns:
            if pattern.id == pattern_id:
                return pattern
        raise ConfigError(f"unknown club pattern: {pattern_id}")

    def by_tier(self, confidence_tier: float) -> tuple[ClubPattern, ...]:
        """Return all patterns with the given confidence tier."""
        return tuple(
            pattern for pattern in self.patterns if pattern.confidence_tier == confidence_tier
        )

    def all_ids(self) -> frozenset[str]:
        """Return all registered pattern identifiers."""
        return frozenset(pattern.id for pattern in self.patterns)

    def __len__(self) -> int:
        """Return the number of registered patterns."""
        return len(self.patterns)


def load_club_patterns(path: str | Path) -> ClubPatterns:
    """Load and validate club patterns from a UTF-8 YAML file.

    Args:
        path: YAML file location (expected top-level key ``patterns``).

    Returns:
        A validated, immutable collection of club patterns.

    Raises:
        ConfigError: If the file cannot be read, its YAML is invalid, or the
            parsed data do not satisfy the club-patterns schema.
    """
    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigError(f"config file not found: {config_path}") from error
    except (OSError, UnicodeError) as error:
        raise ConfigError(f"cannot read config {config_path}: {error}") from error

    try:
        raw_config: object = yaml.safe_load(raw_text)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error

    try:
        return ClubPatterns.model_validate(raw_config)
    except ValidationError as error:
        raise ConfigError(f"invalid club patterns {config_path}: {error}") from error
