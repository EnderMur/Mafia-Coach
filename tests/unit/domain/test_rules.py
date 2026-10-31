"""Tests for pure club meta-rule predicates."""

from __future__ import annotations

import pytest

from domain.rules import RuleCheck, RulesEngine


@pytest.mark.parametrize(
    ("day", "alive_count", "has_argument", "is_rule_breaker", "allowed"),
    [
        (1, 10, False, False, False),
        (1, 10, True, False, True),
        (1, 10, False, True, True),
        (1, 29, False, False, True),
        (3, 10, False, False, True),
        (2, 10, False, False, False),
    ],
)
def test_check_lynch_contract(
    day: int,
    alive_count: int,
    has_argument: bool,
    is_rule_breaker: bool,
    allowed: bool,
) -> None:
    """Lynch checks honor the opening-day restriction and its exceptions."""
    result = RulesEngine.check_lynch(
        day=day,
        alive_count=alive_count,
        has_argument=has_argument,
        is_rule_breaker=is_rule_breaker,
    )
    assert result.allowed is allowed
    if allowed:
        assert result.rule_id is None
        assert result.reason is None
    else:
        assert result.rule_id == "lynch_no_argument"
        assert result.reason is not None


@pytest.mark.parametrize(
    ("night", "role_id", "target_has_antitarget", "alive_count", "allowed"),
    [
        (1, "commissioner", True, 10, False),
        (1, "commissioner", True, 8, True),
        (1, "civilian", True, 10, True),
        (2, "commissioner", True, 10, True),
        (1, "commissioner", False, 10, True),
    ],
)
def test_check_night_action_contract(
    night: int,
    role_id: str,
    target_has_antitarget: bool,
    alive_count: int,
    allowed: bool,
) -> None:
    """Night-action checks honor role, night, target, and alive-count rules."""
    result = RulesEngine.check_night_action(
        night=night,
        role_id=role_id,
        target_has_antitarget=target_has_antitarget,
        alive_count=alive_count,
    )
    assert result.allowed is allowed
    if allowed:
        assert result.rule_id is None
    else:
        assert result.rule_id == "antitarget_night"
        assert result.reason is not None


def test_rule_check_factory_methods() -> None:
    """RuleCheck factories create the documented allowed and forbidden forms."""
    allowed = RuleCheck.ok()
    forbidden = RuleCheck.forbid("example", "Short reason")
    assert allowed.allowed is True
    assert allowed.rule_id is None
    assert allowed.reason is None
    assert forbidden.allowed is False
    assert forbidden.rule_id == "example"
    assert forbidden.reason == "Short reason"


def test_rule_registry_has_unique_entries() -> None:
    """The static rule registry contains multiple uniquely identified rules."""
    rules = RulesEngine.registry()
    ids = [rule.id for rule in rules]
    assert len(rules) >= 2
    assert len(ids) == len(set(ids))
