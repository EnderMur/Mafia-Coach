"""Tests for conservative analyzer heuristic hints."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles
from domain.rules import RuleCheck, RulesEngine
from features.analyzer.heuristic_hints import HeuristicHint, compute_hints
from tests.unit.features.analyzer.test_context_builder import (
    BASE_TS,
    DeathCause,
    make_death_event,
    make_game_state,
    make_phase_change_event,
    player_id,
    player_name,
)

ROLES_PATH = Path(__file__).parents[4] / "configs" / "roles.yaml"


class _AllowingEngine(RulesEngine):
    """Engine stub that always allows a lynch."""

    @staticmethod
    def check_lynch(
        *,
        day: int,
        alive_count: int,
        has_argument: bool,
        is_rule_breaker: bool = False,
    ) -> RuleCheck:
        return RuleCheck.ok()


class _EngineWithoutRuleId(RulesEngine):
    """Engine stub that forbids without naming a rule."""

    @staticmethod
    def check_lynch(
        *,
        day: int,
        alive_count: int,
        has_argument: bool,
        is_rule_breaker: bool = False,
    ) -> RuleCheck:
        return RuleCheck(allowed=False)


@pytest.fixture(name="registry")
def registry_fixture() -> RoleRegistry:
    """Provide the real role registry."""
    return load_roles(ROLES_PATH)


def make_lynch_state(
    registry: RoleRegistry,
    *,
    day: int,
    lynch_count: int = 1,
    cause: DeathCause = "lynch",
    player_count: int = 4,
) -> GameState:
    """Build a day-N state with ``lynch_count`` deaths of the given cause."""
    state = make_game_state(player_count, registry=registry)
    state.apply(
        make_phase_change_event(
            BASE_TS,
            to_phase=Phase.DAY,
            day_number=day,
            from_phase=Phase.REGISTRATION,
        )
    )
    for index in range(lynch_count):
        state.apply(
            make_death_event(
                BASE_TS + timedelta(seconds=index + 1),
                victim_id=player_id(index),
                victim_name=player_name(index),
                cause=cause,
            )
        )
    return state


def test_empty_game_state_has_no_hints(registry: RoleRegistry) -> None:
    """Пустая партия не даёт подсказок."""
    assert compute_hints(game_state=make_game_state(registry=registry), registry=registry) == []


def test_late_day_has_no_hints(registry: RoleRegistry) -> None:
    """На пятый день линч разрешён — подсказок нет."""
    state = make_lynch_state(registry, day=5)

    assert compute_hints(game_state=state, registry=registry) == []


def test_early_lynch_produces_hint(registry: RoleRegistry) -> None:
    """Линч в первый день без аргументации даёт подсказку с весом -0.5."""
    state = make_lynch_state(registry, day=1)

    hints = compute_hints(game_state=state, registry=registry)

    assert len(hints) == 1
    assert isinstance(hints[0], HeuristicHint)
    assert hints[0].rule_id == "early_lynch_no_argument"
    assert hints[0].weight == -0.5
    assert hints[0].description


def test_second_day_lynch_produces_hint(registry: RoleRegistry) -> None:
    """Второй день всё ещё входит в запретное окно."""
    state = make_lynch_state(registry, day=2)

    hints = compute_hints(game_state=state, registry=registry)

    assert [hint.rule_id for hint in hints] == ["early_lynch_no_argument"]


def test_night_kill_is_not_a_lynch(registry: RoleRegistry) -> None:
    """Ночное убийство — не нарушение, подсказки нет."""
    state = make_lynch_state(registry, day=1, cause="night_kill")

    assert compute_hints(game_state=state, registry=registry) == []


def test_third_day_lynch_has_no_hint(registry: RoleRegistry) -> None:
    """Третий день вне запретного окна — подсказки нет."""
    state = make_lynch_state(registry, day=3)

    assert compute_hints(game_state=state, registry=registry) == []


def test_hint_weight_and_rule_id_are_valid(registry: RoleRegistry) -> None:
    """Вес всегда в [-1, 1], rule_id непустой."""
    state = make_lynch_state(registry, day=1)

    for hint in compute_hints(game_state=state, registry=registry):
        assert -1.0 <= hint.weight <= 1.0
        assert hint.rule_id.strip() != ""


def test_multiple_lynches_do_not_duplicate_hints(registry: RoleRegistry) -> None:
    """Три линча дают одну подсказку, а не три."""
    state = make_lynch_state(registry, day=1, lynch_count=3)

    hints = compute_hints(game_state=state, registry=registry)

    assert len(hints) == 1
    assert hints[0].rule_id == "early_lynch_no_argument"


def test_large_lobby_is_exempt_from_the_rule(registry: RoleRegistry) -> None:
    """Исключение правил для больших лобби отменяет подсказку."""
    state = make_lynch_state(registry, day=1, player_count=30)

    assert compute_hints(game_state=state, registry=registry) == []


def test_allowing_engine_suppresses_hint(registry: RoleRegistry) -> None:
    """Явно переданный движок правил определяет результат."""
    state = make_lynch_state(registry, day=1)

    hints = compute_hints(
        game_state=state,
        registry=registry,
        rules_engine=_AllowingEngine(),
    )

    assert hints == []


def test_forbid_without_rule_id_produces_no_hint(registry: RoleRegistry) -> None:
    """Запрет без идентификатора правила не превращается в подсказку."""
    state = make_lynch_state(registry, day=1)

    hints = compute_hints(
        game_state=state,
        registry=registry,
        rules_engine=_EngineWithoutRuleId(),
    )

    assert hints == []


def test_hint_is_immutable(registry: RoleRegistry) -> None:
    """Подсказка неизменяема."""
    hint = compute_hints(game_state=make_lynch_state(registry, day=1), registry=registry)[0]

    with pytest.raises(ValueError):
        hint.weight = 0.0
