"""Conservative heuristic hints derived from the game state and club rules.

Hints are advisory: the analyzer passes them to the LLM as extra context so the
model can look sharper at situations that already look like a meta-rule
violation. The module is pure — no I/O, no async, no clients — and returns an
empty list whenever there is no clear signal, which keeps the prompt honest.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from domain.events import DeathEvent
from domain.game_state import GameState
from domain.roles import RoleRegistry
from domain.rules import RulesEngine

_EARLY_LYNCH_WEIGHT: Final[float] = -0.5
"""Negative weight: a lynch the club rules would forbid reads as suspicious."""

_EARLY_LYNCH_RULE_ID: Final[str] = "early_lynch_no_argument"


class HeuristicHint(BaseModel):
    """One heuristic hint for the LLM context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    weight: float = Field(ge=-1.0, le=1.0)


def compute_hints(
    *,
    game_state: GameState,
    registry: RoleRegistry,
    rules_engine: RulesEngine | None = None,
) -> list[HeuristicHint]:
    """Generate conservative heuristic hints from the current game state.

    Hints are advisory and may be empty. Currently implemented:

    - ``early_lynch_no_argument`` — the club rules forbid a lynch on days 1-2
      without an argument, yet a :class:`DeathEvent` with ``cause='lynch'``
      happened inside that window. Weight ``-0.5``: the model should read the
      situation more sharply.

    At most one hint is emitted per rule even when several matching events exist,
    so hints never flood the prompt.

    Args:
        game_state: aggregate to inspect.
        registry: role registry (reserved for role-aware hints).
        rules_engine: engine to consult; a default :class:`RulesEngine` is used
            when omitted.

    Returns:
        Hints in a stable order; ``[]`` when no signal is present.
    """
    engine = rules_engine or RulesEngine()
    has_lynch = any(
        isinstance(event, DeathEvent) and event.payload.cause == "lynch"
        for event in game_state.events_by_kind("death")
    )
    if not has_lynch:
        return []

    check = engine.check_lynch(
        day=game_state.current_day(),
        alive_count=len(game_state.alive_players()),
        has_argument=False,
        is_rule_breaker=False,
    )
    if check.allowed or check.rule_id is None:
        return []

    return [
        HeuristicHint(
            rule_id=_EARLY_LYNCH_RULE_ID,
            description=(
                "Линч случился в первые дни без аргументации, хотя правила клуба это "
                "запрещают: оценивайте инициаторов голосования строже."
            ),
            weight=_EARLY_LYNCH_WEIGHT,
        )
    ]
