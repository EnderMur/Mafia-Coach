"""End-to-end stub: game state → LLM context → rendered prompt messages."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import jinja2
import pytest

from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles
from features.analyzer.context_builder import build_context, to_llm_messages
from features.analyzer.heuristic_hints import compute_hints
from tests.unit.features.analyzer.test_context_builder import (
    BASE_TS,
    make_death_event,
    make_game_state,
    make_phase_change_event,
    player_id,
    player_name,
)

ROLES_PATH = Path(__file__).parents[4] / "configs" / "roles.yaml"


@pytest.fixture(name="registry")
def registry_fixture() -> RoleRegistry:
    """Provide the real role registry."""
    return load_roles(ROLES_PATH)


@pytest.fixture(name="jinja_env")
def jinja_env_fixture() -> jinja2.Environment:
    """Provide the prompt environment used by the application."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader("infra/llm/prompts"),
        autoescape=False,
    )


def test_context_to_messages_pipeline(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """3 игрока, 5 сообщений и 1 смерть рендерятся в валидный список сообщений."""
    state = make_game_state(3, 5, registry=registry)
    state.apply(make_phase_change_event(BASE_TS + timedelta(seconds=10), Phase.DAY, 1))
    state.apply(
        make_death_event(
            BASE_TS + timedelta(seconds=11),
            victim_id=player_id(1),
            victim_name=player_name(1),
            cause="night_kill",
        )
    )

    context = build_context(game_state=state, registry=registry)
    hints = compute_hints(game_state=state, registry=registry)
    messages = to_llm_messages(
        context,
        registry=registry,
        jinja_env=jinja_env,
        club_notes_summary="; ".join(hint.description for hint in hints),
    )

    assert len(context.players) == 3
    assert len(context.recent_messages) == 5
    assert len(context.recent_deaths) == 1
    assert context.truncated_messages == 0
    assert hints == []
    assert isinstance(messages, list)
    assert len(messages) == 2
    for message in messages:
        assert set(message) == {"role", "content"}
        assert message["content"].strip()
        assert "{{" not in message["content"]
        assert "{%" not in message["content"]
    assert "Игроки" in messages[1]["content"]
    assert "Петя" in messages[1]["content"]
