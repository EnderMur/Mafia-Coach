"""Tests for SFT example construction from a game state and a prediction."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jinja2
import pytest

from domain.events import (
    AliveListEvent,
    AliveListPayload,
    MessageEvent,
    MessagePayload,
    PhaseChangeEvent,
    PhaseChangePayload,
)
from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import load_roles
from features.analyzer.context_builder import PredictionSnapshot
from features.finetune.dataset_builder import build_example
from features.finetune.models import PredictionRow

ROLES_PATH = Path(__file__).parents[4] / "configs" / "roles.yaml"
BASE_TS = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_registry():
    return load_roles(ROLES_PATH)


def make_jinja_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader("infra/llm/prompts"),
        autoescape=False,
    )


def make_game_state(game_id: int = 1) -> GameState:
    state = GameState(game_id=game_id, registry=make_registry())
    state.register_player(101, "Вася")
    state.register_player(102, "Петя")
    state.apply(
        AliveListEvent(
            ts=BASE_TS,
            game_id=game_id,
            payload=AliveListPayload(
                alive_player_names=["Вася", "Петя"],
                role_count={"Мафия": 1, "Мирный житель": 1},
                total_players=2,
            ),
        )
    )
    state.apply(
        PhaseChangeEvent(
            ts=BASE_TS + timedelta(minutes=1),
            game_id=game_id,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
            ),
        )
    )
    state.apply(
        MessageEvent(
            ts=BASE_TS + timedelta(minutes=2),
            game_id=game_id,
            payload=MessagePayload(author_id=101, author_name="Вася", text="Я мирный"),
        )
    )
    return state


def make_prediction(
    *,
    player_id: int = 101,
    role: str = "Мафия",
    confidence: float = 0.8,
    reasoning: str | None = "Вася вёл себя подозрительно.",
    reason: str = "Уклончивые ответы.",
    correct: bool | None = True,
    ts: datetime = BASE_TS + timedelta(minutes=5),
) -> PredictionRow:
    return PredictionRow(
        id=1,
        game_id=1,
        ts=ts,
        player_id=player_id,
        predicted_role=role,
        confidence=confidence,
        reasoning=reasoning,
        reason=reason,
        correct=correct,
    )


def _assistant_payload(record) -> dict:
    return json.loads(record.messages[2].content)


def test_build_example_correct_prediction() -> None:
    """A correct prediction keeps the model's own answer as the target."""
    prediction = make_prediction()
    record = build_example(
        game_state=make_game_state(),
        prediction=prediction,
        previous_snapshots=[],
        registry=make_registry(),
        jinja_env=make_jinja_env(),
        final_role="Мафия",
        correct=True,
        source="current",
    )

    assert record is not None
    assert len(record.messages) == 3
    assert [m.role for m in record.messages] == ["system", "user", "assistant"]
    payload = _assistant_payload(record)
    analysis = payload["analysis"][0]
    assert analysis["suspected_role"] == "Мафия"
    assert analysis["confidence"] == 0.8
    assert analysis["reason"] == "Уклончивые ответы."
    assert payload["reasoning"] == "Вася вёл себя подозрительно."
    assert record.correct is True
    assert record.ground_truth_role == "Мафия"


def test_build_example_wrong_prediction_uses_ground_truth() -> None:
    """A wrong prediction is corrected to the ground-truth role."""
    prediction = make_prediction(role="Мирный житель", confidence=0.7)
    record = build_example(
        game_state=make_game_state(),
        prediction=prediction,
        previous_snapshots=[],
        registry=make_registry(),
        jinja_env=make_jinja_env(),
        final_role="Мафия",
        correct=False,
        source="current",
    )

    payload = _assistant_payload(record)
    analysis = payload["analysis"][0]
    assert analysis["suspected_role"] == "Мафия"
    assert analysis["confidence"] == 0.9
    assert analysis["reason"] == "Роль из финального отчёта бота."
    assert analysis["pattern_ids"] == []
    assert payload["reasoning"] is None
    assert record.correct is False
    assert record.ground_truth_role == "Мафия"


def test_build_example_wrong_prediction_drops_model_reasoning() -> None:
    """The model's wrong reasoning must not survive in the assistant target."""
    prediction = make_prediction(role="Мирный житель", reasoning="Вася вёл себя мирно.")
    record = build_example(
        game_state=make_game_state(),
        prediction=prediction,
        previous_snapshots=[],
        registry=make_registry(),
        jinja_env=make_jinja_env(),
        final_role="Мафия",
        correct=False,
        source="current",
    )
    payload = _assistant_payload(record)
    assert payload["reasoning"] is None


def test_build_example_prompt_contains_context() -> None:
    """The rendered user message carries the game context, not a placeholder."""
    prediction = make_prediction()
    record = build_example(
        game_state=make_game_state(),
        prediction=prediction,
        previous_snapshots=[],
        registry=make_registry(),
        jinja_env=make_jinja_env(),
        final_role="Мафия",
        correct=True,
        source="current",
    )
    user_content = record.messages[1].content
    assert "Вася" in user_content
    assert "Мафия" in user_content
    assert "Состав партии" in user_content
    system_content = record.messages[0].content
    assert "аналитик-тренер" in system_content


def test_build_example_previous_predictions_anchor() -> None:
    """Previous predictions reach both the system and the user prompt."""
    prediction = make_prediction()
    previous = [PredictionSnapshot(player_id=101, suspected_role="Мирный житель", confidence=0.5)]
    record = build_example(
        game_state=make_game_state(),
        prediction=prediction,
        previous_snapshots=previous,
        registry=make_registry(),
        jinja_env=make_jinja_env(),
        final_role="Мафия",
        correct=True,
        source="current",
    )
    user_content = record.messages[1].content
    assert "Мирный житель" in user_content
    assert "якорь" in user_content.lower()


def test_build_example_missing_player_returns_none() -> None:
    """A prediction for a player absent from the state is skipped."""
    prediction = make_prediction(player_id=999)
    record = build_example(
        game_state=make_game_state(),
        prediction=prediction,
        previous_snapshots=[],
        registry=make_registry(),
        jinja_env=make_jinja_env(),
        final_role="Мафия",
        correct=True,
        source="current",
    )
    assert record is None


def test_build_example_patterns_preserved_when_correct() -> None:
    """Club pattern metadata is kept for correct predictions."""
    prediction = make_prediction()
    prediction = prediction.model_copy(
        update={"pattern_ids": ["#2"], "pattern_descriptions": ["Паттерн клуба"]}
    )
    record = build_example(
        game_state=make_game_state(),
        prediction=prediction,
        previous_snapshots=[],
        registry=make_registry(),
        jinja_env=make_jinja_env(),
        final_role="Мафия",
        correct=True,
        source="current",
    )
    payload = _assistant_payload(record)
    assert payload["analysis"][0]["pattern_ids"] == ["#2"]
    assert payload["analysis"][0]["pattern_descriptions"] == ["Паттерн клуба"]


def test_build_example_reason_split_into_reasons() -> None:
    """A multi-line reason becomes the reasons list (capped at 5)."""
    prediction = make_prediction(reason="Первое.\nВторое.\nТретье.")
    record = build_example(
        game_state=make_game_state(),
        prediction=prediction,
        previous_snapshots=[],
        registry=make_registry(),
        jinja_env=make_jinja_env(),
        final_role="Мафия",
        correct=True,
        source="current",
    )
    payload = _assistant_payload(record)
    assert payload["analysis"][0]["reasons"] == ["Первое.", "Второе.", "Третье."]


@pytest.mark.parametrize("source", ["current", "history"])
def test_build_example_source_recorded(source: str) -> None:
    """The source tag is stored in the record metadata."""
    prediction = make_prediction()
    record = build_example(
        game_state=make_game_state(),
        prediction=prediction,
        previous_snapshots=[],
        registry=make_registry(),
        jinja_env=make_jinja_env(),
        final_role="Мафия",
        correct=True,
        source=source,
    )
    assert record is not None
    assert record.source == source
