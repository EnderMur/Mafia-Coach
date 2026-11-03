"""Tests for the analyzer LLM context builder."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import jinja2
import pytest
from jinja2 import DictLoader

from domain.events import (
    AliveListEvent,
    AliveListPayload,
    DeathEvent,
    DeathPayload,
    MessageEvent,
    MessagePayload,
    PhaseChangeEvent,
    PhaseChangePayload,
    RoleRevealEvent,
    RoleRevealPayload,
    VoteEvent,
    VotePayload,
)
from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles
from features.analyzer.context_builder import (
    DEFAULT_MAX_MESSAGES,
    DEFAULT_RESPONSE_SCHEMA_JSON,
    DeathInfo,
    LLMContext,
    MessageInfo,
    PhaseInfo,
    PlayerInfo,
    PredictionSnapshot,
    VoteInfo,
    _render_role_notes,
    build_context,
    to_llm_messages,
)

ROLES_PATH = Path(__file__).parents[4] / "configs" / "roles.yaml"
BASE_TS = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_NAMES = ("Вася", "Петя", "Маша", "Коля", "Света")
DeathCause = Literal["night_kill", "lynch", "unknown"]


# --- helpers --------------------------------------------------------------


def make_registry() -> RoleRegistry:
    """Load the real club role configuration."""
    return load_roles(ROLES_PATH)


def player_name(index: int) -> str:
    """Return a fictional display name for a player index."""
    if index < len(_NAMES):
        return _NAMES[index]
    return f"user_{index + 1}"


def player_id(index: int) -> int:
    """Return a synthetic player identifier for a player index."""
    return 100 + index


def make_message_event(
    ts: datetime,
    author_id: int = 100,
    author_name: str = "Вася",
    text: str = "Я мирный",
    *,
    game_id: int = 1,
) -> MessageEvent:
    """Build a message event for tests."""
    return MessageEvent(
        ts=ts,
        game_id=game_id,
        payload=MessagePayload(author_id=author_id, author_name=author_name, text=text),
    )


def make_death_event(
    ts: datetime,
    victim_id: int = 100,
    victim_name: str = "Вася",
    cause: DeathCause = "night_kill",
    *,
    game_id: int = 1,
) -> DeathEvent:
    """Build a death event for tests."""
    return DeathEvent(
        ts=ts,
        game_id=game_id,
        payload=DeathPayload(victim_id=victim_id, victim_name=victim_name, cause=cause),
    )


def make_phase_change_event(
    ts: datetime,
    to_phase: Phase = Phase.DAY,
    day_number: int = 1,
    from_phase: Phase = Phase.REGISTRATION,
    *,
    game_id: int = 1,
) -> PhaseChangeEvent:
    """Build a phase change event for tests."""
    return PhaseChangeEvent(
        ts=ts,
        game_id=game_id,
        payload=PhaseChangePayload(
            from_phase=from_phase,
            to_phase=to_phase,
            day_number=day_number,
        ),
    )


def make_game_state(
    player_count: int = 0,
    message_count: int = 0,
    *,
    dead_players: int = 0,
    game_id: int = 1,
    registry: RoleRegistry | None = None,
) -> GameState:
    """Build a game state with ``player_count`` players and ``message_count`` messages."""
    state = GameState(
        game_id=game_id, registry=registry if registry is not None else make_registry()
    )
    for index in range(player_count):
        state.register_player(player_id(index), player_name(index))
    for index in range(min(dead_players, player_count)):
        state.apply(
            make_death_event(
                BASE_TS + timedelta(seconds=index),
                victim_id=player_id(index),
                victim_name=player_name(index),
            )
        )
    for index in range(message_count):
        state.apply(
            make_message_event(
                BASE_TS + timedelta(seconds=index),
                author_id=player_id(0),
                author_name=player_name(0),
                text=f"сообщение {index}",
            )
        )
    return state


@pytest.fixture(name="registry")
def registry_fixture() -> RoleRegistry:
    """Provide the real role registry."""
    return make_registry()


@pytest.fixture(name="jinja_env")
def jinja_env_fixture() -> jinja2.Environment:
    """Provide the prompt environment used by the application."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader("infra/llm/prompts"),
        autoescape=False,
    )


# --- build_context --------------------------------------------------------


def test_empty_game_state_builds_empty_context(registry: RoleRegistry) -> None:
    """Пустой GameState даёт пустой контекст без обрезания."""
    context = build_context(game_state=make_game_state(registry=registry), registry=registry)

    assert isinstance(context, LLMContext)
    assert context.game_id == 1
    assert context.players == []
    assert context.recent_messages == []
    assert context.recent_deaths == []
    assert context.recent_votes == []
    assert context.previous_predictions == []
    assert context.truncated_messages == 0
    assert context.total_messages_seen == 0
    assert context.phase == PhaseInfo(current_phase=Phase.REGISTRATION, current_day=0)


def test_players_include_alive_and_dead(registry: RoleRegistry) -> None:
    """В контекст попадают и живые, и мёртвые игроки."""
    state = make_game_state(3, dead_players=2, registry=registry)

    context = build_context(game_state=state, registry=registry)

    assert len(context.players) == 3
    assert [player.alive for player in context.players] == [False, False, True]
    assert [player.display_name for player in context.players] == ["Вася", "Петя", "Маша"]
    assert context.players[0] == PlayerInfo(
        player_id=100, display_name="Вася", alive=False, known_role_name_ru=None
    )


def test_known_role_uses_russian_role_name(registry: RoleRegistry) -> None:
    """Раскрытая роль попадает в контекст как name_ru."""
    state = make_game_state(1, registry=registry)
    role = registry.all_roles()[0]
    state.apply(
        RoleRevealEvent(
            ts=BASE_TS,
            game_id=1,
            payload=RoleRevealPayload(
                player_id=100,
                player_name="Вася",
                role_name_ru=role.name_ru,
            ),
        )
    )

    context = build_context(game_state=state, registry=registry)

    assert context.players[0].known_role_name_ru == role.name_ru
    assert context.players[0].known_role_name_ru != role.id


def test_player_without_reveal_has_no_known_role(registry: RoleRegistry) -> None:
    """Без раскрытия известной роли нет."""
    state = make_game_state(2, registry=registry)

    context = build_context(game_state=state, registry=registry)

    assert [player.known_role_name_ru for player in context.players] == [None, None]


def test_message_window_keeps_all_messages_when_short(registry: RoleRegistry) -> None:
    """Пять сообщений попадают целиком, обрезания нет."""
    state = make_game_state(1, 5, registry=registry)

    context = build_context(game_state=state, registry=registry)

    assert len(context.recent_messages) == 5
    assert context.truncated_messages == 0
    assert context.total_messages_seen == 5
    assert all(isinstance(message, MessageInfo) for message in context.recent_messages)


def test_message_window_truncates_to_max_messages(registry: RoleRegistry) -> None:
    """Из 250 сообщений остаются последние 200, 50 считаются обрезанными."""
    state = make_game_state(1, 250, registry=registry)

    context = build_context(game_state=state, registry=registry, max_messages=200)

    assert len(context.recent_messages) == 200
    assert context.truncated_messages == 50
    assert context.total_messages_seen == 250
    assert context.recent_messages[0].text == "сообщение 50"
    assert context.recent_messages[-1].text == "сообщение 249"


def test_zero_max_messages_drops_whole_window(registry: RoleRegistry) -> None:
    """max_messages=0 оставляет окно пустым, но считает все сообщения."""
    state = make_game_state(1, 3, registry=registry)

    context = build_context(game_state=state, registry=registry, max_messages=0)

    assert context.recent_messages == []
    assert context.truncated_messages == 3
    assert context.total_messages_seen == 3


def test_negative_max_messages_is_rejected(registry: RoleRegistry) -> None:
    """Отрицательный размер окна — ошибка вызывающего кода."""
    with pytest.raises(ValueError, match="max_messages"):
        build_context(
            game_state=make_game_state(1, 1, registry=registry),
            registry=registry,
            max_messages=-1,
        )


def test_default_max_messages_matches_readme(registry: RoleRegistry) -> None:
    """Окно по умолчанию — 200 сообщений (README §3.1)."""
    assert DEFAULT_MAX_MESSAGES == 200


def test_deaths_are_collected(registry: RoleRegistry) -> None:
    """Все смерти попадают в recent_deaths с причиной."""
    state = make_game_state(4, registry=registry)
    causes: tuple[DeathCause, ...] = ("night_kill", "lynch", "unknown")
    for index, cause in enumerate(causes):
        state.apply(
            make_death_event(
                BASE_TS + timedelta(seconds=10 + index),
                victim_id=player_id(index),
                victim_name=player_name(index),
                cause=cause,
            )
        )

    context = build_context(game_state=state, registry=registry)

    assert len(context.recent_deaths) == 3
    assert [death.cause for death in context.recent_deaths] == list(causes)
    assert context.recent_deaths[1] == DeathInfo(
        ts=BASE_TS + timedelta(seconds=11), victim_name="Петя", cause="lynch"
    )


def test_phase_change_is_not_a_message(registry: RoleRegistry) -> None:
    """PhaseChangeEvent не попадает в recent_messages, но меняет фазу."""
    state = make_game_state(2, registry=registry)
    state.apply(make_phase_change_event(BASE_TS, to_phase=Phase.DAY, day_number=1))

    context = build_context(game_state=state, registry=registry)

    assert context.recent_messages == []
    assert context.total_messages_seen == 0
    assert context.phase == PhaseInfo(current_phase=Phase.DAY, current_day=1)


def test_votes_are_not_messages(registry: RoleRegistry) -> None:
    """VoteEvent не попадает в recent_messages."""
    state = make_game_state(2, 1, registry=registry)
    state.apply(
        VoteEvent(
            ts=BASE_TS + timedelta(seconds=1),
            game_id=1,
            payload=VotePayload(
                voter_id=100,
                voter_name="Вася",
                target_id=101,
                target_name="Петя",
            ),
        )
    )

    context = build_context(game_state=state, registry=registry)

    assert len(context.recent_messages) == 1
    assert context.recent_messages[0].author_name == "Вася"
    assert context.total_messages_seen == 1


def test_build_context_includes_recent_votes(registry: RoleRegistry) -> None:
    """Votes retain their timestamp, names, order, and optional target."""
    state = make_game_state(2, registry=registry)
    for index, target_name in enumerate(("Петя", None, "Вася")):
        state.apply(
            VoteEvent(
                ts=BASE_TS + timedelta(seconds=index),
                game_id=1,
                payload=VotePayload(
                    voter_id=100,
                    voter_name=f"Избиратель {index}",
                    target_id=101 if target_name else None,
                    target_name=target_name,
                ),
            )
        )

    context = build_context(game_state=state, registry=registry)

    assert context.recent_votes == [
        VoteInfo(ts=BASE_TS, voter_name="Избиратель 0", target_name="Петя"),
        VoteInfo(ts=BASE_TS + timedelta(seconds=1), voter_name="Избиратель 1", target_name=None),
        VoteInfo(ts=BASE_TS + timedelta(seconds=2), voter_name="Избиратель 2", target_name="Вася"),
    ]
    assert context.recent_messages == []


def test_recent_votes_limit_20(registry: RoleRegistry) -> None:
    """Only the newest twenty applied votes are retained, in their original order."""
    state = make_game_state(2, registry=registry)
    for index in range(25):
        state.apply(
            VoteEvent(
                ts=BASE_TS + timedelta(seconds=index),
                game_id=1,
                payload=VotePayload(
                    voter_id=100,
                    voter_name=f"Избиратель {index}",
                    target_id=101,
                    target_name="Петя",
                ),
            )
        )

    context = build_context(game_state=state, registry=registry)

    assert len(context.recent_votes) == 20
    assert [vote.voter_name for vote in context.recent_votes] == [
        f"Избиратель {index}" for index in range(5, 25)
    ]
    assert context.recent_votes[0].ts == BASE_TS + timedelta(seconds=5)
    assert context.recent_votes[-1].ts == BASE_TS + timedelta(seconds=24)


def test_render_role_notes(registry: RoleRegistry) -> None:
    """Real role notes are rendered; blank notes are ignored and whitespace stripped."""
    notes = _render_role_notes(registry)
    assert "Комиссар" in notes
    assert "Стреляет после проверки" in notes

    first, second = registry.all_roles()[:2]
    trimmed_registry = RoleRegistry(
        [
            first.model_copy(update={"notes": "  проверить голос  "}),
            second.model_copy(update={"notes": "  "}),
        ]
    )
    assert _render_role_notes(trimmed_registry) == f"- {first.name_ru}: проверить голос"


def test_previous_predictions_are_preserved(registry: RoleRegistry) -> None:
    """Прошлые предсказания переносятся в контекст без изменений."""
    predictions = [
        PredictionSnapshot(player_id=100, suspected_role="Мафия", confidence=0.7),
        PredictionSnapshot(player_id=101, suspected_role="Комиссар", confidence=0.2),
    ]

    context = build_context(
        game_state=make_game_state(2, registry=registry),
        registry=registry,
        previous_predictions=predictions,
    )

    assert context.previous_predictions == predictions


def test_recent_messages_are_chronological(registry: RoleRegistry) -> None:
    """Сообщения идут от старых к новым."""
    state = make_game_state(1, 5, registry=registry)

    context = build_context(game_state=state, registry=registry)

    timestamps = [message.ts for message in context.recent_messages]
    assert timestamps == sorted(timestamps)
    assert [message.text for message in context.recent_messages] == [
        "сообщение 0",
        "сообщение 1",
        "сообщение 2",
        "сообщение 3",
        "сообщение 4",
    ]


def test_build_context_includes_role_count(registry: RoleRegistry) -> None:
    """role_count из GameState попадает в LLMContext."""
    state = make_game_state(registry=registry)
    state.apply(
        AliveListEvent(
            ts=BASE_TS,
            game_id=1,
            payload=AliveListPayload(
                alive_player_names=["Вася"],
                role_count={"Мафия": 5},
                total_players=10,
            ),
        )
    )
    context = build_context(game_state=state, registry=registry)
    assert context.role_count == {"Мафия": 5}
    assert context.total_players == 10


def test_build_context_role_count_none_if_not_set(registry: RoleRegistry) -> None:
    """Без AliveListEvent composition-поля контекста — None (back-compat)."""
    context = build_context(game_state=make_game_state(registry=registry), registry=registry)
    assert context.role_count is None
    assert context.total_players is None


# --- to_llm_messages ------------------------------------------------------


def test_to_llm_messages_renders_both_templates(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """system + analyze рендерятся в список из двух сообщений без остатков Jinja."""
    state = make_game_state(3, 2, dead_players=1, registry=registry)
    state.apply(make_phase_change_event(BASE_TS + timedelta(seconds=5), Phase.DAY, 1))
    context = build_context(game_state=state, registry=registry)

    messages = to_llm_messages(context, registry=registry, jinja_env=jinja_env)

    assert isinstance(messages, list)
    assert len(messages) == 2
    assert [message["role"] for message in messages] == ["system", "user"]
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    for role in registry.all_roles():
        assert role.name_ru in system_content
    assert "AnalysisResponse" in system_content
    assert "Вася" in user_content
    assert "ID 100" in user_content
    assert "Игровой день: 1" in user_content
    assert "{{" not in system_content
    assert "{%" not in user_content


def test_to_llm_messages_renders_empty_context(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """Пустой контекст рендерится без падений и пустых секций-заглушек."""
    context = build_context(game_state=make_game_state(registry=registry), registry=registry)

    messages = to_llm_messages(context, registry=registry, jinja_env=jinja_env)

    assert len(messages) == 2
    assert "Список игроков пуст." in messages[1]["content"]
    assert "Сообщений нет." in messages[1]["content"]
    assert "{{" not in messages[1]["content"]


def test_to_llm_messages_marks_known_role(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """В players известная роль передаётся как name_ru."""
    state = make_game_state(1, registry=registry)
    role = registry.all_roles()[0]
    state.apply(
        RoleRevealEvent(
            ts=BASE_TS,
            game_id=1,
            payload=RoleRevealPayload(
                player_id=100,
                player_name="Вася",
                role_name_ru=role.name_ru,
            ),
        )
    )
    context = build_context(game_state=state, registry=registry)

    messages = to_llm_messages(context, registry=registry, jinja_env=jinja_env)

    assert f"известная роль: {role.name_ru}" in messages[1]["content"]


def test_to_llm_messages_includes_role_count(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """Рендер analyze содержит состав партии из GameState."""
    state = make_game_state(registry=registry)
    state.apply(
        AliveListEvent(
            ts=BASE_TS,
            game_id=1,
            payload=AliveListPayload(
                alive_player_names=["Вася", "Петя"],
                role_count={"Мафия": 5, "Мирный Житель": 5},
                total_players=10,
            ),
        )
    )
    context = build_context(game_state=state, registry=registry)

    messages = to_llm_messages(context, registry=registry, jinja_env=jinja_env)

    user_content = messages[1]["content"]
    assert "Мафия" in user_content
    assert "5" in user_content
    assert "{{" not in user_content


def test_to_llm_messages_renders_predictions_and_hints(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """Прошлые предсказания и сводка эвристик попадают в user-сообщение."""
    context = build_context(
        game_state=make_game_state(2, registry=registry),
        registry=registry,
        previous_predictions=[
            PredictionSnapshot(player_id=101, suspected_role="Мафия", confidence=0.65)
        ],
    )

    messages = to_llm_messages(
        context,
        registry=registry,
        jinja_env=jinja_env,
        club_notes_summary="early_lynch_no_argument",
    )

    user_content = messages[1]["content"]
    assert "Прошлые предсказания" in user_content
    assert "Мафия" in user_content
    assert "Сводка эвристик" in user_content
    assert "early_lynch_no_argument" in user_content


def test_to_llm_messages_with_club_patterns(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """Custom club patterns reach the user prompt without changing the system message."""
    context = build_context(game_state=make_game_state(registry=registry), registry=registry)

    messages = to_llm_messages(
        context, registry=registry, jinja_env=jinja_env, club_patterns="TEST"
    )
    empty_messages = to_llm_messages(context, registry=registry, jinja_env=jinja_env)

    assert "Паттерны этого клуба" in messages[1]["content"]
    assert "TEST" in messages[1]["content"]
    assert "TEST" not in messages[0]["content"]
    assert "Паттерны этого клуба" not in empty_messages[1]["content"]
    assert "Голосования (последние" not in empty_messages[1]["content"]


def test_analyze_jinja_renders_votes_and_patterns(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """All three club sections are present when their data is available."""
    state = make_game_state(2, registry=registry)
    for index, target_name in enumerate(("Петя", None)):
        state.apply(
            VoteEvent(
                ts=BASE_TS + timedelta(seconds=index),
                game_id=1,
                payload=VotePayload(
                    voter_id=100,
                    voter_name="Вася",
                    target_id=101 if target_name else None,
                    target_name=target_name,
                ),
            )
        )
    context = build_context(game_state=state, registry=registry)

    messages = to_llm_messages(
        context, registry=registry, jinja_env=jinja_env, club_patterns="Иллюстративный паттерн"
    )

    prompt = messages[1]["content"]
    assert "Голосования (последние 2)" in prompt
    assert "Вася → Петя" in prompt
    assert "Вася → (воздержался)" in prompt
    assert "Тонкости ролей" in prompt
    assert "Стреляет после проверки" in prompt
    assert "Паттерны этого клуба" in prompt
    assert "Иллюстративный паттерн" in prompt


def test_to_llm_messages_accepts_explicit_schema(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """Переданная схема заменяет встроенную по умолчанию."""
    context = build_context(game_state=make_game_state(1, registry=registry), registry=registry)

    messages = to_llm_messages(
        context,
        registry=registry,
        jinja_env=jinja_env,
        response_schema_json='{"title": "CustomSchema"}',
    )

    assert "CustomSchema" in messages[0]["content"]
    assert "PlayerPrediction" not in messages[0]["content"]


def test_to_llm_messages_uses_schema_global(registry: RoleRegistry) -> None:
    """Схему можно подложить через globals окружения Jinja."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("infra/llm/prompts"),
        autoescape=False,
    )
    env.globals["response_schema_json"] = '{"title": "GlobalSchema"}'
    context = build_context(game_state=make_game_state(1, registry=registry), registry=registry)

    messages = to_llm_messages(context, registry=registry, jinja_env=env)

    assert "GlobalSchema" in messages[0]["content"]


def test_to_llm_messages_rejects_unrendered_markers(registry: RoleRegistry) -> None:
    """Оставшиеся маркеры Jinja — понятная ошибка, а не «сырой» промпт."""
    env = jinja2.Environment(
        loader=DictLoader(
            {
                "system.jinja": "{% raw %}{{{% endraw %}",
                "analyze.jinja": "ok",
            }
        ),
        autoescape=False,
    )
    context = build_context(game_state=make_game_state(1, registry=registry), registry=registry)

    with pytest.raises(ValueError, match="Jinja marker"):
        to_llm_messages(context, registry=registry, jinja_env=env)


def test_default_response_schema_matches_contract() -> None:
    """Встроенная схема совпадает со схемой AnalysisResponse из infra."""
    from infra.llm.schemas import AnalysisResponse

    assert json.loads(DEFAULT_RESPONSE_SCHEMA_JSON) == AnalysisResponse.model_json_schema()


# --- revealed_roles (Part C) ----------------------------------------------


def test_build_context_revealed_roles(registry: RoleRegistry) -> None:
    """C. Только раскрытые роли попадают в revealed_roles и сохраняют порядок."""
    state = make_game_state(2, registry=registry)
    revealed_role = registry.all_roles()[0]
    state.apply(
        RoleRevealEvent(
            ts=BASE_TS,
            game_id=1,
            payload=RoleRevealPayload(
                player_id=player_id(0),
                player_name=player_name(0),
                role_name_ru=revealed_role.name_ru,
            ),
        )
    )

    context = build_context(game_state=state, registry=registry)

    assert len(context.revealed_roles) == 1
    revealed = context.revealed_roles[0]
    assert revealed.player_id == player_id(0)
    assert revealed.player_name == player_name(0)
    assert revealed.role_name_ru == revealed_role.name_ru
    assert revealed.is_winner is None
    # Оба игрока остаются в players: блок раскрытых ролей его не заменяет.
    assert len(context.players) == 2


def test_analyze_jinja_renders_revealed(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """C. Рендер analyze содержит блок раскрытых ролей перед списком игроков."""
    state = make_game_state(2, registry=registry)
    role = registry.all_roles()[0]
    state.apply(
        RoleRevealEvent(
            ts=BASE_TS,
            game_id=1,
            payload=RoleRevealPayload(
                player_id=player_id(0),
                player_name=player_name(0),
                role_name_ru=role.name_ru,
            ),
        )
    )
    context = build_context(game_state=state, registry=registry)

    messages = to_llm_messages(context, registry=registry, jinja_env=jinja_env)
    content = messages[1]["content"]

    assert "Раскрытые роли" in content
    assert "ground truth" in content
    assert f"player_id={player_id(0)} ({player_name(0)}): {role.name_ru}" in content
    assert "Не меняй их на" in content
    # Блок стоит до списка игроков — модель видит его первым.
    assert content.index("Раскрытые роли") < content.index("Игроки")


# ===================================================================
# Bot composition → explicit slots, all alive players, full distribution
# ===================================================================


def _apply_club_snapshot(state: GameState, names: list[str]) -> None:
    state.apply(
        AliveListEvent(
            ts=BASE_TS,
            game_id=state.game_id,
            payload=AliveListPayload(
                alive_player_names=names,
                role_count={"Мафия": 1, "Мирный Житель": 2},
                total_players=3,
            ),
        )
    )


def test_build_context_exposes_exact_role_slots(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """14 counters are not enough: the model needs an explicit sum."""
    state = make_game_state(3, registry=registry)
    _apply_club_snapshot(state, ["Вася", "Петя", "| user_4"])

    context = build_context(game_state=state, registry=registry)

    assert context.role_count == {"Мафия": 1, "Мирный житель": 2}
    assert context.role_slots == 3
    assert context.role_kinds == 2
    assert context.total_players == 3
    assert context.composition_matches_total is True
    assert context.alive_list_names == ["Вася", "Петя", "| user_4"]
    # Placeholder for a silent player is addressable by the LLM too.
    assert len(context.players) == 4
    assert any(player.is_placeholder for player in context.players)
    assert context.placeholder_player_names == ["| user_4"]

    content = to_llm_messages(context, registry=registry, jinja_env=jinja_env)[1]["content"]
    assert "Всего ролевых слотов (сумма счётчиков): 3" in content
    assert "Разных ролей: 2" in content
    assert "- Мафия — 1" in content
    assert "- Мирный житель — 2" in content
    assert "Живые игроки из сообщения бота (3)" in content
    assert "РОВНО 3 записей" in content
    assert "синтетический ID" in content
    assert "1 из 30" not in content
    assert "пропусти игрока" not in content.lower()


def test_role_notes_are_limited_to_composition(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """Notes for absent roles must not bury the composition for a 9B model."""
    state = make_game_state(3, registry=registry)
    _apply_club_snapshot(state, ["Вася", "Петя", "Маша"])

    context = build_context(game_state=state, registry=registry)
    content = to_llm_messages(context, registry=registry, jinja_env=jinja_env)[1]["content"]

    assert "Тонкости ролей" in content
    assert "Мафия: Базовой ночной активности нет" in content
    assert "Комиссар Каттани: Стреляет" not in content


def test_previous_predictions_render_as_anchor(
    registry: RoleRegistry, jinja_env: jinja2.Environment
) -> None:
    """Earlier roles are shown with an explicit no-flip-without-reason rule."""
    context = build_context(
        game_state=make_game_state(1, registry=registry),
        registry=registry,
        previous_predictions=[
            PredictionSnapshot(player_id=100, suspected_role="Мирный житель", confidence=0.42)
        ],
    )
    content = to_llm_messages(context, registry=registry, jinja_env=jinja_env)[1]["content"]

    assert "Прошлые предсказания (якорь)" in content
    assert "ID 100: Мирный житель" in content
    assert "НЕ меняй роль игрока без НОВОГО аргумента" in content
