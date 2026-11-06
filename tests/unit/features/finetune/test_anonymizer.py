"""Tests for player alias collection and context/payload anonymisation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from domain.events import (
    AliveListEvent,
    AliveListPayload,
    AnyEvent,
    MessageEvent,
    MessagePayload,
    PhaseChangeEvent,
    PhaseChangePayload,
    VoteEvent,
    VotePayload,
)
from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import load_roles
from features.analyzer.context_builder import (
    PredictionSnapshot,
    build_context,
)
from features.finetune.anonymizer import (
    PlayerAliasMap,
    anonymize_context,
    anonymize_payload,
)

ROLES_PATH = Path(__file__).parents[4] / "configs" / "roles.yaml"
BASE_TS = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_registry():
    return load_roles(ROLES_PATH)


def make_game_state() -> GameState:
    state = GameState(game_id=1, registry=make_registry())
    state.register_player(101, "Вася")
    state.register_player(102, "Петя")
    state.apply(
        AliveListEvent(
            ts=BASE_TS,
            game_id=1,
            payload=AliveListPayload(
                alive_player_names=["Вася", "Петя"],
                role_count={"Мафия": 1, "Мирный житель": 1},
                total_players=2,
            ),
        )
    )
    state.apply(
        PhaseChangeEvent(
            ts=BASE_TS,
            game_id=1,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
            ),
        )
    )
    state.apply(
        MessageEvent(
            ts=BASE_TS,
            game_id=1,
            payload=MessagePayload(author_id=101, author_name="Вася", text="Привет"),
        )
    )
    state.apply(
        VoteEvent(
            ts=BASE_TS,
            game_id=1,
            payload=VotePayload(voter_id=101, voter_name="Вася", target_id=102, target_name="Петя"),
        )
    )
    return state


def test_alias_map_stable_per_id() -> None:
    """The same id always receives the same alias."""
    aliases = PlayerAliasMap()
    assert aliases.alias_for_id(101) == "player_1"
    assert aliases.alias_for_id(101) == "player_1"
    assert aliases.alias_for_id(102) == "player_2"


def test_alias_map_none_passthrough() -> None:
    """None ids/names pass through unchanged."""
    aliases = PlayerAliasMap()
    assert aliases.alias_for_id(None) is None
    assert aliases.alias_for_name(1, None) is None


def test_alias_map_remap_id() -> None:
    """Remapped ids are stable positive integers."""
    aliases = PlayerAliasMap()
    assert aliases.remap_id(101) == 1
    assert aliases.remap_id(102) == 2
    assert aliases.remap_id(None) is None


def test_alias_map_collect_events() -> None:
    """Event payload identities are indexed."""
    aliases = PlayerAliasMap()
    events: list[AnyEvent] = [
        MessageEvent(
            ts=BASE_TS,
            game_id=1,
            payload=MessagePayload(author_id=101, author_name="Вася", text="x"),
        ),
        VoteEvent(
            ts=BASE_TS,
            game_id=1,
            payload=VotePayload(voter_id=101, voter_name="Вася", target_id=102, target_name="Петя"),
        ),
    ]
    aliases.collect_events(events)
    assert aliases.alias_for_name(1, "Вася") == "player_1"
    assert aliases.alias_for_name(1, "Петя") == "player_2"


def test_alias_map_ambiguous_names_get_distinct_aliases() -> None:
    """Two players sharing a display name never merge."""
    aliases = PlayerAliasMap()
    events: list[AnyEvent] = [
        MessageEvent(
            ts=BASE_TS,
            game_id=1,
            payload=MessagePayload(author_id=101, author_name="Одинаковое", text="x"),
        ),
        MessageEvent(
            ts=BASE_TS,
            game_id=1,
            payload=MessagePayload(author_id=102, author_name="Одинаковое", text="y"),
        ),
    ]
    aliases.collect_events(events)
    first = aliases.alias_for_name(1, "Одинаковое", 101)
    second = aliases.alias_for_name(1, "Одинаковое", 102)
    assert first == "player_1"
    assert second == "player_2"
    assert first != second


def test_anonymize_context_structured_fields() -> None:
    """Player ids, names and anchor ids are replaced."""
    aliases = PlayerAliasMap()
    aliases.collect_events(make_game_state().events())
    aliases.collect_player_id(101)
    aliases.collect_player_id(102)

    context = build_context(
        game_state=make_game_state(),
        registry=make_registry(),
        previous_predictions=[
            PredictionSnapshot(player_id=101, suspected_role="Мафия", confidence=0.8)
        ],
    )
    anonymized = anonymize_context(context, aliases)

    assert anonymized.game_id == 1
    assert all(p.player_id > 0 for p in anonymized.players)
    assert {p.player_id for p in anonymized.players} == {1, 2}
    names = {p.display_name for p in anonymized.players}
    assert names == {"player_1", "player_2"}
    assert all(m.author_name.startswith("player_") for m in anonymized.recent_messages)
    assert all(v.voter_name.startswith("player_") for v in anonymized.recent_votes)
    assert all(
        v.target_name.startswith("player_") for v in anonymized.recent_votes if v.target_name
    )
    assert anonymized.previous_predictions[0].player_id == 1


def test_anonymize_payload() -> None:
    """Assistant JSON player ids and names are remapped."""
    aliases = PlayerAliasMap()
    aliases.collect_player_id(101)
    payload = {"analysis": [{"player_id": 101, "player_name": "Вася", "suspected_role": "Мафия"}]}
    anonymized = anonymize_payload(payload, aliases)
    item = anonymized["analysis"][0]
    assert item["player_id"] == 1
    assert item["player_name"] == "player_1"
    assert item["suspected_role"] == "Мафия"


def test_anonymize_context_keeps_free_text() -> None:
    """Message bodies are left untouched (structured-only policy)."""
    aliases = PlayerAliasMap()
    aliases.collect_events(make_game_state().events())
    context = build_context(game_state=make_game_state(), registry=make_registry())
    anonymized = anonymize_context(context, aliases)
    assert anonymized.recent_messages[0].text == "Привет"
