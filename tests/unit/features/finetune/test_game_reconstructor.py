"""Tests for tolerant GameState reconstruction from persisted events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

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
from domain.exceptions import UnknownRoleError
from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import load_roles
from features.finetune.game_reconstructor import reconstruct_game_state

ROLES_PATH = Path(__file__).parents[4] / "configs" / "roles.yaml"
BASE_TS = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_registry():
    return load_roles(ROLES_PATH)


def make_game_state(game_id: int = 1):
    return GameState(game_id=game_id, registry=make_registry())


def test_reconstruct_applies_all_events() -> None:
    """Every valid event lands in the reconstructed state."""
    state = make_game_state()
    state.register_player(101, "Вася")
    state.register_player(102, "Петя")
    events = [
        AliveListEvent(
            ts=BASE_TS,
            game_id=1,
            payload=AliveListPayload(alive_player_names=["user_101", "user_102"]),
        ),
        MessageEvent(
            ts=BASE_TS + timedelta(minutes=1),
            game_id=1,
            payload=MessagePayload(author_id=101, author_name="Вася", text="Привет"),
        ),
        VoteEvent(
            ts=BASE_TS + timedelta(minutes=2),
            game_id=1,
            payload=VotePayload(voter_id=101, voter_name="Вася", target_id=102, target_name="Петя"),
        ),
        DeathEvent(
            ts=BASE_TS + timedelta(minutes=3),
            game_id=1,
            payload=DeathPayload(victim_id=102, victim_name="Петя", cause="lynch"),
        ),
        RoleRevealEvent(
            ts=BASE_TS + timedelta(minutes=4),
            game_id=1,
            payload=RoleRevealPayload(player_id=101, player_name="Вася", role_name_ru="Мафия"),
        ),
    ]
    reconstructed = reconstruct_game_state(events=events, game_id=1, registry=make_registry())

    assert reconstructed.game_id == 1
    assert len(reconstructed.events()) == 5
    player = reconstructed.player(101)
    assert player.known_role is not None
    assert player.known_role.name_ru == "Мафия"
    assert reconstructed.player(102).alive is False


def test_reconstruct_registers_alive_list_placeholders() -> None:
    """Alive-list names become placeholder players in the reconstructed state."""
    events = [
        AliveListEvent(
            ts=BASE_TS,
            game_id=1,
            payload=AliveListPayload(
                alive_player_names=["Скрытый игрок"],
                role_count={"Мафия": 1, "Мирный житель": 2},
                total_players=3,
            ),
        ),
    ]
    reconstructed = reconstruct_game_state(events=events, game_id=1, registry=make_registry())

    players = reconstructed.players()
    assert len(players) == 1
    assert players[0].display_name == "Скрытый игрок"
    assert players[0].player_id < 0
    assert reconstructed.role_count == {"Мафия": 1, "Мирный житель": 2}


def test_reconstruct_skips_broken_phase_transitions() -> None:
    """A phase event that fails validation is skipped, not fatal."""
    events = [
        PhaseChangeEvent(
            ts=BASE_TS,
            game_id=1,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
            ),
        ),
        PhaseChangeEvent(
            ts=BASE_TS + timedelta(minutes=1),
            game_id=1,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION, to_phase=Phase.DAY, day_number=1
            ),
        ),
        MessageEvent(
            ts=BASE_TS + timedelta(minutes=2),
            game_id=1,
            payload=MessagePayload(author_id=101, author_name="Вася", text="Привет"),
        ),
    ]
    reconstructed = reconstruct_game_state(events=events, game_id=1, registry=make_registry())

    assert reconstructed.current_phase() == Phase.NIGHT
    assert len(reconstructed.events()) == 2


def test_reconstruct_skips_unknown_role_reveals() -> None:
    """A reveal with an unregistered role name is skipped without aborting."""
    events = [
        AliveListEvent(
            ts=BASE_TS,
            game_id=1,
            payload=AliveListPayload(alive_player_names=["user_101"]),
        ),
        RoleRevealEvent(
            ts=BASE_TS + timedelta(minutes=1),
            game_id=1,
            payload=RoleRevealPayload(
                player_id=101, player_name="Вася", role_name_ru="Несуществующая роль"
            ),
        ),
        MessageEvent(
            ts=BASE_TS + timedelta(minutes=2),
            game_id=1,
            payload=MessagePayload(author_id=101, author_name="Вася", text="Привет"),
        ),
    ]
    reconstructed = reconstruct_game_state(events=events, game_id=1, registry=make_registry())

    assert len(reconstructed.events()) == 2
    assert reconstructed.player(101).known_role is None


def test_reconstruct_empty_events() -> None:
    """An empty event list yields a clean registration-phase state."""
    reconstructed = reconstruct_game_state(events=[], game_id=7, registry=make_registry())

    assert reconstructed.game_id == 7
    assert reconstructed.players() == ()
    assert reconstructed.current_phase() == Phase.REGISTRATION


def test_reconstruct_does_not_raise_on_unknown_role() -> None:
    """UnknownRoleError from apply() is swallowed by the tolerant loop."""
    events = [
        RoleRevealEvent(
            ts=BASE_TS,
            game_id=1,
            payload=RoleRevealPayload(player_id=101, player_name="Вася", role_name_ru="Тест-роль"),
        ),
    ]
    try:
        reconstruct_game_state(events=events, game_id=1, registry=make_registry())
    except UnknownRoleError:  # pragma: no cover — must not leak
        raise AssertionError("reconstruction must not raise UnknownRoleError")
