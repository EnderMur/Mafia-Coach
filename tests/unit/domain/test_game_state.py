"""Tests for the mutable game state aggregate."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from domain.events import (
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
from domain.exceptions import ConfigError, UnknownRoleError
from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles

ROLES_PATH = Path(__file__).parents[3] / "configs" / "roles.yaml"


@pytest.fixture
def registry() -> RoleRegistry:
    """Load the real configured roles for reveal resolution."""
    return load_roles(ROLES_PATH)


@pytest.fixture
def game(registry: RoleRegistry) -> GameState:
    """Create an empty game state with the real role registry."""
    return GameState(game_id=1, registry=registry)


def event_time() -> datetime:
    """Return a timezone-aware timestamp for event fixtures."""
    return datetime(2025, 1, 1, tzinfo=UTC)


def make_death(player_id: int = 1, game_id: int = 1) -> DeathEvent:
    """Build a death event for a registered player."""
    return DeathEvent(
        ts=event_time(),
        game_id=game_id,
        payload=DeathPayload(victim_id=player_id, victim_name="Player"),
    )


def make_phase_change(
    from_phase: Phase = Phase.REGISTRATION,
    to_phase: Phase = Phase.NIGHT,
    day_number: int = 1,
    game_id: int = 1,
) -> PhaseChangeEvent:
    """Build a phase-change event."""
    return PhaseChangeEvent(
        ts=event_time(),
        game_id=game_id,
        payload=PhaseChangePayload(from_phase=from_phase, to_phase=to_phase, day_number=day_number),
    )


def test_register_player_adds_living_player(game: GameState) -> None:
    """New players are registered alive and returned."""
    player = game.register_player(1, "Player One", "one")
    assert player.alive is True
    assert game.players() == (player,)
    assert game.alive_players() == (player,)


def test_register_player_updates_existing_player_without_duplicate(
    game: GameState,
) -> None:
    """Registering an existing id refreshes its public details."""
    game.register_player(1, "Old Name", "old")
    updated = game.register_player(1, "New Name", "new")
    assert game.players() == (updated,)
    assert updated.display_name == "New Name"
    assert updated.username == "new"


def test_register_player_preserves_existing_game_state(game: GameState) -> None:
    """Refreshing a player does not restore a dead player or erase a reveal."""
    game.register_player(1, "Player")
    game.apply(make_death())
    refreshed = game.register_player(1, "Updated")
    assert refreshed.alive is False
    assert game.dead_players() == (refreshed,)


def test_death_event_marks_player_dead(game: GameState) -> None:
    """Applying a death event moves its player to the dead collection."""
    game.register_player(1, "Player")
    game.apply(make_death())
    assert game.player(1).alive is False
    assert game.alive_players() == ()
    assert game.dead_players() == (game.player(1),)


def test_repeated_death_is_rejected_without_appending(game: GameState) -> None:
    """A repeated death raises and leaves the event log unchanged."""
    game.register_player(1, "Player")
    game.apply(make_death())
    with pytest.raises(ConfigError, match="already dead"):
        game.apply(make_death())
    assert len(game.events()) == 1


def test_role_reveal_resolves_alias_and_marks_player(registry: RoleRegistry) -> None:
    """A bot alias resolves to a role and publicly reveals the player."""
    game = GameState(game_id=1, registry=registry)
    player = game.register_player(1, "Player")
    role = registry.all_roles()[0]
    game.apply(
        RoleRevealEvent(
            ts=event_time(),
            game_id=1,
            payload=RoleRevealPayload(
                player_id=1,
                player_name="Player",
                role_name_ru=role.bot_aliases[0],
            ),
        )
    )
    assert game.player(1).known_role == role
    assert game.player(1).revealed is True
    assert player.revealed is False


def test_unknown_role_reveal_is_atomic(registry: RoleRegistry) -> None:
    """An unresolved role name raises without changing player or event state."""
    game = GameState(game_id=1, registry=registry)
    game.register_player(1, "Player")
    event = RoleRevealEvent(
        ts=event_time(),
        game_id=1,
        payload=RoleRevealPayload(
            player_id=1, player_name="Player", role_name_ru="Unrecognized role"
        ),
    )
    with pytest.raises(UnknownRoleError):
        game.apply(event)
    assert game.player(1).revealed is False
    assert game.player(1).known_role is None
    assert game.events() == ()


def test_phase_change_updates_phase_and_day(game: GameState) -> None:
    """A valid phase transition updates the current phase and day number."""
    game.apply(make_phase_change(to_phase=Phase.NIGHT, day_number=1))
    assert game.current_phase() == Phase.NIGHT
    assert game.current_day() == 1


def test_phase_change_with_wrong_source_phase_is_atomic(game: GameState) -> None:
    """A transition must begin at the aggregate's current phase."""
    with pytest.raises(ConfigError, match="current phase"):
        game.apply(make_phase_change(from_phase=Phase.DAY, to_phase=Phase.VOTE))
    assert game.current_phase() == Phase.REGISTRATION
    assert game.events() == ()


def test_phase_change_to_same_phase_is_rejected(game: GameState) -> None:
    """A phase-change event must represent an actual transition."""
    with pytest.raises(ConfigError, match="different phase"):
        game.apply(make_phase_change(from_phase=Phase.REGISTRATION, to_phase=Phase.REGISTRATION))
    assert game.events() == ()


def test_message_event_is_recorded_without_changing_players(game: GameState) -> None:
    """Messages are kept in the event log but do not alter game state."""
    game.register_player(1, "Player")
    before = game.players()
    event = MessageEvent(
        ts=event_time(),
        game_id=1,
        payload=MessagePayload(author_id=1, author_name="Player", text="Hello"),
    )
    game.apply(event)
    assert game.players() == before
    assert game.events() == (event,)


def test_votes_are_recorded_without_changing_game_state(game: GameState) -> None:
    """Votes are stored for later aggregation without mutating players."""
    game.register_player(1, "Player")
    event = VoteEvent(
        ts=event_time(),
        game_id=1,
        payload=VotePayload(voter_id=1, voter_name="Player", target_id=None, target_name=None),
    )
    game.apply(event)
    assert game.events() == (event,)
    assert game.player(1).alive is True


def test_events_preserve_append_order_and_filter_by_kind(game: GameState) -> None:
    """The event log preserves append order and filters by kind."""
    game.register_player(1, "Player")
    message = MessageEvent(
        ts=event_time(),
        game_id=1,
        payload=MessagePayload(author_id=1, author_name="Player", text="Hello"),
    )
    death = make_death()
    game.apply(message)
    game.apply(death)
    assert game.events() == (message, death)
    assert game.events_by_kind("death") == (death,)
    assert game.events_by_kind("missing") == ()


def test_unknown_player_raises_config_error(game: GameState) -> None:
    """Player lookup reports unregistered ids as configuration errors."""
    with pytest.raises(ConfigError, match="player 404 not registered"):
        game.player(404)


def test_death_of_unknown_player_is_not_recorded(game: GameState) -> None:
    """A death for an unregistered player is rejected atomically."""
    with pytest.raises(ConfigError, match="not registered"):
        game.apply(make_death(player_id=404))
    assert game.events() == ()


def test_event_from_another_game_is_rejected(game: GameState) -> None:
    """Events from a different game cannot enter this aggregate."""
    with pytest.raises(ConfigError, match="belongs to game"):
        game.apply(make_phase_change(game_id=2))
    assert game.events() == ()
