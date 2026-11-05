"""Tests for the mutable game state aggregate."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from loguru import logger

from domain.events import (
    AliveListEvent,
    AliveListPayload,
    DeathEvent,
    DeathPayload,
    EndGameEvent,
    EndGamePayload,
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
from domain.game_state import GameState, _placeholder_player_id
from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles

if TYPE_CHECKING:
    from loguru import Record

ROLES_PATH = Path(__file__).parents[3] / "configs" / "roles.yaml"


@pytest.fixture
def registry() -> RoleRegistry:
    """Load the real configured roles for reveal resolution."""
    return load_roles(ROLES_PATH)


@pytest.fixture
def game(registry: RoleRegistry) -> GameState:
    """Create an empty game state with the real role registry."""
    return GameState(game_id=1, registry=registry)


@pytest.fixture
def log_records() -> Iterator[list[Record]]:
    """Capture WARNING-and-above loguru records emitted during a test."""
    records: list[Record] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def event_time() -> datetime:
    """Return a timezone-aware timestamp for event fixtures."""
    return datetime(2025, 1, 1, tzinfo=UTC)


def test_game_id_property(registry: RoleRegistry) -> None:
    """The game id is available through a public read-only property."""
    game = GameState(game_id=42, registry=registry)

    assert game.game_id == 42


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


def make_alive_list(
    names: list[str],
    *,
    role_count: dict[str, int] | None = None,
    total_players: int | None = None,
    game_id: int = 1,
) -> AliveListEvent:
    """Build a bot alive-list snapshot event."""
    return AliveListEvent(
        ts=event_time(),
        game_id=game_id,
        payload=AliveListPayload(
            alive_player_names=names,
            role_count=role_count,
            total_players=total_players,
        ),
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


def test_register_real_replaces_placeholder(game: GameState, registry: RoleRegistry) -> None:
    """A real id collapses the same-named placeholder and inherits its ground truth."""
    placeholder_id = _placeholder_player_id("Вася")
    assert placeholder_id < 0
    game.register_player(placeholder_id, "Вася")
    game.apply(make_end_game(winners=[("Вася", "Комиссар Каттани")]))
    placeholder = game.player(placeholder_id)
    assert placeholder.final_role is not None
    assert placeholder.is_winner is True

    real = game.register_player(456, "Вася")

    assert [player.player_id for player in game.players()] == [456]
    assert real.display_name == "Вася"
    assert real.final_role == registry.resolve_alias("Комиссар Каттани")
    assert real.final_role is not None
    assert real.revealed is True
    assert real.is_winner is True


def test_register_real_no_placeholder_unchanged(game: GameState) -> None:
    """Without a placeholder the real registration is a plain single player."""
    real = game.register_player(456, "Вася")

    assert game.players() == (real,)
    assert real.player_id == 456
    assert real.known_role is None
    assert real.is_winner is None


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


def test_day_count_follows_night_day_vote_cycles(game: GameState) -> None:
    """Регистрация — 0, затем ночь, день и голосование принадлежат одному циклу."""
    assert game.current_day() == 0
    for source, target, expected in (
        (Phase.REGISTRATION, Phase.NIGHT, 1),
        (Phase.NIGHT, Phase.DAY, 1),
        (Phase.DAY, Phase.VOTE, 1),
        (Phase.VOTE, Phase.NIGHT, 2),
        (Phase.NIGHT, Phase.DAY, 2),
        (Phase.DAY, Phase.VOTE, 2),
    ):
        game.apply(make_phase_change(from_phase=source, to_phase=target, day_number=0))
        assert game.current_day() == expected


def test_second_night_day_vote_are_day_two(game: GameState) -> None:
    game.apply(make_phase_change(to_phase=Phase.VOTE, day_number=1))
    for source, target in (
        (Phase.VOTE, Phase.NIGHT),
        (Phase.NIGHT, Phase.DAY),
        (Phase.DAY, Phase.VOTE),
    ):
        game.apply(make_phase_change(from_phase=source, to_phase=target, day_number=0))
        assert game.current_day() == 2


def test_explicit_night_number_is_not_incremented(game: GameState) -> None:
    game.apply(make_phase_change(to_phase=Phase.NIGHT, day_number=7))
    assert game.current_day() == 7


def test_registration_resets_day_to_zero(game: GameState) -> None:
    game.apply(make_phase_change(to_phase=Phase.NIGHT, day_number=5))
    game.apply(make_phase_change(from_phase=Phase.NIGHT, to_phase=Phase.REGISTRATION, day_number=0))
    assert game.current_day() == 0


def test_explicit_day_is_used_for_night_and_vote(game: GameState) -> None:
    game.apply(make_phase_change(to_phase=Phase.NIGHT, day_number=3))
    assert game.current_day() == 3
    game.apply(make_phase_change(from_phase=Phase.NIGHT, to_phase=Phase.VOTE, day_number=7))
    assert game.current_day() == 7


def test_phase_change_with_explicit_day_number(game: GameState) -> None:
    """Явный номер дня из бота применяется как есть."""
    game.apply(make_phase_change(to_phase=Phase.DAY, day_number=5))
    assert game.current_day() == 5


def test_phase_change_to_end_does_not_increment(game: GameState) -> None:
    """Переход в END без номера дня не считается новым днём."""
    game.apply(make_phase_change(to_phase=Phase.END, day_number=0))
    assert game.current_phase() == Phase.END
    assert game.current_day() == 0


def test_phase_change_with_wrong_source_phase_is_atomic(game: GameState) -> None:
    """A transition must begin at the aggregate's current phase."""
    with pytest.raises(ConfigError, match="current phase"):
        game.apply(make_phase_change(from_phase=Phase.DAY, to_phase=Phase.VOTE))
    assert game.current_phase() == Phase.REGISTRATION
    assert game.events() == ()


def test_phase_change_to_same_phase_is_rejected(game: GameState) -> None:
    """A phase-change event must represent an actual transition outside game start."""
    game.apply(make_phase_change(to_phase=Phase.DAY, day_number=1))
    with pytest.raises(ConfigError, match="different phase"):
        game.apply(make_phase_change(from_phase=Phase.DAY, to_phase=Phase.DAY))
    assert len(game.events()) == 1


def test_new_game_after_end_does_not_mutate_old_state(game: GameState) -> None:
    game.register_player(1, "Игрок")
    game.apply(make_phase_change(to_phase=Phase.END, day_number=0))
    new_game = game.reset_for_new_game(2)
    assert game.current_phase() is Phase.END
    assert len(game.players()) == 1
    assert new_game.current_phase() is Phase.REGISTRATION
    assert new_game.players() == ()


def test_game_start_when_registration_no_players(game: GameState) -> None:
    """A start marker is accepted even when REGISTRATION is already current."""
    start_event = make_phase_change(
        from_phase=Phase.REGISTRATION,
        to_phase=Phase.REGISTRATION,
        day_number=0,
    )

    game.apply(start_event)

    assert game.current_phase() is Phase.REGISTRATION
    assert game.players() == ()
    assert game.events() == (start_event,)


def test_new_game_drops_previous_players(game: GameState) -> None:
    game.register_player(1, "Игрок")
    game.apply(make_phase_change(to_phase=Phase.NIGHT, day_number=2))
    new_game = game.reset_for_new_game(2)
    assert new_game.players() == ()
    assert new_game.current_day() == 0
    assert game.player(1).display_name == "Игрок"


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


# ===================================================================
# Step 10c-fix-8: AliveListEvent (A2-1 + bot snapshot)
# ===================================================================


def test_apply_alive_list_marks_absent_dead(game: GameState) -> None:
    """Every registered player missing from the snapshot is dead (A2-1)."""
    game.register_player(1, "A")
    game.register_player(2, "B")
    game.register_player(3, "C")

    event = make_alive_list(["A"])
    game.apply(event)

    assert game.player(1).alive is True
    assert game.player(2).alive is False
    assert game.player(3).alive is False
    assert game.alive_players() == (game.player(1),)
    assert game.dead_players() == (game.player(2), game.player(3))
    # The event itself is kept in the history, like every other event.
    assert game.events() == (event,)
    assert game.events_by_kind("alive_list") == (event,)

    # Re-applying a snapshot is idempotent: already-dead players stay dead.
    game.apply(make_alive_list(["A"]))
    assert game.player(1).alive is True
    assert game.player(2).alive is False
    assert game.player(3).alive is False


def test_apply_alive_list_sets_role_count(game: GameState) -> None:
    """The bot-reported role composition is stored and exposed as a copy."""
    game.register_player(1, "A")
    role_count = {"Мафия": 5, "Дон": 1, "Мирный Житель": 4}

    game.apply(make_alive_list(["A"], role_count=role_count, total_players=10))

    assert game.role_count == role_count
    assert game.role_count is not role_count


def test_apply_alive_list_sets_total_players(game: GameState) -> None:
    """The bot-reported headcount is stored verbatim."""
    game.register_player(1, "A")

    game.apply(make_alive_list(["A"], total_players=10))

    assert game.total_players == 10
    assert game.role_count is None


def test_alive_list_registers_placeholder_players(game: GameState, registry: RoleRegistry) -> None:
    """New snapshot names get stable negative ids."""
    game.register_player(1, "A")
    game.register_player(2, "B")
    game.register_player(3, "C")

    game.apply(make_alive_list(["A", "D"]))

    assert len(game.players()) == 4
    assert {player.display_name for player in game.players()} == {"A", "B", "C", "D"}
    assert {player.player_id for player in game.players()} == {1, 2, 3, _placeholder_player_id("D")}
    assert game.player(1).alive is True
    assert game.player(2).alive is False
    assert game.player(3).alive is False

    # Defaults before any snapshot arrives.
    fresh = GameState(game_id=1, registry=registry)
    assert fresh.role_count is None
    assert fresh.total_players is None


# ===================================================================
# Step 10c-fix-11: END_GAME → final_role / is_winner (7-10)
# ===================================================================


def make_end_game(
    *,
    winners: list[tuple[str, str]] | None = None,
    losers: list[tuple[str, str]] | None = None,
    duration_minutes: int | None = 20,
    game_id: int = 1,
) -> EndGameEvent:
    """Build the bot's final report event."""
    return EndGameEvent(
        ts=event_time(),
        game_id=game_id,
        payload=EndGamePayload(
            winners=winners or [],
            losers=losers or [],
            duration_minutes=duration_minutes,
        ),
    )


def test_apply_end_game_sets_final_role(game: GameState) -> None:
    """7. Winners and losers get final_role/is_winner; unmentioned players stay None."""
    game.register_player(1, "A")
    game.register_player(2, "B")
    game.register_player(3, "C")

    game.apply(
        make_end_game(
            winners=[("A", "Мафия")],
            losers=[("B", "Мирный Житель")],
        )
    )

    player_a = game.player(1)
    player_b = game.player(2)
    player_c = game.player(3)
    assert player_a.final_role is not None and player_a.final_role.id == "mafia"
    assert player_b.final_role is not None and player_b.final_role.id == "civilian"
    assert player_c.final_role is None
    assert player_a.is_winner is True
    assert player_b.is_winner is False
    assert player_c.is_winner is None


def test_phase_change_after_end_logged_at_debug(game: GameState, log_records: list[Record]) -> None:
    """8. END_GAME sets Phase.END; later phase chatter is ignored at DEBUG."""
    game.register_player(1, "A")
    game.apply(make_phase_change(from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT))
    assert game.current_phase() is Phase.NIGHT

    game.apply(make_end_game(winners=[("A", "Мафия")]))

    assert game.current_phase() is Phase.END
    assert game.events()[-1].kind == "end_game"

    debug_records: list[Record] = []
    debug_sink = logger.add(lambda message: debug_records.append(message.record), level="DEBUG")
    try:
        game.apply(make_phase_change(from_phase=Phase.END, to_phase=Phase.DAY))
    finally:
        logger.remove(debug_sink)

    # 11b-fix-3: repeated post-END phase changes are noise, not WARNINGs.
    assert game.current_phase() is Phase.END
    ignored = [
        record
        for record in debug_records
        if "phase_change to" in record["message"] and "after END" in record["message"]
    ]
    assert len(ignored) == 1
    assert ignored[0]["level"].name == "DEBUG"
    assert not any("phase_change to" in record["message"] for record in log_records)


def test_apply_end_game_warns_on_role_mismatch(game: GameState, log_records: list[Record]) -> None:
    """9. A reveal that contradicts the final report warns but END_GAME wins (B2-3)."""
    game.register_player(1, "A")
    game.apply(
        RoleRevealEvent(
            ts=event_time(),
            game_id=1,
            payload=RoleRevealPayload(
                player_id=1,
                player_name="A",
                role_name_ru="Мирный житель",
            ),
        )
    )
    assert game.player(1).known_role is not None

    game.apply(make_end_game(winners=[("A", "Мафия")]))

    assert any("final_role mismatch for A" in record["message"] for record in log_records)
    final_role = game.player(1).final_role
    assert final_role is not None and final_role.id == "mafia"
    assert game.player(1).is_winner is True


def test_apply_end_game_resolves_name_ru(game: GameState) -> None:
    """10. The bot's Russian role name (any capitalisation) resolves to a Role."""
    game.register_player(1, "A")

    game.apply(make_end_game(losers=[("A", "Мирный Житель")]))

    final_role = game.player(1).final_role
    assert final_role is not None
    assert final_role.id == "civilian"
    assert game.player(1).is_winner is False


def test_apply_end_game_matching_reveal_does_not_warn(
    game: GameState, log_records: list[Record]
) -> None:
    """A reveal that agrees with the final report is updated silently."""
    game.register_player(1, "A")
    game.apply(
        RoleRevealEvent(
            ts=event_time(),
            game_id=1,
            payload=RoleRevealPayload(player_id=1, player_name="A", role_name_ru="Мафия"),
        )
    )

    game.apply(make_end_game(winners=[("A", "Мафия")]))

    assert not any("final_role mismatch" in record["message"] for record in log_records)
    assert game.player(1).is_winner is True


def test_apply_end_game_unknown_role_keeps_final_role_empty(
    game: GameState, log_records: list[Record]
) -> None:
    """An unresolvable role name warns but still records the winner flag."""
    game.register_player(1, "A")

    game.apply(make_end_game(winners=[("A", "Несуществующая роль")]))

    assert any("unknown final role name" in record["message"] for record in log_records)
    assert game.player(1).final_role is None
    assert game.player(1).is_winner is True


def test_phase_end_guard(game: GameState) -> None:
    game.apply(
        PhaseChangeEvent(
            ts=event_time(),
            game_id=1,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION,
                to_phase=Phase.END,
                day_number=0,
            ),
        )
    )
    game.apply(
        PhaseChangeEvent(
            ts=event_time(),
            game_id=1,
            payload=PhaseChangePayload(
                from_phase=Phase.END,
                to_phase=Phase.DAY,
                day_number=1,
            ),
        )
    )
    assert game.current_phase() == Phase.END
    assert game.current_day() == 0
    assert len(game.events()) == 1


def test_placeholder_id_is_stable_negative_int64() -> None:
    pid = _placeholder_player_id("Never Wrote")
    assert -(2**63) < pid < 0
    assert pid == _placeholder_player_id("Never Wrote")
    assert pid != _placeholder_player_id("Another Player")


def test_last_alive_list_names_whitelist(game: GameState) -> None:
    """E. Последний ALIVE_LIST задаёт whitelist реальных игроков."""
    assert game.last_alive_list_names == frozenset()

    game.apply(make_alive_list(["A", "B"]))

    assert game.last_alive_list_names == frozenset({"A", "B"})


def test_last_alive_list_names_reset_on_new_game(game: GameState) -> None:
    """Новая партия не наследует whitelist живых игроков."""
    game.apply(make_alive_list(["A", "B"]))
    new_game = game.reset_for_new_game(2)
    assert new_game.last_alive_list_names == frozenset()
    assert game.last_alive_list_names == frozenset({"A", "B"})


def test_alive_list_matches_decorated_nicknames(game: GameState) -> None:
    """Club markers/emoji must not create duplicate placeholders or kill a player."""
    game.register_player(1, "Дмитрий")
    game.register_player(2, "Shinichiro")
    game.register_player(3, "Петя")

    game.apply(
        make_alive_list(
            ["| Дмитрий", "🕷️ᵗᵏ| Shinichiro🌝🖤"],
            total_players=2,
        )
    )

    assert len(game.players()) == 3
    assert game.player(1).alive is True
    assert game.player(2).alive is True
    assert game.player(3).alive is False
    assert game.last_alive_list_names == frozenset({"| Дмитрий", "🕷️ᵗᵏ| Shinichiro🌝🖤"})


def test_end_game_matches_decorated_display_name(game: GameState) -> None:
    game.register_player(123, "Дмитрий")
    game.apply(make_end_game(winners=[("| Дмитрий", "Мафия")]))
    assert len(game.players()) == 1
    role = game.player(123).final_role
    assert role is not None
    assert role.name_ru == "Мафия"


def test_end_game_matches_decorated_username(game: GameState) -> None:
    game.register_player(123, "Другое имя", username="Дмитрий")
    game.apply(make_end_game(losers=[("| Дмитрий", "Мирный житель")]))
    assert len(game.players()) == 1
    role = game.player(123).final_role
    assert role is not None
    assert role.name_ru == "Мирный житель"


def test_reset_for_new_game_clears_all_state(game: GameState) -> None:
    game.register_name("Вася", 123)
    game.mark_registered(123)
    game.register_player(123, "Вася")
    game.apply(make_phase_change(to_phase=Phase.NIGHT, day_number=0))
    new_game = game.reset_for_new_game(2)
    assert new_game.game_id == 2
    assert not new_game.players() and not new_game.events()
    assert new_game.resolve_name("Вася") is None
    assert not new_game.registered_ids
    assert new_game.current_day() == 0 and new_game.current_phase() is Phase.REGISTRATION


def test_late_end_game_warns_and_does_not_change_state(
    game: GameState, log_records: list[Record]
) -> None:
    event = EndGameEvent(
        ts=event_time(), game_id=2, payload=EndGamePayload(winners=[("А", "Мафия")])
    )
    with pytest.raises(ConfigError):
        game.apply(event)
    assert game.current_phase() is Phase.REGISTRATION
    assert game.events() == ()
    assert any("event game_id=2, current game_id=1" in record["message"] for record in log_records)
