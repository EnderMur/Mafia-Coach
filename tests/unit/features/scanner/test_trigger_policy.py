"""Tests for ``features.scanner.trigger_policy`` — pure flush-decision logic.

No ``async``, no timers, no I/O: the policy only answers questions about
events and buffer sizes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from domain.events import (
    AliveListEvent,
    AliveListPayload,
    AnyEvent,
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
from domain.phases import Phase
from features.scanner.trigger_policy import TriggerMode, TriggerPolicy

TS = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)
GAME_ID = 1


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def _msg_event() -> MessageEvent:
    """A plain player message — the only kind that never forces a flush."""
    return MessageEvent(
        ts=TS,
        game_id=GAME_ID,
        payload=MessagePayload(author_id=100, author_name="Вася", text="я мирный"),
    )


def _death_event() -> DeathEvent:
    """A night kill event."""
    return DeathEvent(
        ts=TS,
        game_id=GAME_ID,
        payload=DeathPayload(victim_id=100, victim_name="Вася", cause="night_kill"),
    )


def _vote_event() -> VoteEvent:
    """A single vote event."""
    return VoteEvent(
        ts=TS,
        game_id=GAME_ID,
        payload=VotePayload(voter_id=100, voter_name="Вася", target_id=101, target_name="Петя"),
    )


def _phase_event() -> PhaseChangeEvent:
    """A night → day transition event."""
    return PhaseChangeEvent(
        ts=TS,
        game_id=GAME_ID,
        payload=PhaseChangePayload(from_phase=Phase.NIGHT, to_phase=Phase.DAY, day_number=2),
    )


def _role_reveal_event() -> RoleRevealEvent:
    """A public role reveal event."""
    return RoleRevealEvent(
        ts=TS,
        game_id=GAME_ID,
        payload=RoleRevealPayload(player_id=100, player_name="Вася", role_name_ru="Мафия"),
    )


def _alive_list_event() -> AliveListEvent:
    """A bot snapshot with alive nicknames and role composition."""
    return AliveListEvent(
        ts=TS,
        game_id=GAME_ID,
        payload=AliveListPayload(
            alive_player_names=["Вася", "Петя"],
            role_count={"Мафия": 1, "Мирный житель": 1},
            total_players=2,
        ),
    )


def _policy(mode: TriggerMode, **kwargs: Any) -> TriggerPolicy:
    """Build a policy for one mode with optional field overrides."""
    return TriggerPolicy(mode=mode, **kwargs)


# ===================================================================
# should_force_now (1-8)
# ===================================================================


def test_every_message_forces_on_plain_message() -> None:
    """1. ``every_message`` analyzes literally every event."""
    assert _policy(TriggerMode.EVERY_MESSAGE).should_force_now(_msg_event()) is True


def test_every_message_forces_on_death() -> None:
    """2. ``every_message`` also forces on phase events."""
    policy = _policy(TriggerMode.EVERY_MESSAGE)

    assert policy.should_force_now(_death_event()) is True
    assert policy.should_force_now(_phase_event()) is True
    assert policy.should_force_now(_vote_event()) is True
    assert policy.should_force_now(_role_reveal_event()) is True


def test_queue_never_forces() -> None:
    """3. ``queue`` leaves flushing to the orchestrator, even for a death."""
    policy = _policy(TriggerMode.QUEUE)

    assert policy.should_force_now(_death_event()) is False
    assert policy.should_force_now(_phase_event()) is False
    assert policy.should_force_now(_vote_event()) is False
    assert policy.should_force_now(_role_reveal_event()) is False
    assert policy.should_force_now(_alive_list_event()) is False
    assert policy.should_force_now(_msg_event()) is False


def test_batch_debounce_forces_on_death() -> None:
    """4. A death bypasses the debounce timer."""
    assert _policy(TriggerMode.BATCH_DEBOUNCE).should_force_now(_death_event()) is True


def test_batch_debounce_forces_on_phase_change() -> None:
    """5. A phase change bypasses the debounce timer."""
    assert _policy(TriggerMode.BATCH_DEBOUNCE).should_force_now(_phase_event()) is True


def test_batch_debounce_forces_on_vote() -> None:
    """6. A vote bypasses the debounce timer."""
    assert _policy(TriggerMode.BATCH_DEBOUNCE).should_force_now(_vote_event()) is True


def test_batch_debounce_does_not_force_on_message() -> None:
    """7. Plain messages accumulate until the timer or the size limit fires."""
    assert _policy(TriggerMode.BATCH_DEBOUNCE).should_force_now(_msg_event()) is False


def test_batch_debounce_forces_on_role_reveal() -> None:
    """8. A role reveal bypasses the debounce timer."""
    assert _policy(TriggerMode.BATCH_DEBOUNCE).should_force_now(_role_reveal_event()) is True


def test_batch_debounce_forces_on_alive_list() -> None:
    """8a. The bot's alive/role snapshot must not wait for the debounce timer."""
    policy = _policy(TriggerMode.BATCH_DEBOUNCE)
    assert policy.should_force_now(_alive_list_event()) is True
    assert _policy(TriggerMode.QUEUE).should_force_now(_alive_list_event()) is False


def test_force_flush_covers_all_event_kinds() -> None:
    """8b. Every non-message kind forces a flush; a message never does."""
    policy = _policy(TriggerMode.BATCH_DEBOUNCE)
    game_state_events: list[AnyEvent] = [
        _alive_list_event(),
        _death_event(),
        _vote_event(),
        _phase_event(),
        _role_reveal_event(),
    ]

    assert all(policy.should_force_now(event) is True for event in game_state_events)
    assert policy.should_force_now(_msg_event()) is False


# ===================================================================
# should_force_on_size (9-11)
# ===================================================================


def test_should_force_on_size_below_limit() -> None:
    """9. Nine buffered messages out of ten are not enough."""
    policy = _policy(TriggerMode.BATCH_DEBOUNCE, batch_max_messages=10)

    assert policy.should_force_on_size(9) is False


def test_should_force_on_size_at_limit() -> None:
    """10. Exactly ten messages fill the batch."""
    policy = _policy(TriggerMode.BATCH_DEBOUNCE, batch_max_messages=10)

    assert policy.should_force_on_size(10) is True


def test_should_force_on_size_above_limit() -> None:
    """11. An over-full buffer must also be flushed."""
    policy = _policy(TriggerMode.BATCH_DEBOUNCE, batch_max_messages=10)

    assert policy.should_force_on_size(11) is True


@pytest.mark.parametrize("mode", list(TriggerMode))
def test_should_force_on_size_is_mode_independent(mode: TriggerMode) -> None:
    """11b. The size limit is a plain comparison in every mode."""
    policy = _policy(mode, batch_max_messages=2)

    assert policy.should_force_on_size(0) is False
    assert policy.should_force_on_size(1) is False
    assert policy.should_force_on_size(2) is True
    assert policy.should_force_on_size(3) is True


# ===================================================================
# Mode predicates (12)
# ===================================================================


def test_is_batch_true_only_for_batch_debounce() -> None:
    """12. ``is_batch`` is exclusive to ``batch_debounce``."""
    assert _policy(TriggerMode.BATCH_DEBOUNCE).is_batch() is True
    assert _policy(TriggerMode.EVERY_MESSAGE).is_batch() is False
    assert _policy(TriggerMode.QUEUE).is_batch() is False


def test_is_every_message_true_only_for_every_message() -> None:
    """12b. ``is_every_message`` is exclusive to ``every_message``."""
    assert _policy(TriggerMode.EVERY_MESSAGE).is_every_message() is True
    assert _policy(TriggerMode.BATCH_DEBOUNCE).is_every_message() is False
    assert _policy(TriggerMode.QUEUE).is_every_message() is False


def test_is_queue_true_only_for_queue() -> None:
    """12c. ``is_queue`` is exclusive to ``queue``."""
    assert _policy(TriggerMode.QUEUE).is_queue() is True
    assert _policy(TriggerMode.BATCH_DEBOUNCE).is_queue() is False
    assert _policy(TriggerMode.EVERY_MESSAGE).is_queue() is False


def test_exactly_one_predicate_is_true_per_mode() -> None:
    """12d. The three predicates partition the modes."""
    for mode in TriggerMode:
        policy = _policy(mode)
        flags = [policy.is_batch(), policy.is_every_message(), policy.is_queue()]

        assert sum(flags) == 1


# ===================================================================
# Validation (13-15)
# ===================================================================


def test_policy_is_frozen() -> None:
    """13. The model is immutable: mutation raises ``ValidationError``."""
    policy = _policy(TriggerMode.BATCH_DEBOUNCE)

    with pytest.raises(ValidationError):
        policy.mode = TriggerMode.QUEUE
    with pytest.raises(ValidationError):
        policy.batch_max_messages = 1

    assert policy.mode is TriggerMode.BATCH_DEBOUNCE
    assert policy.batch_max_messages == 10


@pytest.mark.parametrize("value", [-0.1, -1.0, -100.0])
def test_negative_debounce_seconds_rejected(value: float) -> None:
    """14. ``batch_debounce_sec`` must be >= 0."""
    with pytest.raises(ValidationError):
        _policy(TriggerMode.BATCH_DEBOUNCE, batch_debounce_sec=value)


@pytest.mark.parametrize("value", [0, -1, -10])
def test_non_positive_max_messages_rejected(value: int) -> None:
    """15. ``batch_max_messages`` must be >= 1."""
    with pytest.raises(ValidationError):
        _policy(TriggerMode.BATCH_DEBOUNCE, batch_max_messages=value)


def test_zero_debounce_seconds_allowed() -> None:
    """15b. Zero delay is a valid (if aggressive) configuration."""
    policy = _policy(TriggerMode.BATCH_DEBOUNCE, batch_debounce_sec=0.0)

    assert policy.batch_debounce_sec == 0.0


def test_defaults_match_settings_example() -> None:
    """Defaults mirror ``configs/settings.example.yaml`` (README §3.1)."""
    policy = _policy(TriggerMode.BATCH_DEBOUNCE)

    assert policy.batch_debounce_sec == 3.0
    assert policy.batch_max_messages == 10


def test_extra_fields_are_forbidden() -> None:
    """Unknown configuration keys are rejected instead of ignored."""
    with pytest.raises(ValidationError):
        TriggerPolicy(mode=TriggerMode.QUEUE, unknown_field=1)  # type: ignore[call-arg]


def test_mode_accepts_settings_string() -> None:
    """The mode can be built from the raw YAML string."""
    policy = TriggerPolicy(mode="batch_debounce")  # type: ignore[arg-type]

    assert policy.mode is TriggerMode.BATCH_DEBOUNCE
    assert policy.is_batch() is True


def test_trigger_mode_values_match_documented_strings() -> None:
    """Mode values are exactly the strings documented in README §3.1."""
    assert TriggerMode.EVERY_MESSAGE.value == "every_message"
    assert TriggerMode.QUEUE.value == "queue"
    assert TriggerMode.BATCH_DEBOUNCE.value == "batch_debounce"
