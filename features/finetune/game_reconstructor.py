"""Rebuild a :class:`GameState` from persisted events at a prediction moment.

The reconstruction is deliberately tolerant: backfilled games may contain
phase transitions that no longer validate (the live aggregate enforced them
strictly, the raw event log does not). A failed event is skipped with a
WARNING instead of aborting the whole example — the dataset pipeline must
survive imperfect history, and the skipped event is a known limitation of
the source data, not a bug in the builder.
"""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

from domain.events import AnyEvent
from domain.exceptions import ConfigError, UnknownRoleError
from domain.game_state import GameState
from domain.roles import RoleRegistry

LOG_CHANNEL: str = "finetune.reconstruct"


def reconstruct_game_state(
    *,
    events: Sequence[AnyEvent],
    game_id: int,
    registry: RoleRegistry,
) -> GameState:
    """Apply ``events`` to a fresh :class:`GameState` in order.

    Args:
        events: domain events for one game, chronological (insertion order).
        game_id: identifier the reconstructed state will carry.
        registry: role catalogue used by ``GameState``.

    Returns:
        A state holding every successfully applied event.
    """
    state = GameState(game_id=game_id, registry=registry)
    skipped = 0
    for event in events:
        try:
            state.apply(event)
        except (ConfigError, UnknownRoleError) as error:
            skipped += 1
            logger.bind(channel=LOG_CHANNEL).warning(
                "reconstruct: skipped {} event (ts={}, game_id={}): {}",
                event.kind,
                event.ts,
                event.game_id,
                error,
            )
    if skipped:
        logger.bind(channel=LOG_CHANNEL).info(
            "reconstruct: game_id={} applied={} skipped={}",
            game_id,
            len(events) - skipped,
            skipped,
        )
    return state
