"""End-of-game retrospective: compare predictions with the real final roles.

The module never touches SQL directly — every read and write goes through the
passed repositories — and it is fully synchronous. The orchestrator decides
whether to run it in a worker thread.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from domain.roles import RoleRegistry
from infra.db.repositories.game_players import GamePlayersRepository
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.predictions import PredictionsRepository

LOG_CHANNEL: str = "retro"


class RoleAccuracy(BaseModel):
    """Per-role accuracy stats."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(ge=0)
    correct: int = Field(ge=0)


class GameRetroReport(BaseModel):
    """End-of-game retrospective."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    game_id: int
    total_predictions: int = Field(ge=0)
    correct: int = Field(ge=0)
    wrong: int = Field(ge=0)
    unknown: int = Field(ge=0)  # predictions с no final_role
    by_role: dict[str, RoleAccuracy]


def retro_analyze(
    *,
    game_id: int,
    registry: RoleRegistry,
    predictions_repo: PredictionsRepository,
    game_players_repo: GamePlayersRepository,
    games_repo: GamesRepository | None = None,
) -> GameRetroReport:
    """Compare predictions with final_role for a finished game.

    1. Fetch all predictions for ``game_id``.
    2. For each prediction take ``game_players.final_role`` of the player. A
       missing row or a NULL role marks the prediction "unknown": its
       ``predictions.correct`` stays NULL. Otherwise correctness is
       ``predicted_role == final_role``, it is persisted through
       ``mark_correct`` and aggregated into ``by_role[predicted_role]``.
    3. Return :class:`GameRetroReport`.

    Args:
        game_id: Identifier of the finished game.
        registry: Role registry; reserved for alias-aware comparisons and kept
            in the signature so callers do not change when that lands.
        predictions_repo: Repository holding the model predictions.
        game_players_repo: Repository holding per-game participation rows.
        games_repo: Optional games repository. When supplied, the game must
            exist, otherwise a :class:`~domain.exceptions.ConfigError` is
            raised. The predictions and game_players repositories alone cannot
            distinguish a missing game from a game without rows.

    Returns:
        The immutable retrospective report.

    Raises:
        ConfigError: If ``games_repo`` is supplied and the game is absent.
    """
    del registry  # reserved: alias-aware comparison is not part of step 8c

    if games_repo is not None:
        games_repo.get_game(game_id)

    predictions = predictions_repo.list_predictions(game_id)

    correct = 0
    wrong = 0
    unknown = 0
    by_role: dict[str, RoleAccuracy] = {}

    for prediction in predictions:
        player_id = int(prediction["player_id"])
        predicted_role = str(prediction["predicted_role"])
        final_role = _final_role_of(game_players_repo, game_id=game_id, player_id=player_id)

        if final_role is None:
            unknown += 1
            continue

        is_correct = predicted_role == final_role
        predictions_repo.mark_correct(prediction_id=int(prediction["id"]), correct=is_correct)

        if is_correct:
            correct += 1
        else:
            wrong += 1

        previous = by_role.get(predicted_role, RoleAccuracy(total=0, correct=0))
        by_role[predicted_role] = RoleAccuracy(
            total=previous.total + 1,
            correct=previous.correct + (1 if is_correct else 0),
        )

    report = GameRetroReport(
        game_id=game_id,
        total_predictions=len(predictions),
        correct=correct,
        wrong=wrong,
        unknown=unknown,
        by_role=by_role,
    )
    logger.bind(channel=LOG_CHANNEL).info(
        "retro_analyze: game_id={} total={} correct={} wrong={} unknown={}",
        report.game_id,
        report.total_predictions,
        report.correct,
        report.wrong,
        report.unknown,
    )
    if report.unknown > 0:
        logger.bind(channel=LOG_CHANNEL).info(
            "retro: {} predictions lack final_role (unknown) — run backfill or check game_players",
            report.unknown,
        )
    return report


def _final_role_of(
    game_players_repo: GamePlayersRepository, *, game_id: int, player_id: int
) -> str | None:
    """Return the stored final role of a player, or None when it is unknown."""
    row = game_players_repo.get_player_in_game(game_id=game_id, player_id=player_id)
    if row is None:
        return None
    final_role = row["final_role"]
    if final_role is None:
        return None
    return str(final_role)
