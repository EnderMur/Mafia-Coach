"""Build one SFT example per prediction from a reconstructed game state.

The builder is pure: it receives an already reconstructed :class:`GameState`,
a :class:`PredictionRow`, and the dependencies of ``context_builder``
(registry, jinja environment, club patterns). It reuses
:func:`features.analyzer.context_builder.build_context` and
:func:`~features.analyzer.context_builder.to_llm_messages` so the rendered
prompt is byte-for-byte the same format the live analyzer produces.

Assistant answer policy:

- correct prediction — the model's own JSON (role, confidence, reason,
  reasons, patterns, reasoning) is kept as the teaching target;
- wrong prediction with a known final role — the role is replaced by the
  ground truth, patterns are cleared, and the model's reasoning is dropped
  (it led to the wrong answer; keeping it would teach a contradiction).
"""

from __future__ import annotations

import json
from typing import Any

import jinja2
from loguru import logger

from domain.game_state import GameState
from domain.roles import RoleRegistry
from features.analyzer.context_builder import (
    LLMContext,
    PredictionSnapshot,
    build_context,
    to_llm_messages,
)
from features.analyzer.heuristic_hints import HeuristicHint, compute_hints
from features.finetune.anonymizer import PlayerAliasMap, anonymize_context, anonymize_payload
from features.finetune.models import DatasetRecord, Message, PredictionRow

LOG_CHANNEL: str = "finetune.builder"

#: Confidence for a corrected (ground-truth) assistant answer.
_CORRECTED_CONFIDENCE: float = 0.9

#: Reason used when the prediction was wrong but the final role is known.
_CORRECTED_REASON: str = "Роль из финального отчёта бота."


def build_example(
    *,
    game_state: GameState,
    prediction: PredictionRow,
    previous_snapshots: list[PredictionSnapshot],
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    club_patterns: list[dict[str, object]] | None = None,
    final_role: str,
    correct: bool,
    source: str,
    max_messages: int = 200,
    aliases: PlayerAliasMap | None = None,
) -> DatasetRecord | None:
    """Render one ``[system, user, assistant]`` example for a prediction.

    Args:
        game_state: reconstructed aggregate at ``prediction.ts``.
        prediction: the prediction row being exported.
        previous_snapshots: earlier predictions of the same game (anchor).
        registry: role catalogue.
        jinja_env: environment loaded from ``infra/llm/prompts``.
        club_patterns: structured club patterns for the prompt, or None.
        final_role: canonical ground-truth role (``name_ru``).
        correct: whether ``prediction`` matched ``final_role``.
        source: ``"current"`` or ``"history"`` — recorded in metadata.
        max_messages: size of the message window, as in the live analyzer.
        aliases: optional alias map; when set, the context is anonymized
            before rendering and the assistant/metadata ids are remapped.

    Returns:
        A :class:`DatasetRecord`, or ``None`` when the predicted player is
        absent from the reconstructed state (the example would be broken).
    """
    players_by_id = {player.player_id: player for player in game_state.players()}
    player = players_by_id.get(prediction.player_id)
    if player is None:
        logger.bind(channel=LOG_CHANNEL).warning(
            "builder: player {} not in reconstructed game {} — example skipped",
            prediction.player_id,
            prediction.game_id,
        )
        return None

    hints = compute_hints(game_state=game_state, registry=registry)
    context = build_context(
        game_state=game_state,
        registry=registry,
        max_messages=max_messages,
        previous_predictions=previous_snapshots,
    )

    player_id = prediction.player_id
    original_display_name = player.display_name
    if aliases is not None:
        context = anonymize_context(context, aliases)
        remapped_id = aliases.remap_id(prediction.player_id)
        if remapped_id is None:  # pragma: no cover — id is never None here
            raise ValueError(f"cannot anonymize player {prediction.player_id}")
        player_id = remapped_id

    messages = to_llm_messages(
        context,
        registry=registry,
        jinja_env=jinja_env,
        club_notes_summary=_render_hints(hints),
        club_patterns=club_patterns,
    )

    assistant_payload = _assistant_payload(
        context=context,
        prediction=prediction,
        player_id=prediction.player_id,
        player_name=original_display_name,
        final_role=final_role,
        correct=correct,
        aliases=aliases,
    )
    assistant_content = json.dumps(assistant_payload, ensure_ascii=False)

    record_messages = [
        Message(role="system", content=messages[0]["content"]),
        Message(role="user", content=messages[1]["content"]),
        Message(role="assistant", content=assistant_content),
    ]
    return DatasetRecord(
        messages=record_messages,
        game_id=prediction.game_id,
        prediction_id=prediction.id,
        player_id=player_id,
        ts=prediction.ts,
        source=source,  # type: ignore[arg-type]
        correct=correct,
        ground_truth_role=final_role,
        predicted_role=prediction.predicted_role,
        confidence=prediction.confidence,
    )


def _assistant_payload(
    *,
    context: LLMContext,
    prediction: PredictionRow,
    player_id: int,
    player_name: str,
    final_role: str,
    correct: bool,
    aliases: PlayerAliasMap | None,
) -> dict[str, Any]:
    """Build the correct JSON response for one player."""
    if correct:
        suspected_role = prediction.predicted_role or final_role
        confidence = prediction.confidence
        reason = prediction.reason or _CORRECTED_REASON
        reasons = _split_reasons(reason)
        pattern_ids = list(prediction.pattern_ids)
        pattern_descriptions = list(prediction.pattern_descriptions)
        reasoning = prediction.reasoning
    else:
        suspected_role = final_role
        confidence = _CORRECTED_CONFIDENCE
        reason = _CORRECTED_REASON
        reasons = [reason]
        pattern_ids = []
        pattern_descriptions = []
        reasoning = None
    payload: dict[str, Any] = {
        "analysis": [
            {
                "player_id": player_id,
                "player_name": player_name,
                "suspected_role": suspected_role,
                "confidence": confidence,
                "reason": reason,
                "alternative": None,
                "reasons": reasons,
                "pattern_ids": pattern_ids,
                "pattern_descriptions": pattern_descriptions,
            }
        ],
        "game_read": _synthesize_game_read(context, player_name, suspected_role),
        "reasoning": reasoning,
    }
    if aliases is not None:
        payload = anonymize_payload(payload, aliases)
    return payload


def _synthesize_game_read(context: LLMContext, player_name: str, suspected_role: str) -> str:
    """Build a short game-read sentence for the assistant JSON.

    The original ``game_read`` is not persisted in SQLite; a factual
    one-liner keeps the response schema-valid and informative.
    """
    return (
        f"Партия {context.game_id}, фаза {context.phase.current_phase.value}, "
        f"день {context.phase.current_day}. Игрок {player_name}: {suspected_role}."
    )


def _split_reasons(reason: str) -> list[str]:
    """Turn a stored per-player reason into the ``reasons`` list (1–5 items)."""
    parts = [part.strip() for part in reason.splitlines() if part.strip()]
    if not parts:
        parts = [_CORRECTED_REASON]
    return parts[:5]


def _render_hints(hints: list[HeuristicHint]) -> str:
    """Render hints as the bullet list used by ``analyze.jinja``."""
    if not hints:
        return ""
    return "\n".join(f"- [{hint.weight:+.2f}] {hint.description}" for hint in hints)
