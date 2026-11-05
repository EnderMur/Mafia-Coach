"""Orchestrate one game-analysis cycle across context, LLM, and persistence."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Final

from jinja2 import Environment
from loguru import logger
from pydantic import ValidationError

from domain.club_patterns import ClubPatterns
from domain.exceptions import LLMConnectionError, LLMError
from domain.game_state import GameState
from domain.roles import RoleRegistry
from features.analyzer.context_builder import (
    LLMContext,
    PredictionSnapshot,
    build_context,
    to_llm_messages,
)
from features.analyzer.heuristic_hints import HeuristicHint, compute_hints
from features.analyzer.pattern_detector import PatternDetector
from infra.db.repositories.predictions import PredictionsRepository
from infra.llm.client import LLMClient
from infra.llm.schemas import AnalysisResponse, PlayerPrediction

DEFAULT_REASONING_MAX_CHARS: Final[int] = 4096

#: Reasons that carry no new evidence for a role change (behavioural emptiness).
_WEAK_REASONS: Final[frozenset[str]] = frozenset(
    {
        "молчит",
        "не участвует",
        "не пишет",
        "нет данных",
        "пассивен",
        "не активен",
        "не ясно",
        "неизвестно",
        "не знаю",
    }
)

#: Confidence for a slot assigned deterministically from the bot composition.
_COMPOSITION_FILL_CONFIDENCE: Final[float] = 0.3


def _partial_reasoning(max_chars: int, chunks: list[str]) -> str:
    """Return the reasoning streamed before a failure, clipped to ``max_chars``.

    Preserve partial reasoning even if the structured response never arrived.

    Args:
        max_chars: configured maximum, ``llm.reasoning_max_chars``.
        chunks: reasoning fragments collected so far.

    Returns:
        The clipped text, or ``""`` when the model streamed nothing.
    """
    return _truncate_reasoning("".join(chunks), max_chars) or ""


def _has_new_evidence(reasons: list[str]) -> bool:
    """Return whether a role change is backed by a concrete new argument.

    A reason like ``"сохранена прежняя роль …"`` is continuity bookkeeping, not
    evidence: it must not justify a *future* unexplained flip.
    """
    for reason in reasons:
        normalized = reason.strip().casefold()
        if normalized in _WEAK_REASONS or normalized.startswith("сохранена прежняя роль"):
            continue
        if len(normalized) >= 15:
            return True
    return False


def _truncate_reasoning(reasoning_full: str, max_chars: int) -> str:
    """Clip streamed reasoning to ``max_chars`` and record the omission.

    Args:
        reasoning_full: everything the model streamed before the response ended
            (or before the request failed).
        max_chars: configured maximum, ``llm.reasoning_max_chars``.

    Returns:
        The clipped text, with a ``[truncated, total N]`` marker when something
        was cut. Empty input yields an empty string.
    """
    truncated = reasoning_full[:max_chars]
    if len(reasoning_full) > max_chars:
        truncated += f"\n[truncated, total {len(reasoning_full)}]"
    return truncated


@dataclass(frozen=True)
class AnalysisResult:
    """Outcome of one analyzer cycle."""

    response: AnalysisResponse | None
    error: str | None
    duration_sec: float
    # elapsed wall time of one analyze() cycle, clamped to >= 0.
    # Clamped because Windows QPC (perf_counter) can roll back on
    # virtualized CI runners (vCPU migration) — negative duration
    # has no semantic meaning.
    reasoning_chars: int
    warnings: tuple[str, ...] = ()
    reasoning_text: str = ""

    @property
    def success(self) -> bool:
        """Return whether an analysis response was produced without an error."""
        return self.response is not None and self.error is None


class GameAnalyzer:
    """High-level analyzer tying game state, the LLM, and prediction storage.

    Assumes a single-threaded asyncio caller (orchestrator in step 10). This
    object is not thread-safe.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        predictions_repo: PredictionsRepository,
        registry: RoleRegistry,
        jinja_env: Environment,
        max_messages: int = 200,
        reasoning_max_chars: int = DEFAULT_REASONING_MAX_CHARS,
        club_patterns: list[dict[str, object]] | None = None,
        patterns: ClubPatterns | None = None,
    ) -> None:
        """Retain the dependencies and context/reasoning limits for analysis.

        Args:
            club_patterns: structured club patterns as list of dicts with keys
                id, trigger, conclusion_role, confidence_tier,
                requires_confirmation, description. Passed to the Jinja
                prompt templates.
            patterns: validated club patterns from ``configs/club_patterns.yaml``.
                When provided, pattern-based role overrides are applied before
                composition enforcement.
        """
        self._llm_client = llm_client
        self._predictions_repo = predictions_repo
        self._registry = registry
        self._jinja_env = jinja_env
        self._max_messages = max_messages
        self._reasoning_max_chars = reasoning_max_chars
        self._club_patterns = club_patterns
        if self._club_patterns is None and patterns is not None:
            self._club_patterns = [
                {
                    "id": p.id,
                    "trigger": p.trigger,
                    "conclusion_role": p.conclusion_role,
                    "confidence_tier": p.confidence_tier,
                    "requires_confirmation": p.requires_confirmation,
                    "description": p.description,
                }
                for p in patterns.patterns
            ]
        self._pattern_detector = (
            PatternDetector(patterns=patterns, registry=registry) if patterns is not None else None
        )

    async def analyze(
        self,
        game_state: GameState,
        *,
        previous_predictions: list[PredictionSnapshot] | None = None,
        persist: bool = True,
    ) -> AnalysisResult:
        """Build context, request and log reasoning, then optionally persist.

        Transient LLM errors are returned in :class:`AnalysisResult` rather than
        raised, so an analysis failure does not stop game processing. Prediction
        persistence failures are recorded as warnings while retaining the LLM
        response.

        When ``previous_predictions`` is omitted, the last persisted predictions
        are loaded from SQLite so the model sees its own earlier decisions
        (otherwise it "forgets" a role and flips it without a reason). A bot
        composition is enforced afterwards: quotas cap roles and missing slots
        are filled only when their total matches the current alive headcount.
        """
        start = time.perf_counter()
        if previous_predictions is None:
            previous_predictions = await self._load_previous_predictions(game_state.game_id)
        hints = compute_hints(game_state=game_state, registry=self._registry)
        context = build_context(
            game_state=game_state,
            registry=self._registry,
            max_messages=self._max_messages,
            previous_predictions=previous_predictions,
        )
        messages = to_llm_messages(
            context,
            registry=self._registry,
            jinja_env=self._jinja_env,
            club_notes_summary=self._render_hints(hints),
            club_patterns=self._club_patterns,
        )

        reasoning_buffer: list[str] = []

        def on_reasoning(chunk: str) -> None:
            reasoning_buffer.append(chunk)

        def resolve_role_alias(role_name: str) -> str | None:
            role = self._registry.resolve_alias(role_name)
            return role.name_ru if role is not None else None

        try:
            response = await self._llm_client.analyze_stream(
                messages=messages,
                known_player_ids=frozenset(player.player_id for player in context.players),
                role_names_ru=frozenset(role.name_ru for role in self._registry.all_roles()),
                role_alias_resolver=resolve_role_alias,
                on_reasoning_chunk=on_reasoning,
                log_reasoning_chunks=False,
                silent_player_ids=context.silent_player_ids,
            )
        except LLMConnectionError as error:
            logger.bind(channel="llm").error("analysis_failed: connection: {}", str(error))
            return AnalysisResult(
                response=None,
                error=f"connection: {error}",
                duration_sec=max(0.0, time.perf_counter() - start),
                reasoning_chars=sum(len(chunk) for chunk in reasoning_buffer),
                reasoning_text=_partial_reasoning(self._reasoning_max_chars, reasoning_buffer),
            )
        except ValidationError as error:
            llm_error = LLMError(f"invalid response: {error}")
            logger.bind(channel="llm").error("analysis_failed: {}", str(llm_error))
            return AnalysisResult(
                response=None,
                error=str(llm_error),
                duration_sec=max(0.0, time.perf_counter() - start),
                reasoning_chars=sum(len(chunk) for chunk in reasoning_buffer),
                reasoning_text=_partial_reasoning(self._reasoning_max_chars, reasoning_buffer),
            )
        except LLMError as error:
            logger.bind(channel="llm").error("analysis_failed: {}", str(error))
            return AnalysisResult(
                response=None,
                error=str(error),
                duration_sec=max(0.0, time.perf_counter() - start),
                reasoning_chars=sum(len(chunk) for chunk in reasoning_buffer),
                reasoning_text=_partial_reasoning(self._reasoning_max_chars, reasoning_buffer),
            )

        # Keep earlier decisions sticky and enforce the bot's exact role slots
        # even when the local model skipped players or exceeded a counter.
        live_predictions, live_warnings = self._filter_current_predictions(
            response.analysis, game_state
        )
        if live_warnings:
            response = response.model_copy(update={"analysis": live_predictions})

        # Молчащие игроки (без сообщений и без голосов) получают null-роль с
        # confidence 0.0 — им не назначаются роли из состава бота.
        silent_forced = self._force_silent_player_null_roles(
            response.analysis, game_state, context.silent_player_ids
        )
        if silent_forced != response.analysis:
            response = response.model_copy(update={"analysis": silent_forced})

        # Club-pattern overrides: pattern-based roles take priority over
        # bot-composition slots. Released slots are redistributed by
        # _enforce_composition inside _postprocess_response.
        if self._pattern_detector is not None:
            previous_roles_by_id = {p.player_id: p.suspected_role for p in response.analysis}
            name_by_id = {player.player_id: player.display_name for player in game_state.players()}
            pattern_matches, slot_releases = self._pattern_detector.detect(
                game_state=game_state,
                previous_predictions_role_by_id=previous_roles_by_id,
            )
            if pattern_matches or slot_releases:
                overridden = self._pattern_detector.apply_overrides(
                    predictions=response.analysis,
                    matches=pattern_matches,
                    releases=slot_releases,
                    name_by_id=name_by_id,
                )
                if overridden != response.analysis:
                    response = response.model_copy(update={"analysis": overridden})

        response, correction_warnings = self._postprocess_response(
            response,
            context=context,
            previous_predictions=previous_predictions,
            game_state=game_state,
        )

        reasoning_full = "".join(reasoning_buffer)
        reasoning_source = "stream"
        if not reasoning_full and response.reasoning:
            # Non-thinking models may return reasoning only in the JSON response.
            reasoning_full = response.reasoning
            reasoning_source = "json"

        reasoning_truncated = reasoning_full[: self._reasoning_max_chars]
        if len(reasoning_full) > self._reasoning_max_chars:
            reasoning_truncated += f"\n[truncated, total {len(reasoning_full)} chars]"
        if reasoning_truncated:
            logger.bind(channel="reasoning").info(
                "reasoning ({}) [{}]: {}",
                reasoning_source,
                len(reasoning_full),
                reasoning_truncated[:200],
            )

        warnings: list[str] = [*live_warnings, *correction_warnings]
        if persist:
            retained = []
            for prediction in response.analysis:
                current, dropped_warnings = self._filter_current_predictions(
                    [prediction], game_state
                )
                warnings.extend(dropped_warnings)
                if not current:
                    continue
                retained.append(prediction)
                try:
                    await asyncio.to_thread(
                        self._predictions_repo.upsert_prediction,
                        game_id=game_state.game_id,
                        player_id=prediction.player_id,
                        predicted_role=prediction.suspected_role,
                        confidence=prediction.confidence,
                        reasoning=response.reasoning or "\n".join(prediction.reasons),
                    )
                except Exception as error:  # noqa: BLE001 — persistence is defensive
                    warnings.append(f"persistence_failed: {error}")
                    logger.bind(channel="llm").error("persistence_failed: {}", str(error))
                    continue
                try:
                    await asyncio.to_thread(
                        self._predictions_repo.insert_history,
                        game_id=game_state.game_id,
                        player_id=prediction.player_id,
                        predicted_role=prediction.suspected_role,
                        confidence=prediction.confidence,
                        reasoning="\n".join(prediction.reasons),
                    )
                except Exception as error:  # noqa: BLE001 — history is optional
                    warnings.append(f"history_failed: {error}")
                    logger.bind(channel="llm").warning("history_failed: {}", str(error))

            if retained != response.analysis:
                response = response.model_copy(update={"analysis": retained})

        return AnalysisResult(
            response=response,
            error=None,
            duration_sec=max(0.0, time.perf_counter() - start),
            reasoning_chars=len(reasoning_full),
            warnings=tuple(warnings),
            reasoning_text=reasoning_truncated or "",
        )

    async def _load_previous_predictions(self, game_id: int) -> list[PredictionSnapshot]:
        """Read the last persisted predictions so the next cycle sees them.

        A DB failure must never abort analysis: an empty anchor list only means
        the model starts fresh, which is the previous (buggy) behaviour.
        """
        try:
            rows = await asyncio.to_thread(self._predictions_repo.list_predictions, game_id)
        except Exception as error:  # noqa: BLE001 — optional context enrichment
            logger.bind(channel="llm").debug("previous predictions unavailable: {}", error)
            return []

        snapshots: list[PredictionSnapshot] = []
        for row in rows:
            try:
                player_id = int(row["player_id"])
                role = row["predicted_role"]
                if role is None:
                    continue
                role = str(role)
                confidence = float(row["confidence"])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if not role:
                continue
            try:
                snapshots.append(
                    PredictionSnapshot(
                        player_id=player_id,
                        suspected_role=role,
                        confidence=min(1.0, max(0.0, confidence)),
                    )
                )
            except ValidationError:
                continue
        return snapshots

    @staticmethod
    def _filter_current_predictions(
        predictions: list[PlayerPrediction], game_state: GameState
    ) -> tuple[list[PlayerPrediction], list[str]]:
        """Drop predictions for players missing or dead in the current game state."""
        current = {player.player_id: player for player in game_state.players()}
        kept: list[PlayerPrediction] = []
        warnings: list[str] = []
        for prediction in predictions:
            player = current.get(prediction.player_id)
            if player is not None and player.alive:
                kept.append(prediction)
                continue
            status = "missing" if player is None else "dead"
            warning = f"prediction_dropped: player {prediction.player_id} is {status}"
            warnings.append(warning)
            logger.bind(channel="llm").warning("{}", warning)
        return kept, warnings

    def _force_silent_player_null_roles(
        self,
        predictions: list[PlayerPrediction],
        game_state: GameState,
        silent_player_ids: frozenset[int],
    ) -> list[PlayerPrediction]:
        """Force null-role predictions for silent players (no messages, no votes).

        A silent player gets ``suspected_role = null``, ``confidence = 0.0``,
        ``reason = "нет данных"`` and empty pattern lists. Existing predictions
        for silent players are overridden; missing ones are created.
        """
        if not silent_player_ids:
            return predictions

        name_by_id = {player.player_id: player.display_name for player in game_state.players()}
        prediction_ids = {prediction.player_id for prediction in predictions}

        result: list[PlayerPrediction] = []
        for prediction in predictions:
            if prediction.player_id in silent_player_ids:
                result.append(
                    prediction.model_copy(
                        update={
                            "suspected_role": None,
                            "confidence": 0.0,
                            "reason": "нет данных",
                            "pattern_ids": [],
                            "pattern_descriptions": [],
                            "reasons": ["нет данных"],
                        }
                    )
                )
            else:
                result.append(prediction)

        for player_id in sorted(silent_player_ids - prediction_ids):
            result.append(
                PlayerPrediction(
                    player_id=player_id,
                    player_name=name_by_id.get(player_id, f"Игрок {player_id}"),
                    suspected_role=None,
                    confidence=0.0,
                    reason="нет данных",
                    reasons=["нет данных"],
                )
            )

        return result

    def _postprocess_response(
        self,
        response: AnalysisResponse,
        *,
        context: LLMContext,
        previous_predictions: list[PredictionSnapshot],
        game_state: GameState,
    ) -> tuple[AnalysisResponse, list[str]]:
        """Keep role continuity, then enforce the bot's exact composition."""
        warnings: list[str] = []
        previous_by_id = {
            snapshot.player_id: snapshot.suspected_role for snapshot in previous_predictions
        }

        adjusted: list[PlayerPrediction] = []
        for prediction in response.analysis:
            old_role = previous_by_id.get(prediction.player_id)
            if (
                old_role is not None
                # ``suspected_role = null`` means "no data", not a role flip:
                # a data-free player keeps null instead of resurrecting an old role.
                and prediction.suspected_role is not None
                and prediction.suspected_role != old_role
                and not _has_new_evidence(prediction.reasons)
            ):
                warnings.append(
                    f"role_change_without_reason: player {prediction.player_id} "
                    f"kept {old_role!r} instead of {prediction.suspected_role!r}"
                )
                reasons = list(prediction.reasons)
                reasons.append(f"сохранена прежняя роль {old_role}: новый аргумент не приведён")
                adjusted.append(
                    prediction.model_copy(
                        update={
                            "suspected_role": old_role,
                            "confidence": min(prediction.confidence, 0.5),
                            "reasons": reasons[:5],
                        }
                    )
                )
            else:
                adjusted.append(prediction)

        if context.role_count:
            adjusted, composition_warnings = self._enforce_composition(
                adjusted, context, game_state
            )
            warnings.extend(composition_warnings)

        # Preserve object identity for unmodified responses (tests and callers
        # rely on getting the exact validated AnalysisResponse back).
        if not warnings and adjusted == response.analysis:
            return response, []
        return response.model_copy(update={"analysis": adjusted}), warnings

    def _enforce_composition(
        self,
        predictions: list[PlayerPrediction],
        context: LLMContext,
        game_state: GameState,
    ) -> tuple[list[PlayerPrediction], list[str]]:
        """Cap role quotas and fill missing predictions only for a complete composition.

        The model may skip silent players, repeat ``Мафия`` too often, or name a
        role that is not in the bot's composition. Those entries are corrected
        deterministically instead of silently persisting a broken distribution.
        """
        warnings: list[str] = []
        quota = dict(context.role_count or {})
        alive = game_state.alive_players()
        alive_ids = [player.player_id for player in alive]
        alive_set = set(alive_ids)
        silent_ids = context.silent_player_ids
        names_by_id = {player.player_id: player.display_name for player in alive}

        def canonical(role_name: str | None) -> str | None:
            if role_name is None:
                # ``suspected_role = null`` means "no data": the slot stays
                # unfilled for silent players.
                return None
            role = self._registry.resolve_alias(role_name)
            return role.name_ru if role is not None else role_name

        used: dict[str, int] = {}
        kept: dict[int, PlayerPrediction] = {}
        for prediction in predictions:
            if prediction.player_id not in alive_set:
                warnings.append(f"prediction_for_non_alive_player: {prediction.player_id} dropped")
                continue
            if prediction.player_id in kept:
                warnings.append(f"duplicate_prediction: {prediction.player_id} dropped")
                continue
            role = canonical(prediction.suspected_role)
            if role is None:
                # Null role (silent player): keep, do not consume a quota slot.
                kept[prediction.player_id] = prediction
                continue
            if role not in quota:
                warnings.append(
                    f"role_not_in_composition: {prediction.suspected_role!r} "
                    f"for {prediction.player_id}"
                )
                continue
            if used.get(role, 0) >= quota[role]:
                warnings.append(f"role_quota_exceeded: {role!r} for {prediction.player_id}")
                continue
            used[role] = used.get(role, 0) + 1
            kept[prediction.player_id] = (
                prediction.model_copy(update={"suspected_role": role})
                if role != prediction.suspected_role
                else prediction
            )

        total_slots = sum(quota.values())
        non_silent_alive_count = len([pid for pid in alive_ids if pid not in silent_ids])
        if total_slots != non_silent_alive_count:
            warning = (
                f"composition_mismatch: role slots={total_slots}, "
                f"non-silent alive players={non_silent_alive_count}; "
                f"deterministic filling skipped"
            )
            warnings.append(warning)
            logger.bind(channel="llm").warning("{}", warning)
            return [kept[pid] for pid in alive_ids if pid in kept], warnings

        remaining: list[str] = []
        for role, count in quota.items():
            remaining.extend([role] * max(0, count - used.get(role, 0)))
        # Only non-silent players without a prediction are filled from
        # remaining composition slots; silent players keep their null role.
        missing_ids = [pid for pid in alive_ids if pid not in kept and pid not in silent_ids]

        for player_id, role in zip(missing_ids, remaining, strict=False):
            warnings.append(f"composition_slot_filled: {player_id} → {role!r}")
            kept[player_id] = PlayerPrediction(
                player_id=player_id,
                player_name=names_by_id[player_id],
                suspected_role=role,
                confidence=_COMPOSITION_FILL_CONFIDENCE,
                reason=f"Роль «{role}» из точного состава бота.",
                reasons=[f"распределение состава бота: слот «{role}»"],
            )

        if len(missing_ids) > len(remaining):
            warnings.append(
                "composition_mismatch: more non-silent alive players than role slots; "
                "some players left without a prediction"
            )
        elif len(remaining) > len(missing_ids):
            warnings.append(
                "composition_mismatch: more role slots than non-silent alive players; "
                "unused slots remain unfilled"
            )

        ordered = [kept[pid] for pid in alive_ids if pid in kept]
        return ordered, warnings

    def _render_hints(self, hints: list[HeuristicHint]) -> str:
        """Render hints as a short bullet list for ``analyze.jinja``."""
        if not hints:
            return ""
        return "\n".join(f"- [{hint.weight:+.2f}] {hint.description}" for hint in hints)
