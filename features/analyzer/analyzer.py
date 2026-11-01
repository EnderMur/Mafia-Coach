"""Orchestrate one game-analysis cycle across context, LLM, and persistence."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

from jinja2 import Environment
from loguru import logger
from pydantic import ValidationError

from domain.exceptions import LLMConnectionError, LLMError
from domain.game_state import GameState
from domain.roles import RoleRegistry
from features.analyzer.context_builder import (
    PredictionSnapshot,
    build_context,
    to_llm_messages,
)
from features.analyzer.heuristic_hints import HeuristicHint, compute_hints
from infra.db.repositories.predictions import PredictionsRepository
from infra.llm.client import LLMClient
from infra.llm.schemas import AnalysisResponse

DEFAULT_REASONING_MAX_CHARS: Final[int] = 4096


@dataclass(frozen=True)
class AnalysisResult:
    """Outcome of one analyzer cycle."""

    response: AnalysisResponse | None
    error: str | None
    duration_sec: float
    reasoning_chars: int
    warnings: tuple[str, ...] = ()

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
    ) -> None:
        """Retain the dependencies and context/reasoning limits for analysis."""
        self._llm_client = llm_client
        self._predictions_repo = predictions_repo
        self._registry = registry
        self._jinja_env = jinja_env
        self._max_messages = max_messages
        self._reasoning_max_chars = reasoning_max_chars

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
        """
        start = time.monotonic()
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
            )
        except LLMConnectionError as error:
            logger.bind(channel="llm").error("analysis_failed: connection: {}", str(error))
            return AnalysisResult(
                response=None,
                error=f"connection: {error}",
                duration_sec=time.monotonic() - start,
                reasoning_chars=sum(len(chunk) for chunk in reasoning_buffer),
            )
        except ValidationError as error:
            llm_error = LLMError(f"invalid response: {error}")
            logger.bind(channel="llm").error("analysis_failed: {}", str(llm_error))
            return AnalysisResult(
                response=None,
                error=str(llm_error),
                duration_sec=time.monotonic() - start,
                reasoning_chars=sum(len(chunk) for chunk in reasoning_buffer),
            )
        except LLMError as error:
            logger.bind(channel="llm").error("analysis_failed: {}", str(error))
            return AnalysisResult(
                response=None,
                error=str(error),
                duration_sec=time.monotonic() - start,
                reasoning_chars=sum(len(chunk) for chunk in reasoning_buffer),
            )

        reasoning_full = "".join(reasoning_buffer)
        reasoning_truncated = reasoning_full[: self._reasoning_max_chars]
        if len(reasoning_full) > self._reasoning_max_chars:
            reasoning_truncated += f"\n[truncated, total {len(reasoning_full)}]"
        if reasoning_truncated:
            logger.bind(channel="reasoning").info(reasoning_truncated)

        warnings: list[str] = []
        if persist:
            for prediction in response.analysis:
                try:
                    self._predictions_repo.insert_prediction(
                        game_id=game_state.game_id,
                        player_id=prediction.player_id,
                        predicted_role=prediction.suspected_role,
                        confidence=prediction.confidence,
                        reasoning="\n".join(prediction.reasons),
                    )
                except Exception as error:  # noqa: BLE001 — persistence is defensive
                    warnings.append(f"persistence_failed: {error}")
                    logger.bind(channel="llm").error("persistence_failed: {}", str(error))

        return AnalysisResult(
            response=response,
            error=None,
            duration_sec=time.monotonic() - start,
            reasoning_chars=len(reasoning_full),
            warnings=tuple(warnings),
        )

    def _render_hints(self, hints: list[HeuristicHint]) -> str:
        """Render hints as a short bullet list for ``analyze.jinja``."""
        if not hints:
            return ""
        return "\n".join(f"- [{hint.weight:+.2f}] {hint.description}" for hint in hints)
