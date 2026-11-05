"""Unit tests for high-level game analysis orchestration."""

from __future__ import annotations

import io
import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2
import pytest
from loguru import logger
from pydantic import ValidationError

if TYPE_CHECKING:
    from loguru import Record

from domain.club_patterns import ClubPatterns, load_club_patterns
from domain.events import (
    AliveListEvent,
    AliveListPayload,
    DeathEvent,
    DeathPayload,
    MessageEvent,
    MessagePayload,
    PhaseChangeEvent,
    PhaseChangePayload,
    VoteEvent,
    VotePayload,
)
from domain.exceptions import LLMConnectionError, LLMError
from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles
from features.analyzer.analyzer import AnalysisResult, GameAnalyzer
from infra.db.connection import connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository
from infra.llm.client import LLMClient, ReasoningSink
from infra.llm.schemas import (
    AlternativePrediction,
    AnalysisResponse,
    PlayerPrediction,
)

ROLES_PATH = Path(__file__).parents[4] / "configs" / "roles.yaml"
PATTERNS_PATH = Path(__file__).parents[4] / "configs" / "club_patterns.yaml"
BASE_TS = datetime(2026, 1, 1, tzinfo=UTC)


class FakeLLMClient(LLMClient):
    """Test double that returns predefined responses or errors."""

    def __init__(self) -> None:
        self._next_response: AnalysisResponse | None = None
        self._next_error: Exception | None = None
        self._next_reasoning = ""
        self.calls: list[dict[str, Any]] = []

    def set_response(self, response: AnalysisResponse, reasoning: str = "") -> None:
        """Configure the next call to return a response and reasoning stream."""
        self._next_response = response
        self._next_error = None
        self._next_reasoning = reasoning

    def set_error(self, error: Exception) -> None:
        """Configure the next call to raise an LLM error."""
        self._next_error = error
        self._next_response = None
        self._next_reasoning = ""

    async def healthcheck(self) -> None:
        """Implement the no-op fake health check."""

    async def analyze(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        silent_player_ids: frozenset[int] = frozenset(),
    ) -> AnalysisResponse:
        """Delegate non-streaming calls to the fake streaming implementation."""
        return await self.analyze_stream(
            messages=messages,
            known_player_ids=known_player_ids,
            role_names_ru=role_names_ru,
            role_alias_resolver=role_alias_resolver,
            silent_player_ids=silent_player_ids,
        )

    async def analyze_stream(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        on_reasoning_chunk: ReasoningSink | None = None,
        log_reasoning_chunks: bool = True,
        silent_player_ids: frozenset[int] = frozenset(),
    ) -> AnalysisResponse:
        """Record LLM arguments and return the configured fake result."""
        call: dict[str, Any] = {
            "messages": messages,
            "known_player_ids": known_player_ids,
            "role_names_ru": role_names_ru,
            "role_alias_resolver": role_alias_resolver,
            "on_reasoning_chunk": on_reasoning_chunk,
            "log_reasoning_chunks": log_reasoning_chunks,
        }
        alias_resolutions: list[tuple[str, str | None]] = []
        if self._next_response is not None and role_alias_resolver is not None:
            for prediction in self._next_response.analysis:
                alternative = prediction.alternative
                if alternative is not None and alternative.role not in role_names_ru:
                    alias_resolutions.append(
                        (alternative.role, role_alias_resolver(alternative.role))
                    )
        call["alias_resolutions"] = alias_resolutions
        self.calls.append(call)

        if on_reasoning_chunk is not None and self._next_reasoning:
            for chunk in self._next_reasoning:
                on_reasoning_chunk(chunk)
        if self._next_error is not None:
            error = self._next_error
            self._next_error = None
            raise error
        if self._next_response is None:
            raise LLMError("FakeLLMClient: no response configured")
        return self._next_response

    async def aclose(self) -> None:
        """Implement the no-op fake close operation."""


@pytest.fixture
def registry() -> RoleRegistry:
    """Load the real configured role registry."""
    return load_roles(ROLES_PATH)


@pytest.fixture
def jinja_env() -> jinja2.Environment:
    """Create the environment used to render application prompt templates."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader("infra/llm/prompts"),
        autoescape=False,
    )


@pytest.fixture
def memory_database() -> Iterator[tuple[sqlite3.Connection, PredictionsRepository]]:
    """Create an in-memory SQLite database with all migrations applied."""
    connection = connect(":memory:")
    apply_migrations(connection)
    try:
        yield connection, PredictionsRepository(connection)
    finally:
        connection.close()


def create_game_with_players(connection: sqlite3.Connection, player_ids: list[int]) -> int:
    """Insert the foreign-key rows required for predictions and return the game id."""
    game_id = GamesRepository(connection).create_game(
        chat_id=1,
        player_count=len(player_ids),
    )
    players_repository = PlayersRepository(connection)
    for player_id in player_ids:
        players_repository.upsert_player(
            player_id=player_id,
            username=None,
            display_name=f"Player {player_id}",
        )
    return game_id


def make_game_state(
    registry: RoleRegistry,
    *,
    game_id: int = 1,
    player_ids: tuple[int, ...] = (),
) -> GameState:
    """Build a game state and register the requested test players."""
    game_state = GameState(game_id=game_id, registry=registry)
    for player_id in player_ids:
        game_state.register_player(player_id, f"Player {player_id}")
    return game_state


def add_message(
    state: GameState,
    player_id: int,
    *,
    text: str = "тестовое сообщение",
) -> None:
    """Add a message event so the player is not classified as silent."""
    state.apply(
        MessageEvent(
            ts=BASE_TS,
            game_id=state.game_id,
            payload=MessagePayload(
                author_id=player_id,
                author_name=f"Player {player_id}",
                text=text,
            ),
        )
    )


def make_prediction(
    player_id: int,
    role_name: str,
    *,
    confidence: float = 0.7,
    reason: str = "observed behavior",
    reasons: list[str] | None = None,
    alternative: AlternativePrediction | None = None,
) -> PlayerPrediction:
    """Build a valid prediction for a synthetic player."""
    return PlayerPrediction(
        player_id=player_id,
        player_name=f"Player {player_id}",
        suspected_role=role_name,
        confidence=confidence,
        reason=reason,
        alternative=alternative,
        reasons=reasons if reasons is not None else ["observed behavior"],
    )


def make_response(predictions: list[PlayerPrediction]) -> AnalysisResponse:
    """Build a valid model response around the given player predictions."""
    return AnalysisResponse(
        analysis=predictions,
        game_read="A short game read.",
        reasoning="A short response reasoning.",
    )


def is_reasoning_log(record: Record) -> bool:
    """Select log records bound to the reasoning channel."""
    return bool(record["extra"].get("channel") == "reasoning")


def make_analyzer(
    fake_llm: FakeLLMClient,
    predictions_repo: PredictionsRepository,
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    *,
    max_messages: int = 200,
    reasoning_max_chars: int = 4096,
    club_patterns: list[dict[str, object]] | None = None,
    patterns: ClubPatterns | None = None,
) -> GameAnalyzer:
    """Create a GameAnalyzer with test dependencies."""
    return GameAnalyzer(
        llm_client=fake_llm,
        predictions_repo=predictions_repo,
        registry=registry,
        jinja_env=jinja_env,
        max_messages=max_messages,
        reasoning_max_chars=reasoning_max_chars,
        club_patterns=club_patterns,
        patterns=patterns,
    )


async def test_analyze_success_returns_response(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """A valid response is returned as a successful AnalysisResult."""
    _connection, repository = memory_database
    fake_llm = FakeLLMClient()
    response = make_response([])
    fake_llm.set_response(response)
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(make_game_state(registry), persist=False)

    assert isinstance(result, AnalysisResult)
    assert result.success is True
    assert result.response is response
    assert result.error is None
    assert fake_llm.calls[0]["log_reasoning_chunks"] is False


async def test_analyzer_passes_club_patterns(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """The analyzer sends configured club patterns through the actual prompt builder."""
    _connection, repository = memory_database
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([]))
    analyzer = make_analyzer(
        fake_llm,
        repository,
        registry,
        jinja_env,
        club_patterns=[
            {
                "id": "#1",
                "trigger": "claim_investigation",
                "conclusion_role": "Комиссар Каттани",
                "confidence_tier": 0.8,
                "requires_confirmation": True,
                "description": "Игрок заявляет, что проверил чью-то роль.",
            }
        ],
    )

    result = await analyzer.analyze(make_game_state(registry), persist=False)

    assert result.success
    prompt = fake_llm.calls[0]["messages"][1]["content"]
    assert "Паттерны этого клуба" in prompt
    assert "#1" in prompt
    assert "claim_investigation" in prompt
    assert "Комиссар Каттани" in prompt
    assert "0.8" in prompt


async def test_analyze_llm_error_returns_result_with_error(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """An LLMError becomes an unsuccessful result instead of escaping."""
    _connection, repository = memory_database
    fake_llm = FakeLLMClient()
    fake_llm.set_error(LLMError("boom"))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(make_game_state(registry))

    assert result.success is False
    assert result.response is None
    assert "boom" in (result.error or "")


async def test_analyze_validation_error_returns_llm_error_result(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Direct response validation errors are translated into LLM errors."""
    _connection, repository = memory_database
    fake_llm = FakeLLMClient()
    with pytest.raises(ValidationError) as validation_error:
        AnalysisResponse.model_validate(
            {"analysis": "not a list", "game_read": "ok", "reasoning": "ok"}
        )
    fake_llm.set_error(validation_error.value)
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(make_game_state(registry))

    assert result.success is False
    assert result.response is None
    assert (result.error or "").startswith("invalid response:")


async def test_analyze_llm_connection_error(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """A connection error is returned with the documented connection prefix."""
    _connection, repository = memory_database
    fake_llm = FakeLLMClient()
    fake_llm.set_error(LLMConnectionError("refused"))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(make_game_state(registry))

    assert result.success is False
    assert result.response is None
    assert (result.error or "").startswith("connection:")


async def test_analyze_persists_predictions(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Current prediction stores full reasoning; per-player reasons remain in history."""
    connection, repository = memory_database
    player_ids = [100, 101]
    game_id = create_game_with_players(connection, player_ids)
    game_state = make_game_state(registry, game_id=game_id, player_ids=tuple(player_ids))
    add_message(game_state, 100)
    add_message(game_state, 101)
    roles = registry.all_roles()
    fake_llm = FakeLLMClient()
    fake_llm.set_response(
        make_response(
            [
                make_prediction(100, roles[0].name_ru, reasons=["first reason", "second reason"]),
                make_prediction(101, roles[1].name_ru, confidence=0.4),
            ]
        ).model_copy(update={"reasoning": "JSON reasoning " * 500}),
        reasoning="Different streamed reasoning",
    )
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(game_state)

    rows = repository.list_predictions(game_id)
    assert result.success is True
    assert len(rows) == 2
    assert rows[0]["player_id"] == 100
    assert rows[0]["predicted_role"] == roles[0].name_ru
    assert rows[0]["confidence"] == pytest.approx(0.7)
    assert rows[0]["reasoning"] == "JSON reasoning " * 500
    history = connection.execute(
        "SELECT reasoning FROM predictions_history WHERE game_id = ? AND player_id = ?",
        (game_id, 100),
    ).fetchone()
    assert history is not None and history["reasoning"] == "first reason\nsecond reason"
    assert rows[1]["player_id"] == 101


async def test_analyze_persist_false_skips_db(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Disabling persistence leaves the in-memory predictions table empty."""
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100])
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([make_prediction(100, registry.all_roles()[0].name_ru)]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(
        make_game_state(registry, game_id=game_id, player_ids=(100,)),
        persist=False,
    )

    assert result.success is True
    assert repository.list_predictions(game_id) == []


async def test_analyze_persistence_failure_adds_warning(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """A repository failure is reported as a warning without losing the response."""
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100])

    class FailingPredictionsRepository(PredictionsRepository):
        """Repository double that always fails to insert predictions."""

        def upsert_prediction(
            self,
            *,
            game_id: int,
            player_id: int,
            predicted_role: str | None,
            confidence: float,
            reasoning: str,
            reason: str = "",
            pattern_ids: list[str] | None = None,
            pattern_descriptions: list[str] | None = None,
        ) -> int:
            raise sqlite3.OperationalError("disk full")

    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([make_prediction(100, registry.all_roles()[0].name_ru)]))
    failing_repository = FailingPredictionsRepository(connection)
    analyzer = make_analyzer(fake_llm, failing_repository, registry, jinja_env)

    result = await analyzer.analyze(make_game_state(registry, game_id=game_id, player_ids=(100,)))

    assert result.success is True
    assert result.warnings
    assert "persistence_failed" in result.warnings[0]
    assert "disk full" in result.warnings[0]
    assert repository.list_predictions(game_id) == []
    assert connection.execute("SELECT count(*) FROM predictions_history").fetchone()[0] == 0


async def test_non_thinking_uses_json_reasoning_as_fallback(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """A response without streamed chunks displays and counts JSON reasoning."""
    _connection, repository = memory_database
    response = make_response([]).model_copy(update={"reasoning": "My COT"})
    fake_llm = FakeLLMClient()
    fake_llm.set_response(response)
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)
    captured = io.StringIO()
    sink = logger.add(captured, level="INFO", format="{message}", filter=is_reasoning_log)
    try:
        result = await analyzer.analyze(make_game_state(registry), persist=False)
    finally:
        logger.remove(sink)

    assert result.success
    assert result.reasoning_text == "My COT"
    assert result.reasoning_chars == len("My COT")
    assert captured.getvalue().rstrip("\n") == "reasoning (json) [6]: My COT"


async def test_thinking_uses_streaming_not_json(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """A nonempty reasoning stream always takes precedence over JSON."""
    _connection, repository = memory_database
    response = make_response([]).model_copy(update={"reasoning": "json COT"})
    fake_llm = FakeLLMClient()
    fake_llm.set_response(response, reasoning="stream COT")
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)
    captured = io.StringIO()
    sink = logger.add(captured, level="INFO", format="{message}", filter=is_reasoning_log)
    try:
        result = await analyzer.analyze(make_game_state(registry), persist=False)
    finally:
        logger.remove(sink)

    assert result.success
    assert result.reasoning_text == "stream COT"
    assert result.reasoning_chars == len("stream COT")
    assert captured.getvalue().rstrip("\n") == "reasoning (stream) [10]: stream COT"


async def test_json_reasoning_fallback_is_truncated_and_counted(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """JSON fallback obeys the configured limit and retains the full character count."""
    _connection, repository = memory_database
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([]).model_copy(update={"reasoning": "z" * 500}))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env, reasoning_max_chars=80)

    result = await analyzer.analyze(make_game_state(registry), persist=False)

    assert result.success
    assert result.reasoning_text == "z" * 80 + "\n[truncated, total 500 chars]"
    assert result.reasoning_chars == 500


async def test_failed_analysis_preserves_partial_stream(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """An LLM failure has no JSON fallback but still preserves partial streamed text."""
    _connection, repository = memory_database
    fake_llm = FakeLLMClient()
    fake_llm.set_error(LLMConnectionError("offline"))
    fake_llm._next_reasoning = "p" * 150
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env, reasoning_max_chars=100)

    result = await analyzer.analyze(make_game_state(registry), persist=False)

    assert not result.success
    assert result.reasoning_chars == 150
    assert result.reasoning_text == "p" * 100 + "\n[truncated, total 150]"


async def test_analyze_reasoning_truncated(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Long reasoning is truncated in logs but fully counted in the result."""
    _connection, repository = memory_database
    reasoning = "x" * 5000
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([]), reasoning=reasoning)
    analyzer = make_analyzer(
        fake_llm,
        repository,
        registry,
        jinja_env,
        reasoning_max_chars=1000,
    )
    captured = io.StringIO()
    sink_id = logger.add(
        captured,
        level="INFO",
        format="{message}",
        filter=is_reasoning_log,
    )
    try:
        result = await analyzer.analyze(make_game_state(registry), persist=False)
    finally:
        logger.remove(sink_id)

    logged_reasoning = captured.getvalue().rstrip("\n")
    assert logged_reasoning == f"reasoning (stream) [5000]: {'x' * 200}"
    assert result.reasoning_chars == 5000
    assert result.reasoning_text == "x" * 1000 + "\n[truncated, total 5000 chars]"


async def test_analyze_reasoning_not_truncated_when_short(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Short reasoning is logged with its source and full length."""
    _connection, repository = memory_database
    reasoning = "r" * 100
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([]), reasoning=reasoning)
    analyzer = make_analyzer(
        fake_llm,
        repository,
        registry,
        jinja_env,
        reasoning_max_chars=1000,
    )
    captured = io.StringIO()
    sink_id = logger.add(
        captured,
        level="INFO",
        format="{message}",
        filter=is_reasoning_log,
    )
    try:
        result = await analyzer.analyze(make_game_state(registry), persist=False)
    finally:
        logger.remove(sink_id)

    assert captured.getvalue().rstrip("\n") == f"reasoning (stream) [100]: {reasoning}"
    assert result.reasoning_chars == 100
    assert result.reasoning_text == reasoning


async def test_analyze_passes_hints_to_context(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """An early lynch hint is rendered into the user prompt for the LLM."""
    _connection, repository = memory_database
    game_state = make_game_state(registry, game_id=1, player_ids=(100, 101, 102, 103))
    game_state.apply(
        PhaseChangeEvent(
            ts=BASE_TS,
            game_id=1,
            payload=PhaseChangePayload(
                from_phase=Phase.REGISTRATION,
                to_phase=Phase.DAY,
                day_number=1,
            ),
        )
    )
    game_state.apply(
        DeathEvent(
            ts=BASE_TS,
            game_id=1,
            payload=DeathPayload(victim_id=100, victim_name="Player 100", cause="lynch"),
        )
    )
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(game_state, persist=False)

    assert result.success is True
    user_message = fake_llm.calls[0]["messages"][1]["content"]
    assert "Линч случился в первые дни без аргументации" in user_message
    assert "- [-0.50]" in user_message


async def test_analyze_empty_players_ok(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """A game with no registered players succeeds with empty analysis."""
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [])
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(make_game_state(registry, game_id=game_id))

    assert result.success is True
    assert result.response is not None
    assert result.response.analysis == []
    assert repository.list_predictions(game_id) == []


async def test_analyze_no_crash_on_validation_edge(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """A structurally valid response with zero predictions remains valid."""
    _connection, repository = memory_database
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(make_game_state(registry), persist=False)

    assert result.success is True
    assert result.response is not None
    assert result.response.analysis == []


async def test_analyze_uses_game_id_from_game_state(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Predictions are stored under the state's public game id."""
    connection, repository = memory_database
    other_game_id = create_game_with_players(connection, [999])
    target_game_id = create_game_with_players(connection, [100])
    assert other_game_id != target_game_id
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([make_prediction(100, registry.all_roles()[0].name_ru)]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(
        make_game_state(registry, game_id=target_game_id, player_ids=(100,))
    )

    assert result.success is True
    assert repository.list_predictions(other_game_id) == []
    assert len(repository.list_predictions(target_game_id)) == 1


async def test_analyze_duration_positive(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """The analyzer reports a positive elapsed duration for a completed cycle."""
    _connection, repository = memory_database
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(make_game_state(registry), persist=False)

    assert result.duration_sec >= 0


async def test_analyze_duration_clamped_non_negative(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """duration_sec is clamped to >= 0 even if perf_counter rolls
    back (QPC bug on Windows CI)."""
    import time as _time

    # Сделать так, что второй вызов perf_counter < первого.
    calls = iter([100.0, 99.97])  # delta = -0.03
    real_pc = _time.perf_counter

    def fake_pc() -> float:
        try:
            return next(calls)
        except StopIteration:
            return real_pc()

    # Dotted string form: ``analyzer_mod.time`` triggers mypy attr-defined
    # (strict no_implicit_reexport), while pytest resolves this path itself.
    monkeypatch.setattr("features.analyzer.analyzer.time.perf_counter", fake_pc)

    _connection, repository = memory_database
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(make_game_state(registry), persist=False)

    assert result.duration_sec == 0.0, f"expected clamp to 0.0, got {result.duration_sec}"


async def test_analyze_known_player_ids_passed(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """The LLM client receives exactly the player ids in the game context."""
    _connection, repository = memory_database
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    await analyzer.analyze(
        make_game_state(registry, player_ids=(100, 101)),
        persist=False,
    )

    assert fake_llm.calls[0]["known_player_ids"] == frozenset({100, 101})


async def test_analyze_role_alias_resolver_works(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """An alternative role alias is resolved by the callback sent to the LLM."""
    _connection, repository = memory_database
    role_names = {role.name_ru for role in registry.all_roles()}
    alias = next(
        alias
        for role in registry.all_roles()
        for alias in role.bot_aliases
        if alias not in role_names
    )
    resolved_role = registry.resolve_alias(alias)
    assert resolved_role is not None
    primary_role = next(
        role for role in registry.all_roles() if role.name_ru != resolved_role.name_ru
    )
    prediction = make_prediction(
        100,
        primary_role.name_ru,
        confidence=0.6,
        alternative=AlternativePrediction(role=alias, confidence=0.2),
    )
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([prediction]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    result = await analyzer.analyze(
        make_game_state(registry, player_ids=(100,)),
        persist=False,
    )

    assert result.success is True
    assert fake_llm.calls[0]["alias_resolutions"] == [(alias, resolved_role.name_ru)]


async def test_analyze_persist_runs_in_thread(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All repository calls are offloaded via asyncio.to_thread,
    not blocking the event loop (including the previous-predictions read)."""
    import asyncio as _asyncio

    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100])

    # Подменить asyncio.to_thread на spy, чтобы убедиться,
    # что оба метода репозитория вызываются через него.
    real_to_thread = _asyncio.to_thread
    calls: list[str] = []

    async def spy_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(func.__name__)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(
        "features.analyzer.analyzer.asyncio.to_thread",
        spy_to_thread,
    )

    fake_llm = FakeLLMClient()
    fake_llm.set_response(
        make_response(
            [
                make_prediction(player_id=100, role_name="Мафия"),
            ]
        )
    )
    analyzer = make_analyzer(
        fake_llm,
        repository,
        registry,
        jinja_env,
    )
    state = make_game_state(registry, game_id=game_id, player_ids=(100,))
    add_message(state, 100)
    result = await analyzer.analyze(state, persist=True)
    assert result.success
    assert calls == ["list_predictions", "upsert_prediction", "insert_history"]


async def test_analyzer_uses_upsert_and_history(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100, 101])
    calls: list[tuple[str, int]] = []
    for name in ("upsert_prediction", "insert_history"):
        original = getattr(repository, name)

        def spy(*, _name: str = name, _original: Any = original, **kwargs: Any) -> int:
            calls.append((_name, kwargs["player_id"]))
            return int(_original(**kwargs))

        monkeypatch.setattr(repository, name, spy)
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([make_prediction(i, "Мафия") for i in (100, 101)]))
    state = make_game_state(registry, game_id=game_id, player_ids=(100, 101))
    add_message(state, 100)
    add_message(state, 101)
    result = await make_analyzer(fake_llm, repository, registry, jinja_env).analyze(state)
    assert result.success and not result.warnings
    assert calls == [
        ("upsert_prediction", 100),
        ("insert_history", 100),
        ("upsert_prediction", 101),
        ("insert_history", 101),
    ]


async def test_history_failure_is_warning(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100])

    def fail(**kwargs: Any) -> int:
        raise sqlite3.OperationalError("history unavailable")

    monkeypatch.setattr(repository, "insert_history", fail)
    llm = FakeLLMClient()
    llm.set_response(make_response([make_prediction(100, "Мафия")]))
    state = make_game_state(registry, game_id=game_id, player_ids=(100,))
    add_message(state, 100)
    result = await make_analyzer(llm, repository, registry, jinja_env).analyze(state)
    assert result.success and result.warnings == ("history_failed: history unavailable",)
    assert len(repository.list_predictions(game_id)) == 1


# ===================================================================
# Role continuity + exact bot composition
# ===================================================================


async def test_analyzer_preserves_previous_role_without_new_evidence(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """'Мирный житель' must not silently become 'Мафия' on the next flush."""
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100])
    fake_llm = FakeLLMClient()
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env)

    fake_llm.set_response(
        make_response(
            [
                make_prediction(
                    100,
                    "Мирный житель",
                    confidence=0.42,
                    reasons=["распределение состава бота: слот «Мирный житель»"],
                )
            ]
        )
    )
    state1 = make_game_state(registry, game_id=game_id, player_ids=(100,))
    add_message(state1, 100)
    first = await analyzer.analyze(state1, persist=True)
    assert first.success
    assert len(repository.list_predictions(game_id)) == 1

    fake_llm.set_response(
        make_response([make_prediction(100, "Мафия", confidence=0.8, reasons=["молчит"])])
    )
    state2 = make_game_state(registry, game_id=game_id, player_ids=(100,))
    add_message(state2, 100)
    second = await analyzer.analyze(state2, persist=True)

    assert second.success and second.response is not None
    kept = second.response.analysis[0]
    assert kept.suspected_role == "Мирный житель"
    assert kept.confidence <= 0.5
    assert any("role_change_without_reason" in warning for warning in second.warnings)
    # The second prompt explicitly carried the earlier decision.
    second_prompt = fake_llm.calls[1]["messages"][1]["content"]
    assert "Прошлые предсказания (якорь)" in second_prompt
    assert "ID 100: Мирный житель" in second_prompt
    assert "НЕ меняй роль игрока без НОВОГО аргумента" in second_prompt
    assert repository.list_predictions(game_id)[0]["predicted_role"] == "Мирный житель"


async def test_analyzer_fills_full_composition_and_caps_quotas(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """3 bot slots → exactly 3 predictions; 'Мафия' is used only once."""
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100, 101, 102])
    state = make_game_state(registry, game_id=game_id, player_ids=(100, 101, 102))
    add_message(state, 100)
    add_message(state, 101)
    add_message(state, 102)
    state.apply(
        AliveListEvent(
            ts=BASE_TS,
            game_id=game_id,
            payload=AliveListPayload(
                alive_player_names=["Player 100", "Player 101", "Player 102"],
                role_count={"Мафия": 1, "Мирный Житель": 2},
                total_players=3,
            ),
        )
    )

    fake_llm = FakeLLMClient()
    fake_llm.set_response(
        make_response(
            [
                make_prediction(100, "Мафия", reasons=["активно давит в первый день"]),
                make_prediction(101, "Мафия", reasons=["тоже подозрительно молчит"]),
            ]
        )
    )
    result = await make_analyzer(fake_llm, repository, registry, jinja_env).analyze(
        state, persist=True
    )

    assert result.success and result.response is not None
    roles = {p.player_id: p.suspected_role for p in result.response.analysis}
    assert len(roles) == 3
    assert roles[100] == "Мафия"
    assert roles[101] == "Мирный житель"
    assert roles[102] == "Мирный житель"
    assert any("role_quota_exceeded" in warning for warning in result.warnings)
    assert sum("composition_slot_filled" in warning for warning in result.warnings) == 2
    persisted = {
        int(row["player_id"]): str(row["predicted_role"])
        for row in repository.list_predictions(game_id)
    }
    assert persisted == roles


@pytest.mark.parametrize("reasoning", ["", None, "missing"])
async def test_analyze_missing_reasoning_falls_back_to_player_reasons(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
    reasoning: str | None,
) -> None:
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100])
    state = make_game_state(registry, game_id=game_id, player_ids=(100,))
    add_message(state, 100)
    prediction = make_prediction(100, registry.all_roles()[0].name_ru, reasons=["причина", "довод"])
    raw: dict[str, Any] = {"analysis": [prediction], "game_read": "Обзор"}
    if reasoning != "missing":
        raw["reasoning"] = reasoning
    response = AnalysisResponse.model_validate(raw)
    fake_llm = FakeLLMClient()
    fake_llm.set_response(response)
    result = await make_analyzer(fake_llm, repository, registry, jinja_env).analyze(state)
    assert result.success
    assert repository.list_predictions(game_id)[0]["reasoning"] == "причина\nдовод"
    assert (
        connection.execute("SELECT reasoning FROM predictions_history").fetchone()[0]
        == "причина\nдовод"
    )


async def test_incomplete_composition_never_fills_slots_and_logs_warning(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    connection, repository = memory_database
    ids = [100, 101, 102, 103, 104]
    game_id = create_game_with_players(connection, ids)
    state = make_game_state(registry, game_id=game_id, player_ids=tuple(ids))
    add_message(state, 100)  # player 100 is not silent; 101-104 stay silent
    state.apply(
        AliveListEvent(
            ts=BASE_TS,
            game_id=game_id,
            payload=AliveListPayload(
                alive_player_names=[f"Player {i}" for i in ids],
                role_count={"Мафия": 2},
                total_players=5,
            ),
        )
    )
    llm = FakeLLMClient()
    llm.set_response(make_response([make_prediction(100, "Мафия")]))
    logs: list[str] = []
    sink = logger.add(lambda message: logs.append(str(message)), level="WARNING")
    try:
        result = await make_analyzer(llm, repository, registry, jinja_env).analyze(state)
    finally:
        logger.remove(sink)
    assert result.response is not None
    assert [p.player_id for p in result.response.analysis] == ids
    assert result.response.analysis[0].suspected_role == "Мафия"
    assert all(p.suspected_role is None for p in result.response.analysis[1:])
    assert all(p.confidence == 0.0 for p in result.response.analysis[1:])
    assert all(p.reason == "нет данных" for p in result.response.analysis[1:])
    assert [r["player_id"] for r in repository.list_predictions(game_id)] == ids
    assert any("composition_mismatch" in w and "2" in w and "1" in w for w in result.warnings)
    assert any("composition_mismatch" in msg and "2" in msg and "1" in msg for msg in logs)


@pytest.mark.parametrize("quota", [None, {}, {"Мафия": 0}])
def test_composition_empty_quota_does_not_fill(
    quota: dict[str, int] | None,
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    from features.analyzer.context_builder import build_context

    state = make_game_state(registry, player_ids=(100,))
    context = build_context(game_state=state, registry=registry, max_messages=20).model_copy(
        update={"role_count": quota}
    )
    analyzer = make_analyzer(FakeLLMClient(), memory_database[1], registry, jinja_env)
    predictions, _ = analyzer._enforce_composition([], context, state)
    assert predictions == []


def load_real_patterns() -> ClubPatterns:
    """Load the repository's real club-patterns configuration."""
    return load_club_patterns(PATTERNS_PATH)


async def test_pattern_direct_claim_gives_role_conf09_pattern_ids(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Pattern #2 (direct role claim) → claimed role, confidence 0.9, pattern_ids=["#2"]."""
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100])
    state = make_game_state(registry, game_id=game_id, player_ids=(100,))
    add_message(state, 100, text="я - Мирный")

    patterns = load_real_patterns()
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env, patterns=patterns)

    result = await analyzer.analyze(state, persist=False)

    assert result.success and result.response is not None
    assert len(result.response.analysis) == 1
    p = result.response.analysis[0]
    assert p.player_id == 100
    assert p.suspected_role == "Мирный житель"
    assert p.confidence == 0.9
    assert p.pattern_ids == ["#2"]
    assert len(p.pattern_descriptions) == len(p.pattern_ids)
    assert "паттерн #2" in p.reason


async def test_pattern_investigation_without_confirmation_does_not_fire(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Pattern #1 (investigation claim) without vote confirmation → no override."""
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100])
    state = make_game_state(registry, game_id=game_id, player_ids=(100,))
    add_message(state, 100, text="я проверил user_101, он Комиссар")

    patterns = load_real_patterns()
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([make_prediction(100, "Мафия", confidence=0.5)]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env, patterns=patterns)

    result = await analyzer.analyze(state, persist=False)

    assert result.success and result.response is not None
    p = result.response.analysis[0]
    # No vote confirmation → pattern #1 must not fire → LLM role stays
    assert p.suspected_role == "Мафия"
    assert p.pattern_ids == []
    assert p.pattern_descriptions == []


async def test_unmatched_direct_claim_empty_pattern_ids_and_reason(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Unmatched direct claim → empty pattern_ids, reason notes no match, LLM role stays."""
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100])
    state = make_game_state(registry, game_id=game_id, player_ids=(100,))
    add_message(state, 100, text="я - Мафия")

    patterns = load_real_patterns()
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([make_prediction(100, "Мафия", confidence=0.5)]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env, patterns=patterns)

    result = await analyzer.analyze(state, persist=False)

    assert result.success and result.response is not None
    p = result.response.analysis[0]
    assert p.player_id == 100
    # No pattern matched → LLM role stays, no override
    assert p.suspected_role == "Мафия"
    assert p.confidence == 0.5
    assert p.pattern_ids == []
    assert p.pattern_descriptions == []
    assert "прямое заявление не сопоставлено с паттерном" in p.reason


async def test_pattern_investigation_with_confirmation_fires(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Pattern #1 (investigation claim) with vote confirmation → override applies."""
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100])
    state = make_game_state(registry, game_id=game_id, player_ids=(100,))
    add_message(state, 100, text="я проверил user_101, он Комиссар")
    # Vote confirms the claim: author_id (100) is in vote_targets
    state.apply(
        VoteEvent(
            ts=BASE_TS,
            game_id=game_id,
            payload=VotePayload(
                voter_id=100, voter_name="Player 100", target_id=101, target_name="Player 101"
            ),
        )
    )

    patterns = load_real_patterns()
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([make_prediction(100, "Мафия", confidence=0.5)]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env, patterns=patterns)

    result = await analyzer.analyze(state, persist=False)

    assert result.success and result.response is not None
    p = result.response.analysis[0]
    # Vote confirmation present → pattern #1 fires → Комиссар Каттани, 0.8
    assert p.suspected_role == "Комиссар Каттани"
    assert p.confidence == 0.8
    assert p.pattern_ids == ["#1"]
    assert len(p.pattern_descriptions) == len(p.pattern_ids)
    assert "паттерн #1" in p.reason


async def test_pattern_override_releases_slot_and_redistributes(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Pattern override releases a composition slot which is then redistributed."""
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100, 101])
    state = make_game_state(registry, game_id=game_id, player_ids=(100, 101))
    add_message(state, 100, text="я - Мирный")
    add_message(state, 101, text="whatever")
    state.apply(
        AliveListEvent(
            ts=BASE_TS,
            game_id=game_id,
            payload=AliveListPayload(
                alive_player_names=["Player 100", "Player 101"],
                role_count={"Мафия": 1, "Мирный житель": 1},
                total_players=2,
            ),
        )
    )

    patterns = load_real_patterns()
    fake_llm = FakeLLMClient()
    fake_llm.set_response(make_response([make_prediction(100, "Мафия")]))
    analyzer = make_analyzer(fake_llm, repository, registry, jinja_env, patterns=patterns)

    result = await analyzer.analyze(state, persist=False)

    assert result.success and result.response is not None
    by_id = {p.player_id: p for p in result.response.analysis}
    # Player 100: pattern #2 overrides Мафия → Мирный, confidence 0.9
    assert by_id[100].suspected_role == "Мирный житель"
    assert by_id[100].confidence == 0.9
    assert by_id[100].pattern_ids == ["#2"]
    # Player 101: no LLM prediction, gets released Мафия slot via composition fill
    assert by_id[101].suspected_role == "Мафия"
    assert by_id[101].confidence == 0.3  # composition fill confidence


@pytest.mark.parametrize("death_before_request", [True, False])
async def test_dead_during_llm_response_is_not_persisted(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
    monkeypatch: pytest.MonkeyPatch,
    death_before_request: bool,
) -> None:
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100, 101])
    state = make_game_state(registry, game_id=game_id, player_ids=(100, 101))
    llm = FakeLLMClient()
    death = DeathEvent(
        ts=BASE_TS,
        game_id=game_id,
        payload=DeathPayload(victim_id=100, victim_name="Player 100"),
    )
    if death_before_request:
        state.apply(death)

    async def respond(**kwargs: Any) -> AnalysisResponse:
        if not death_before_request:
            state.apply(death)
        return make_response([make_prediction(100, "Мафия"), make_prediction(101, "Мафия")])

    monkeypatch.setattr(llm, "analyze_stream", respond)
    logs: list[str] = []
    sink = logger.add(lambda message: logs.append(str(message)), level="WARNING")
    try:
        result = await make_analyzer(llm, repository, registry, jinja_env).analyze(state)
    finally:
        logger.remove(sink)
    assert result.response is not None
    assert [p.player_id for p in result.response.analysis] == [101]
    assert [r["player_id"] for r in repository.list_predictions(game_id)] == [101]
    assert any("100" in msg and "dead" in msg for msg in logs)


async def test_dead_during_llm_gets_no_deterministic_composition_slot(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100, 101])
    state = make_game_state(registry, game_id=game_id, player_ids=(100, 101))
    add_message(state, 100)
    add_message(state, 101)
    state.apply(
        AliveListEvent(
            ts=BASE_TS,
            game_id=game_id,
            payload=AliveListPayload(
                alive_player_names=["Player 100", "Player 101"],
                role_count={"Мафия": 1, "Мирный житель": 1},
                total_players=2,
            ),
        )
    )
    llm = FakeLLMClient()

    async def respond(**kwargs: Any) -> AnalysisResponse:
        state.apply(
            DeathEvent(
                ts=BASE_TS,
                game_id=game_id,
                payload=DeathPayload(victim_id=100, victim_name="Player 100"),
            )
        )
        return make_response([])

    monkeypatch.setattr(llm, "analyze_stream", respond)
    result = await make_analyzer(llm, repository, registry, jinja_env).analyze(state)
    assert result.response is not None and result.response.analysis == []
    assert repository.list_predictions(game_id) == []
    assert any("composition_mismatch" in warning for warning in result.warnings)


async def test_player_removed_during_llm_is_dropped_with_warning(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, repository = memory_database
    game_id = create_game_with_players(connection, [-1])
    state = make_game_state(registry, game_id=game_id, player_ids=(-1,))
    llm = FakeLLMClient()

    async def respond(**kwargs: Any) -> AnalysisResponse:
        state.register_player(101, "Player -1")
        return make_response([make_prediction(-1, "Мафия")])

    monkeypatch.setattr(llm, "analyze_stream", respond)
    logs: list[str] = []
    sink = logger.add(lambda message: logs.append(str(message)), level="WARNING")
    try:
        result = await make_analyzer(llm, repository, registry, jinja_env).analyze(state)
    finally:
        logger.remove(sink)
    assert result.response is not None and result.response.analysis == []
    assert repository.list_predictions(game_id) == []
    assert any("-1" in msg and "missing" in msg for msg in logs)


async def test_death_during_persistence_drops_remaining_prediction(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    memory_database: tuple[sqlite3.Connection, PredictionsRepository],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    connection, repository = memory_database
    game_id = create_game_with_players(connection, [100, 101])
    state = make_game_state(registry, game_id=game_id, player_ids=(100, 101))
    llm = FakeLLMClient()
    llm.set_response(make_response([make_prediction(100, "Мафия"), make_prediction(101, "Мафия")]))
    to_thread = asyncio.to_thread

    async def spy(func: Any, *args: Any, **kwargs: Any) -> Any:
        if func.__name__ == "insert_history" and kwargs.get("player_id") == 100:
            state.apply(
                DeathEvent(
                    ts=BASE_TS,
                    game_id=game_id,
                    payload=DeathPayload(victim_id=101, victim_name="Player 101"),
                )
            )
        return await to_thread(func, *args, **kwargs)

    monkeypatch.setattr("features.analyzer.analyzer.asyncio.to_thread", spy)
    result = await make_analyzer(llm, repository, registry, jinja_env).analyze(state)
    assert result.response is not None
    assert [p.player_id for p in result.response.analysis] == [100]
    assert [row["player_id"] for row in repository.list_predictions(game_id)] == [100]
