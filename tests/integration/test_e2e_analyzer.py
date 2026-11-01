"""End-to-end tests for analysis, prediction persistence, and retro labeling."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jinja2
import pytest

from domain.events import MessageEvent, MessagePayload
from domain.game_state import GameState
from domain.roles import RoleRegistry, load_roles
from features.analyzer.analyzer import GameAnalyzer
from infra.db.connection import connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository
from infra.llm.client import LLMClient, ReasoningSink
from infra.llm.schemas import AnalysisResponse, PlayerPrediction

ROLES_PATH = Path(__file__).resolve().parents[2] / "configs" / "roles.yaml"
BASE_TS = datetime(2026, 1, 1, tzinfo=UTC)


class FakeLLMClient(LLMClient):
    """Fake streaming LLM that returns one predefined AnalysisResponse."""

    def __init__(self, response: AnalysisResponse) -> None:
        self._response = response

    async def healthcheck(self) -> None:
        """Implement the no-op fake health check."""

    async def analyze(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
    ) -> AnalysisResponse:
        """Return the configured response for a non-streaming call."""
        return self._response

    async def analyze_stream(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        on_reasoning_chunk: ReasoningSink | None = None,
        log_reasoning_chunks: bool = True,
    ) -> AnalysisResponse:
        """Return the configured response for a streaming call."""
        return self._response

    async def aclose(self) -> None:
        """Implement the no-op fake close operation."""


@pytest.fixture
def registry() -> RoleRegistry:
    """Load the configured role definitions."""
    return load_roles(ROLES_PATH)


@pytest.fixture
def jinja_env() -> jinja2.Environment:
    """Create a Jinja environment using the application's prompt templates."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader("infra/llm/prompts"),
        autoescape=False,
    )


@pytest.fixture
def in_memory_database() -> Iterator[tuple[sqlite3.Connection, PredictionsRepository]]:
    """Provide a migrated in-memory SQLite database."""
    connection = connect(":memory:")
    apply_migrations(connection)
    try:
        yield connection, PredictionsRepository(connection)
    finally:
        connection.close()


def make_game(
    connection: sqlite3.Connection,
    registry: RoleRegistry,
) -> tuple[int, GameState]:
    """Create an in-memory game with two players and five recorded messages."""
    player_ids = (100, 101)
    game_id = GamesRepository(connection).create_game(chat_id=123, player_count=2)
    players_repository = PlayersRepository(connection)
    for player_id in player_ids:
        players_repository.upsert_player(
            player_id=player_id,
            username=None,
            display_name=f"Player {player_id}",
        )

    game_state = GameState(game_id=game_id, registry=registry)
    for player_id in player_ids:
        game_state.register_player(player_id, f"Player {player_id}")
    for index in range(5):
        game_state.apply(
            MessageEvent(
                ts=BASE_TS + timedelta(seconds=index),
                game_id=game_id,
                payload=MessagePayload(
                    author_id=player_ids[index % len(player_ids)],
                    author_name=f"Player {player_ids[index % len(player_ids)]}",
                    text=f"Game message {index + 1}",
                ),
            )
        )
    return game_id, game_state


def make_response(role_name: str) -> AnalysisResponse:
    """Build one role prediction for the end-to-end test game."""
    return AnalysisResponse(
        analysis=[
            PlayerPrediction(
                player_id=100,
                player_name="Player 100",
                suspected_role=role_name,
                confidence=0.75,
                reasons=["consistent vote", "specific argument"],
            )
        ],
        game_read="The table is taking shape.",
        reasoning="The observed votes support this estimate.",
    )


async def test_e2e_analyze_and_persist_in_memory(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    in_memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Run the complete GameState-to-LLM-to-SQLite prediction path."""
    connection, predictions_repository = in_memory_database
    game_id, game_state = make_game(connection, registry)
    role_name = registry.all_roles()[0].name_ru
    analyzer = GameAnalyzer(
        llm_client=FakeLLMClient(make_response(role_name)),
        predictions_repo=predictions_repository,
        registry=registry,
        jinja_env=jinja_env,
    )

    result = await analyzer.analyze(game_state)

    predictions = predictions_repository.list_predictions(game_id)
    assert result.success is True
    assert result.response is not None
    assert len(predictions) == 1
    assert predictions[0]["player_id"] == 100
    assert predictions[0]["predicted_role"] == role_name
    assert predictions[0]["confidence"] == pytest.approx(0.75)


async def test_e2e_analyze_persist_full_cycle_with_retro_stub(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    in_memory_database: tuple[sqlite3.Connection, PredictionsRepository],
) -> None:
    """Smoke-test the final-role and prediction-correctness database cycle."""
    connection, predictions_repository = in_memory_database
    game_id, game_state = make_game(connection, registry)
    role_name = registry.all_roles()[0].name_ru
    connection.execute(
        "INSERT INTO game_players(game_id, player_id) VALUES (?, ?)",
        (game_id, 100),
    )
    analyzer = GameAnalyzer(
        llm_client=FakeLLMClient(make_response(role_name)),
        predictions_repo=predictions_repository,
        registry=registry,
        jinja_env=jinja_env,
    )

    result = await analyzer.analyze(game_state)

    predictions = predictions_repository.list_predictions(game_id)
    assert result.success is True
    assert len(predictions) == 1
    prediction_id = predictions[0]["id"]
    before_label = connection.execute(
        "SELECT correct FROM predictions WHERE id = ?",
        (prediction_id,),
    ).fetchone()
    assert before_label is not None
    assert before_label["correct"] is None

    connection.execute(
        "UPDATE game_players SET final_role = ? WHERE game_id = ? AND player_id = ?",
        (role_name, game_id, 100),
    )
    connection.execute(
        "UPDATE predictions SET correct = 1 WHERE id = ?",
        (prediction_id,),
    )

    final_role = connection.execute(
        "SELECT final_role FROM game_players WHERE game_id = ? AND player_id = ?",
        (game_id, 100),
    ).fetchone()
    correctness = connection.execute(
        "SELECT correct FROM predictions WHERE id = ?",
        (prediction_id,),
    ).fetchone()
    assert final_role is not None
    assert final_role["final_role"] == role_name
    assert correctness is not None
    assert correctness["correct"] == 1
