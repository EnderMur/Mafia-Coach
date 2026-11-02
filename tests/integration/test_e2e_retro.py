"""End-to-end tests for the analyzer → predictions → retro cycle (C4)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jinja2
import pytest

from app.bootstrap import BootstrapResult, Repositories
from app.config import load_config
from app.orchestrator import Orchestrator
from domain.events import EndGameEvent, PhaseChangeEvent, PhaseChangePayload
from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles
from features.analyzer.analyzer import GameAnalyzer
from features.memory.retro import retro_analyze
from features.scanner.trigger_policy import TriggerMode, TriggerPolicy
from infra.db.connection import connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.events import EventsRepository
from infra.db.repositories.game_players import GamePlayersRepository
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository
from infra.llm.schemas import AnalysisResponse, PlayerPrediction
from tests.unit.app.conftest import FakeLLMClient, FakeTelegram, create_settings_file

ROOT = Path(__file__).resolve().parents[2]
ROLES_PATH = ROOT / "configs" / "roles.yaml"
CHAT_ID = -1001111111111
BASE_TS = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)


@pytest.fixture(name="registry")
def registry_fixture() -> RoleRegistry:
    """Load the configured role definitions."""
    return load_roles(ROLES_PATH)


@pytest.fixture(name="conn")
def conn_fixture() -> Iterator[sqlite3.Connection]:
    """Provide a migrated in-memory SQLite database."""
    connection = connect(":memory:")
    apply_migrations(connection)
    try:
        yield connection
    finally:
        connection.close()


def stored_correct(conn: sqlite3.Connection, game_id: int) -> list[int | None]:
    """Return the stored correctness labels of one game in insertion order."""
    rows = conn.execute(
        "SELECT correct FROM predictions WHERE game_id = ? ORDER BY id", (game_id,)
    ).fetchall()
    return [row["correct"] for row in rows]


def prepare_full_game(conn: sqlite3.Connection) -> int:
    """Create a finished three-player game with labelled predictions."""
    games_repo = GamesRepository(conn)
    players_repo = PlayersRepository(conn)
    game_players_repo = GamePlayersRepository(conn)
    predictions_repo = PredictionsRepository(conn)

    game_id = games_repo.create_game(chat_id=CHAT_ID, player_count=3)
    final_roles = {101: "Мафия", 102: "Дон", 103: "Мирный житель"}
    predictions = {101: "Мафия", 102: "Дон", 103: "Маньяк"}

    for player_id, final_role in final_roles.items():
        players_repo.upsert_player(
            player_id=player_id, username=None, display_name=f"Player {player_id}"
        )
        game_players_repo.add_player(
            game_id=game_id, player_id=player_id, final_role=final_role, survived=True
        )
    for player_id, predicted_role in predictions.items():
        predictions_repo.insert_prediction(
            game_id=game_id,
            player_id=player_id,
            predicted_role=predicted_role,
            confidence=0.6,
            reasoning="e2e prediction",
        )
    games_repo.finish_game(game_id=game_id, winner_faction="mafia")
    return game_id


def test_e2e_full_retro_cycle(conn: sqlite3.Connection, registry: RoleRegistry) -> None:
    """Predictions are labelled, aggregated, and reported in one pass."""
    game_id = prepare_full_game(conn)

    report = retro_analyze(
        game_id=game_id,
        registry=registry,
        predictions_repo=PredictionsRepository(conn),
        game_players_repo=GamePlayersRepository(conn),
        games_repo=GamesRepository(conn),
    )

    assert report.game_id == game_id
    assert report.total_predictions == 3
    assert report.correct == 2
    assert report.wrong == 1
    assert report.unknown == 0
    assert set(report.by_role) == {"Мафия", "Дон", "Маньяк"}
    assert report.by_role["Мафия"].correct == 1
    assert report.by_role["Дон"].correct == 1
    assert report.by_role["Маньяк"].correct == 0
    assert set(stored_correct(conn, game_id)) == {0, 1}


def test_e2e_retro_no_final_roles(conn: sqlite3.Connection, registry: RoleRegistry) -> None:
    """A finished game without final roles leaves every label untouched."""
    games_repo = GamesRepository(conn)
    players_repo = PlayersRepository(conn)
    game_players_repo = GamePlayersRepository(conn)
    predictions_repo = PredictionsRepository(conn)

    game_id = games_repo.create_game(chat_id=CHAT_ID, player_count=2)
    for player_id in (201, 202):
        players_repo.upsert_player(
            player_id=player_id, username=None, display_name=f"Player {player_id}"
        )
        game_players_repo.add_player(game_id=game_id, player_id=player_id)
    for player_id in (201, 202):
        predictions_repo.insert_prediction(
            game_id=game_id,
            player_id=player_id,
            predicted_role="Мафия",
            confidence=0.4,
            reasoning="e2e prediction without ground truth",
        )

    report = retro_analyze(
        game_id=game_id,
        registry=registry,
        predictions_repo=predictions_repo,
        game_players_repo=game_players_repo,
        games_repo=games_repo,
    )

    assert report.total_predictions == 2
    assert report.unknown == 2
    assert report.correct == 0
    assert report.wrong == 0
    assert report.by_role == {}
    assert stored_correct(conn, game_id) == [None, None]


# ---------------------------------------------------------------------------
# Step 10c-fix-11: END_GAME → auto-retro through the orchestrator
# ---------------------------------------------------------------------------

PLAYER_A = 301
PLAYER_B = 302
CHAT_ID_E2E = -1007654321
BOT_ID = 42

END_GAME_TEXT = """Игра окончена!

Победители:
    1. Вася - 🤵🏼 Мафия

Другие пользователи:
    2. Петя - 👨🏼 Мирный Житель

Игра длилась: 20 минут"""


class FakeMessage:
    """Telegram message subset consumed by MessageScanner."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeSender:
    """Telegram sender subset consumed by MessageScanner."""

    def __init__(self, first_name: str) -> None:
        self.first_name = first_name
        self.last_name: str | None = None


def telegram_event(sender_id: int, name: str, text: str, index: int) -> Any:
    """Build one Telethon-shaped message event for the scanner."""
    return SimpleNamespace(
        sender_id=sender_id,
        sender=FakeSender(name),
        message=FakeMessage(text),
        date=BASE_TS + timedelta(seconds=index),
    )


@dataclass(frozen=True)
class EndGameEnvironment:
    """Fully wired orchestrator over an empty, migrated in-memory database."""

    conn: sqlite3.Connection
    repos: Repositories
    game_state: GameState
    orchestrator: Orchestrator
    game_id: int


def build_end_game_environment(tmp_path: Path) -> EndGameEnvironment:
    """Create the migrated database, analyzer, and orchestrator used by the e2e runs."""
    config = load_config(
        create_settings_file(
            tmp_path,
            chat_id=CHAT_ID_E2E,
            trigger_mode="every_message",
            db_path=":memory:",
        )
    )
    conn = connect(":memory:")
    apply_migrations(conn)
    repos = Repositories(
        games=GamesRepository(conn),
        players=PlayersRepository(conn),
        events=EventsRepository(conn),
        predictions=PredictionsRepository(conn),
        game_players=GamePlayersRepository(conn),
    )
    game_id = repos.games.create_game(chat_id=CHAT_ID_E2E, player_count=2)

    registry = load_roles(config.roles_file)
    game_state = GameState(game_id=game_id, registry=registry)
    response = AnalysisResponse(
        analysis=[
            PlayerPrediction(
                player_id=PLAYER_A,
                player_name="Вася",
                suspected_role="Мафия",
                confidence=0.8,
                reasons=["путается в показаниях"],
            ),
            PlayerPrediction(
                player_id=PLAYER_B,
                player_name="Петя",
                suspected_role="Мирный житель",
                confidence=0.6,
                reasons=["спокойно отвечает"],
            ),
        ],
        game_read="Вася — мафия.",
        reasoning="Вася давит на Петю без аргументов.",
    )
    llm_client = FakeLLMClient(response)
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(ROOT / "infra" / "llm" / "prompts")),
        autoescape=False,
    )
    analyzer = GameAnalyzer(
        llm_client=llm_client,
        predictions_repo=repos.predictions,
        registry=registry,
        jinja_env=jinja_env,
        max_messages=config.llm.context_window_messages,
        reasoning_max_chars=config.llm.reasoning_max_chars,
    )
    bootstrap_result = BootstrapResult(
        config=config,
        conn=conn,
        repos=repos,
        registry=registry,
        llm_client=llm_client,
        telegram=FakeTelegram(),  # type: ignore[arg-type]
        game_state=game_state,
        analyzer=analyzer,
        trigger_policy=TriggerPolicy(mode=TriggerMode.EVERY_MESSAGE),
        jinja_env=jinja_env,
        owns_telegram=False,
        owns_llm=False,
    )
    return EndGameEnvironment(
        conn=conn,
        repos=repos,
        game_state=game_state,
        orchestrator=Orchestrator(bootstrap_result=bootstrap_result, enable_ui=False),
        game_id=game_id,
    )


async def test_e2e_end_game_auto_retro(tmp_path: Path) -> None:
    """13. The bot's final report labels predictions and stores the ground truth."""
    environment = build_end_game_environment(tmp_path)
    repos = environment.repos
    game_state = environment.game_state
    game_id = environment.game_id
    scanner = environment.orchestrator._default_scanner(CHAT_ID_E2E)
    await scanner.start()
    try:
        game_state.apply(
            PhaseChangeEvent(
                ts=BASE_TS,
                game_id=game_id,
                payload=PhaseChangePayload(
                    from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
                ),
            )
        )
        for index, (player_id, name) in enumerate(((PLAYER_A, "Вася"), (PLAYER_B, "Петя"))):
            await scanner._on_event(
                telegram_event(
                    player_id,
                    name,
                    "Я думаю, что кто-то из нас двоих ведёт себя подозрительно",
                    index,
                )
            )

        stored_before = repos.predictions.list_predictions(game_id)
        assert len(stored_before) == 2
        assert {row["correct"] for row in stored_before} == {None}

        for player_id in (PLAYER_A, PLAYER_B):
            repos.game_players.add_player(game_id=game_id, player_id=player_id)

        await scanner._on_event(telegram_event(BOT_ID, "Бот", END_GAME_TEXT, 10))
    finally:
        await scanner.shutdown()

    # --- Ground truth -----------------------------------------------------
    assert game_state.current_phase() is Phase.END
    winner = repos.game_players.get_player_in_game(game_id=game_id, player_id=PLAYER_A)
    loser = repos.game_players.get_player_in_game(game_id=game_id, player_id=PLAYER_B)
    assert winner is not None and loser is not None
    assert winner["final_role"] == "Мафия"
    assert winner["is_winner"] == 1
    assert loser["final_role"] == "Мирный житель"
    assert loser["is_winner"] == 0

    # --- Auto-retro labelled the predictions ------------------------------
    stored_after = repos.predictions.list_predictions(game_id)
    labels = {row["player_id"]: row["correct"] for row in stored_after}
    assert labels == {PLAYER_A: 1, PLAYER_B: 1}

    # --- The final report itself is persisted and round-trips -------------
    stored_events = repos.events.list_events(game_id)
    end_game_events = [event for event in stored_events if event.kind == "end_game"]
    assert len(end_game_events) == 1
    assert isinstance(end_game_events[0], EndGameEvent)
    assert end_game_events[0].payload.winners == [("Вася", "Мафия")]
    assert end_game_events[0].payload.losers == [("Петя", "Мирный Житель")]
    assert end_game_events[0].payload.duration_minutes == 20


# ---------------------------------------------------------------------------
# Step 10c-fix-12: registration alone fills game_players
# ---------------------------------------------------------------------------


async def test_e2e_end_game_auto_retro_with_persist_player(tmp_path: Path) -> None:
    """14. ``_persist_player`` alone fills game_players, so auto-retro labels predictions.

    No manual ``game_players.add_player`` call: the orchestrator's registration
    callback must create the participation rows, otherwise every final role
    write is rejected and ``predictions.correct`` stays NULL.
    """
    environment = build_end_game_environment(tmp_path)
    repos = environment.repos
    game_state = environment.game_state
    game_id = environment.game_id
    orchestrator = environment.orchestrator
    scanner = orchestrator._default_scanner(CHAT_ID_E2E)
    await scanner.start()
    try:
        game_state.apply(
            PhaseChangeEvent(
                ts=BASE_TS,
                game_id=game_id,
                payload=PhaseChangePayload(
                    from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
                ),
            )
        )
        for index, (player_id, name) in enumerate(((PLAYER_A, "Вася"), (PLAYER_B, "Петя"))):
            await scanner._on_event(
                telegram_event(
                    player_id,
                    name,
                    "Я думаю, что кто-то из нас двоих ведёт себя подозрительно",
                    index,
                )
            )

        # --- Registration created the participation rows -------------------
        rows_before = repos.game_players.list_game_players(game_id)
        assert [row["player_id"] for row in rows_before] == [PLAYER_A, PLAYER_B]
        assert {row["final_role"] for row in rows_before} == {None}
        assert len(repos.predictions.list_predictions(game_id)) == 2

        await scanner._on_event(telegram_event(BOT_ID, "Бот", END_GAME_TEXT, 10))
    finally:
        await scanner.shutdown()

    # --- Ground truth is stored on the pre-existing participation rows -----
    winner = repos.game_players.get_player_in_game(game_id=game_id, player_id=PLAYER_A)
    loser = repos.game_players.get_player_in_game(game_id=game_id, player_id=PLAYER_B)
    assert winner is not None and loser is not None
    assert winner["final_role"] == "Мафия"
    assert loser["final_role"] == "Мирный житель"
    assert len(repos.game_players.list_game_players(game_id)) == 2

    # --- Auto-retro filled predictions.correct instead of leaving it NULL --
    labels = {
        row["player_id"]: row["correct"] for row in repos.predictions.list_predictions(game_id)
    }
    assert labels == {PLAYER_A: 1, PLAYER_B: 1}

    await orchestrator.shutdown()
