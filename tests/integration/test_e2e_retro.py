"""End-to-end tests for the analyzer → predictions → retro cycle (C4)."""

from __future__ import annotations

import asyncio
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


async def wait_for_scheduled_retro(orchestrator: Orchestrator) -> None:
    """Await the deferred END_GAME retro before checking labelled predictions.

    Step 11b-fix-3 schedules retro a few seconds after the final report so
    the final analysis flush can persist its predictions first. Tests shorten
    the delay through ``app.orchestrator._RETRO_DELAY_SEC``.
    """
    task = orchestrator._pending_retro_task
    assert task is not None
    await asyncio.wait_for(task, timeout=2.0)


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
                reason="Путаница в показаниях днём и ночью.",
                reasons=["путается в показаниях"],
            ),
            PlayerPrediction(
                player_id=PLAYER_B,
                player_name="Петя",
                suspected_role="Мирный житель",
                confidence=0.6,
                reason="Спокойно отвечает, без подозрительных сдвигов.",
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


async def test_e2e_end_game_auto_retro(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """13. The bot's final report labels predictions and stores the ground truth."""
    monkeypatch.setattr("app.orchestrator._RETRO_DELAY_SEC", 0.01)
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

        stored_before = repos.predictions.list_predictions(game_id)
        assert len(stored_before) == 2
        assert {row["correct"] for row in stored_before} == {None}

        for player_id in (PLAYER_A, PLAYER_B):
            repos.game_players.add_player(game_id=game_id, player_id=player_id)

        await scanner._on_event(telegram_event(BOT_ID, "Бот", END_GAME_TEXT, 10))
        await wait_for_scheduled_retro(orchestrator)
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


async def test_e2e_end_game_auto_retro_with_persist_player(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """14. ``_persist_player`` alone fills game_players, so auto-retro labels predictions.

    No manual ``game_players.add_player`` call: the orchestrator's registration
    callback must create the participation rows, otherwise every final role
    write is rejected and ``predictions.correct`` stays NULL.
    """
    monkeypatch.setattr("app.orchestrator._RETRO_DELAY_SEC", 0.01)
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
        await wait_for_scheduled_retro(orchestrator)
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


async def test_two_games_without_reset_keep_data_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GAME_START создаёт отдельную партию и завершает ретроспективу предыдущей."""
    monkeypatch.setattr("app.orchestrator._RETRO_DELAY_SEC", 0.01)
    env = build_end_game_environment(tmp_path)
    orch = env.orchestrator
    scanner = orch._default_scanner(CHAT_ID_E2E)
    orch._scanner = scanner
    await scanner.start()
    try:
        start = "Город ветров начинает свою игру над жителями!"
        await scanner._on_event(telegram_event(BOT_ID, "Бот", start, 1))
        assert orch._result.game_state.game_id == env.game_id
        env.game_state.apply(
            PhaseChangeEvent(
                ts=BASE_TS,
                game_id=env.game_id,
                payload=PhaseChangePayload(
                    from_phase=Phase.REGISTRATION, to_phase=Phase.DAY, day_number=1
                ),
            )
        )
        for i, (pid, name) in enumerate(((PLAYER_A, "Вася"), (PLAYER_B, "Петя"))):
            await scanner._on_event(
                telegram_event(pid, name, "Я думаю, что это подозрительно", 2 + i)
            )
        await scanner._on_event(telegram_event(BOT_ID, "Бот", END_GAME_TEXT, 5))
        await scanner._on_event(telegram_event(BOT_ID, "Бот", start, 6))
        new_state = orch._result.game_state
        assert new_state.game_id != env.game_id
        previous = env.repos.games.get_game(env.game_id)
        assert previous["ended_at"] is not None
        assert previous["winner_faction"] == "mafia"
        assert not new_state.players()
        assert len(new_state.events_by_kind("phase_change")) == 1
        assert new_state.events()[0].game_id == new_state.game_id
        new_state.apply(
            PhaseChangeEvent(
                ts=BASE_TS,
                game_id=new_state.game_id,
                payload=PhaseChangePayload(
                    from_phase=Phase.REGISTRATION, to_phase=Phase.DAY, day_number=1
                ),
            )
        )
        await scanner._on_event(telegram_event(PLAYER_A, "Вася", "Во второй игре я мирный", 7))
        old_player = env.repos.game_players.get_player_in_game(
            game_id=env.game_id, player_id=PLAYER_A
        )
        new_player = env.repos.game_players.get_player_in_game(
            game_id=new_state.game_id, player_id=PLAYER_A
        )
        assert old_player is not None and old_player["final_role"] == "Мафия"
        assert new_player is not None and new_player["final_role"] is None
        assert env.repos.predictions.list_predictions(env.game_id)
        assert env.repos.predictions.list_predictions(new_state.game_id)
        ids = env.conn.execute("SELECT DISTINCT game_id FROM events").fetchall()
        assert {row[0] for row in ids} == {env.game_id, new_state.game_id}
        assert all(event.game_id == new_state.game_id for event in new_state.events())
    finally:
        await scanner.shutdown()
        env.conn.close()


@pytest.mark.parametrize(
    ("winners", "expected"),
    [
        ([("А", "Доктор"), ("Б", "Мирный")], "town"),
        ([("А", "Дон"), ("Б", "Мафия")], "mafia"),
        ([("А", "Маньяк")], "neutral"),
        ([("А", "Доктор"), ("Б", "Маньяк")], "mixed"),
        ([("А", "Мафия"), ("Б", "Неизвестная роль")], "unknown"),
        ([], "unknown"),
    ],
)
async def test_end_game_winner_faction(
    tmp_path: Path, winners: list[tuple[str, str]], expected: str
) -> None:
    from domain.events import EndGamePayload

    env = build_end_game_environment(tmp_path)
    event = EndGameEvent(ts=BASE_TS, game_id=env.game_id, payload=EndGamePayload(winners=winners))
    try:
        await env.orchestrator._persist_event(event)
        game = env.repos.games.get_game(env.game_id)
        assert game["winner_faction"] == expected
        assert game["ended_at"] is not None
    finally:
        await env.orchestrator.shutdown()


async def test_start_closes_abandoned_game_and_resets_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    retro = Mock()
    monkeypatch.setattr("features.memory.retro.retro_analyze", retro)
    env = build_end_game_environment(tmp_path)
    orch = env.orchestrator
    scanner = orch._default_scanner(CHAT_ID_E2E)
    orch._scanner = scanner
    await scanner.start()
    try:
        await scanner._on_event(telegram_event(PLAYER_A, "Вася", "Привет", 1))
        env.game_state.apply(
            PhaseChangeEvent(
                ts=BASE_TS,
                game_id=env.game_id,
                payload=PhaseChangePayload(
                    from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
                ),
            )
        )
        await scanner._on_event(telegram_event(BOT_ID, "Бот", "Город ветров начинает игру", 2))
        row = env.repos.games.get_game(env.game_id)
        assert row["ended_at"] is not None and row["winner_faction"] is None
        state = orch._result.game_state
        assert state.game_id != env.game_id
        assert not state.players() and not state.registered_ids
        assert state.resolve_name("Вася") is None
        assert state.current_day() == 0 and state.current_phase() is Phase.REGISTRATION
        retro.assert_not_called()
    finally:
        await orch.shutdown()


async def test_duplicate_start_does_not_close_or_reset(tmp_path: Path) -> None:
    env = build_end_game_environment(tmp_path)
    orch = env.orchestrator
    scanner = orch._default_scanner(CHAT_ID_E2E)
    orch._scanner = scanner
    await scanner.start()
    try:
        start = "Город ветров начинает игру"
        await scanner._on_event(telegram_event(BOT_ID, "Бот", start, 1))
        await scanner._on_event(telegram_event(PLAYER_A, "Вася", "Привет", 2))
        await scanner._on_event(telegram_event(BOT_ID, "Бот", start, 3))
        assert orch._result.game_state.game_id == env.game_id
        assert env.repos.games.get_game(env.game_id)["ended_at"] is None
        assert orch._result.game_state.player(PLAYER_A).display_name == "Вася"
        assert scanner.name_to_id["Вася"] == PLAYER_A
    finally:
        await orch.shutdown()


async def test_unknown_to_registration_keeps_bootstrap_game(tmp_path: Path) -> None:
    env = build_end_game_environment(tmp_path)
    orch = env.orchestrator
    scanner = orch._default_scanner(CHAT_ID_E2E)
    orch._scanner = scanner
    await scanner.start()
    try:
        env.game_state.apply(
            PhaseChangeEvent(
                ts=BASE_TS,
                game_id=env.game_id,
                payload=PhaseChangePayload(
                    from_phase=Phase.REGISTRATION, to_phase=Phase.UNKNOWN, day_number=0
                ),
            )
        )
        await scanner._on_event(telegram_event(BOT_ID, "Бот", "Город ветров начинает игру", 1))
        assert orch._result.game_state.game_id == env.game_id
        assert len(env.repos.games.list_games()) == 1
    finally:
        await orch.shutdown()


async def test_late_end_game_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    from loguru import logger

    from domain.events import EndGamePayload

    env = build_end_game_environment(tmp_path)
    orch = env.orchestrator
    scanner = orch._default_scanner(CHAT_ID_E2E)
    orch._scanner = scanner
    await scanner.start()
    retro = Mock()
    monkeypatch.setattr("features.memory.retro.retro_analyze", retro)
    logs: list[str] = []
    sink = logger.add(lambda message: logs.append(str(message)), level="WARNING")
    try:
        env.game_state.apply(
            PhaseChangeEvent(
                ts=BASE_TS,
                game_id=env.game_id,
                payload=PhaseChangePayload(
                    from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
                ),
            )
        )
        await scanner._on_event(telegram_event(BOT_ID, "Бот", "Город ветров начинает игру", 1))
        old_events = env.repos.events.list_events(env.game_id)
        current = orch._result.game_state
        event = EndGameEvent(
            ts=BASE_TS, game_id=env.game_id, payload=EndGamePayload(winners=[("А", "Мафия")])
        )
        await orch._persist_event(event)
        assert env.repos.events.list_events(env.game_id) == old_events
        assert env.repos.games.get_game(env.game_id)["winner_faction"] is None
        assert orch._result.game_state is current and current.current_phase() is Phase.REGISTRATION
        assert any(
            f"event game_id={env.game_id}, current game_id={current.game_id}" in msg for msg in logs
        )
        retro.assert_not_called()
    finally:
        logger.remove(sink)
        await orch.shutdown()
