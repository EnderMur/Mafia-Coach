"""End-to-end coverage for scanner-to-database headless orchestration."""

from __future__ import annotations

import asyncio
import queue
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import jinja2

from app.bootstrap import BootstrapResult, Repositories
from app.config import load_config
from app.orchestrator import Orchestrator
from domain.game_state import GameState
from domain.roles import load_roles
from features.analyzer.analyzer import GameAnalyzer
from features.scanner.scanner import MessageScanner
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
CHAT_ID = -1007654321
BOT_ID = 42
PLAYER_A = 100
PLAYER_B = 101
BASE_TS = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)


class CommandInput:
    """Thread-safe stdin fake that lets the test send /quit after events."""

    def __init__(self) -> None:
        self._lines: queue.Queue[str] = queue.Queue()

    def readline(self) -> str:
        """Block until the test queues a command."""
        return self._lines.get()

    def send(self, line: str) -> None:
        """Queue a command line for the orchestrator's reader."""
        self._lines.put(line)


class FakeMessage:
    """Telegram message subset consumed by MessageScanner."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeSender:
    """Telegram sender subset consumed by MessageScanner."""

    def __init__(self, first_name: str) -> None:
        self.first_name = first_name
        self.last_name = None


def message_event(player_id: int, name: str, text: str, index: int) -> SimpleNamespace:
    """Construct one Telethon-shaped player message for the scanner."""
    return SimpleNamespace(
        sender_id=player_id,
        sender=FakeSender(name),
        message=FakeMessage(text),
        date=BASE_TS + timedelta(seconds=index),
    )


async def test_e2e_headless_pipeline(tmp_path: Path) -> None:
    """Five events flow through scanner callbacks into SQLite and predictions."""
    config_path = create_settings_file(
        tmp_path,
        chat_id=CHAT_ID,
        trigger_mode="every_message",
        db_path=":memory:",
    )
    config = load_config(config_path)
    conn = connect(":memory:")
    apply_migrations(conn)
    repos = Repositories(
        games=GamesRepository(conn),
        players=PlayersRepository(conn),
        events=EventsRepository(conn),
        predictions=PredictionsRepository(conn),
        game_players=GamePlayersRepository(conn),
    )
    game_id = repos.games.create_game(chat_id=CHAT_ID, player_count=0)
    for player_id, name in ((PLAYER_A, "Вася"), (PLAYER_B, "Петя")):
        repos.players.upsert_player(
            player_id=player_id,
            username=None,
            display_name=name,
        )

    registry = load_roles(config.roles_file)
    game_state = GameState(game_id=game_id, registry=registry)
    telegram = FakeTelegram()
    response = AnalysisResponse(
        analysis=[
            PlayerPrediction(
                player_id=PLAYER_A,
                player_name="Вася",
                suspected_role="Мафия",
                confidence=0.7,
                reasons=["несколько раз менял версию"],
            )
        ],
        game_read="Вася вызывает подозрение.",
        reasoning="Проверяю последовательность его ответов.",
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
    trigger_policy = TriggerPolicy(
        mode=TriggerMode(config.llm.trigger_mode),
        batch_debounce_sec=config.llm.batch_debounce_sec,
        batch_max_messages=config.llm.batch_max_messages,
    )
    bootstrap_result = BootstrapResult(
        config=config,
        conn=conn,
        repos=repos,
        registry=registry,
        llm_client=llm_client,
        telegram=telegram,  # type: ignore[arg-type]
        game_state=game_state,
        analyzer=analyzer,
        trigger_policy=trigger_policy,
        jinja_env=jinja_env,
        owns_telegram=False,
        owns_llm=False,
    )
    command_input = CommandInput()
    orchestrator = Orchestrator(
        bootstrap_result=bootstrap_result,
        stdin=command_input,  # type: ignore[arg-type]
    )

    try:
        run_task = asyncio.create_task(orchestrator.run())
        await asyncio.sleep(0)
        scanner = orchestrator._scanner
        assert isinstance(scanner, MessageScanner)
        assert telegram.handler is not None
        assert telegram.handler_chat_id == CHAT_ID

        texts = (
            "Всем привет, предлагаю внимательно слушать друг друга",
            "Я думаю, что Вася слишком быстро обвиняет игроков",
            "Не согласен, давайте сначала разберём аргументы",
            "Петя выглядит спокойно, но его версия непоследовательна",
            "Перед голосованием хочу уточнить, кто за кого голосовал",
        )
        player_messages = (
            (PLAYER_A, "Вася"),
            (PLAYER_B, "Петя"),
            (PLAYER_A, "Вася"),
            (PLAYER_B, "Петя"),
            (PLAYER_A, "Вася"),
        )
        for index, (player, text) in enumerate(zip(player_messages, texts, strict=True)):
            player_id, name = player
            await scanner._on_event(message_event(player_id, name, text, index))

        command_input.send("/quit\n")
        await asyncio.wait_for(run_task, timeout=2.0)

        stored_events = repos.events.list_events(game_id)
        stored_predictions = repos.predictions.list_predictions(game_id)
        assert len(stored_events) == 5
        assert {event.kind for event in stored_events} == {"message"}
        assert len(stored_predictions) == 5
        assert llm_client.calls == 5
    finally:
        await orchestrator.shutdown()
