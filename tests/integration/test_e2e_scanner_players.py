"""Offline scanner-to-SQLite persistence with foreign keys enabled."""

from pathlib import Path

from app.bootstrap import bootstrap
from app.orchestrator import Orchestrator
from infra.llm.schemas import AnalysisResponse, PlayerPrediction
from tests.integration.test_e2e_scanner import bot_event, player_event
from tests.unit.app.conftest import FakeLLMClient, FakeTelegram, create_settings_file


async def test_e2e_players_persisted_and_predictions_save(tmp_path: Path) -> None:
    telegram = FakeTelegram()
    llm = FakeLLMClient(
        AnalysisResponse(
            analysis=[
                PlayerPrediction(
                    player_id=100,
                    player_name="Вася",
                    suspected_role="Мафия",
                    confidence=0.7,
                    reasons=["подозрительно"],
                ),
                PlayerPrediction(
                    player_id=101,
                    player_name="Петя",
                    suspected_role="Мафия",
                    confidence=0.6,
                    reasons=["молчит"],
                ),
            ],
            game_read="Игра продолжается",
            reasoning="Разбор",
        )
    )
    result = await bootstrap(
        create_settings_file(tmp_path, db_path=":memory:"),
        telegram=telegram,  # type: ignore[arg-type]
        llm_client=llm,
    )
    orchestrator = Orchestrator(bootstrap_result=result, enable_ui=False)
    scanner = orchestrator._default_scanner(-100123)
    try:
        await scanner._on_event(player_event(100, "Вася", "Я мирный житель", second=0))
        await scanner._on_event(player_event(101, "Петя", "Я тоже мирный житель", second=1))
        phase_event = bot_event("🌙 Наступает ночь", second=2)
        phase_event.sender_id = 42
        await scanner._on_event(phase_event)
        await scanner._flush()
        assert result.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert result.conn.execute("SELECT count(*) FROM players").fetchone()[0] == 2
        assert len(result.repos.predictions.list_predictions(result.game_state.game_id)) == 2
        assert len(result.repos.events.list_events(result.game_state.game_id)) == 3
    finally:
        await scanner.shutdown()
        await orchestrator.shutdown()
