"""End-to-end coverage for orchestration through the fake UI transport."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from app.bootstrap import BootstrapResult, bootstrap
from app.orchestrator import Orchestrator
from domain.events import MessageEvent, MessagePayload, PhaseChangeEvent, PhaseChangePayload
from domain.phases import Phase
from features.ui.protocol import ReasoningMessage, StatsMessage
from features.ui.window_host import WindowHost
from tests.unit.app.conftest import FakeLLMClient, FakeTelegram, create_settings_file


class FakeWindowHost:
    """No-op window implementation that records UI frames."""

    def __init__(self) -> None:
        self.stats: list[StatsMessage] = []
        self.reasoning: list[ReasoningMessage] = []

    async def start(self) -> None:
        """Mirror child startup without creating a process."""

    async def send_stats(self, message: StatsMessage) -> None:
        self.stats.append(message)

    async def send_reasoning(self, message: ReasoningMessage) -> None:
        self.reasoning.append(message)

    async def shutdown(self) -> None:
        """Mirror child shutdown without creating a process."""


async def test_e2e_ui_mode_with_fake_window(tmp_path: Path) -> None:
    """Persistence refreshes stats and scanner analysis forwards reasoning."""
    settings_path = create_settings_file(
        tmp_path,
        chat_id=-100555,
        trigger_mode="queue",
        db_path=":memory:",
    )
    # The test is specifically exercising the configured UI-mode activation.
    import yaml

    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    settings["ui"].update({"stats_window": True, "reasoning_window": True})
    settings_path.write_text(
        yaml.safe_dump(settings, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    fake_telegram = FakeTelegram()
    fake_llm = FakeLLMClient()
    bootstrap_result = await bootstrap(
        settings_path,
        telegram=fake_telegram,  # type: ignore[arg-type]
        llm_client=fake_llm,
    )
    assert isinstance(bootstrap_result, BootstrapResult)
    ui = FakeWindowHost()
    orchestrator = Orchestrator(
        bootstrap_result=bootstrap_result,
        window_host=cast(WindowHost, ui),
    )
    message_event = MessageEvent(
        ts=datetime(2026, 11, 1, 20, 0, tzinfo=UTC),
        game_id=bootstrap_result.game_state.game_id,
        payload=MessagePayload(
            author_id=712,
            author_name="Игрок UI",
            text="Предлагаю внимательно слушать аргументы",
        ),
    )

    try:
        await orchestrator._persist_event(message_event)
        bootstrap_result.game_state.apply(
            PhaseChangeEvent(
                ts=message_event.ts,
                game_id=bootstrap_result.game_state.game_id,
                payload=PhaseChangePayload(
                    from_phase=Phase.REGISTRATION, to_phase=Phase.NIGHT, day_number=0
                ),
            )
        )
        scanner = orchestrator._default_scanner(-100555)
        await scanner._flush()
    finally:
        await orchestrator.shutdown()

    assert len(ui.stats) >= 1
    assert len(ui.reasoning) >= 1
    assert "Проверяю игровые события" in ui.reasoning[0].text
