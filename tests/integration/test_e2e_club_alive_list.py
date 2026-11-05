"""End-to-end regression: a decorated 16-player alive list reaches the LLM.

Reproduces the reported club snapshot (synthetic nicknames only) through
parser → GameState → LLMContext/prompt → GameAnalyzer and asserts that all 16
role slots are distributed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jinja2
import pytest

from domain.events import AliveListEvent, AliveListPayload, MessageEvent, MessagePayload
from domain.game_state import GameState, _placeholder_player_id
from domain.roles import RoleRegistry, load_roles
from features.analyzer.analyzer import GameAnalyzer
from features.parser.bot_messages import BotMessageKind, parse_bot_message
from infra.db.connection import connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository
from infra.llm.client import LLMClient, ReasoningSink
from infra.llm.schemas import AnalysisResponse, PlayerPrediction

ROLES_PATH = Path(__file__).resolve().parents[2] / "configs" / "roles.yaml"
TS = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)

CLUB_TEXT = """Живые игроки:

1. | user_1

3. user_2🏹

4. user_3'

5. 🇦🇪

8. user_4

9. user_5🍒

10. user_6

11. moon

14. 𝖁𝖊𝖑𝖑𝖆𝖓𝖊 ᥫ᭡

15. user_7

16. н 🐮

17. Е А

18. 🕷️ᵗᵏ| user_8🌝🖤

19. mzhaaaa

20. ♯𝅃𝐫᪶⃕𝐚᪶͡𝐲᪶𝆬𝐚᪶𝆬 ✟

21. M•

Из них:  👨🏼‍💼 Адвокат, 🧟 Поджигатель, 👨🏼 Мирный Житель, 🕵🏼 Комиссар Каттани, 🤵🏻 Дон, 🧙🏼 Бомж, 💣 Камикадзе, 🤞🏼 Счастливчик, 🤵🏼 Мафия - 3, 🧙‍ Маг, 💃 Любовница, 👮🏼 Сержант, 🤦🏼 Самоубийца, 🐺 Оборотень

Всего: 16
"""


class RecordingLLM(LLMClient):
    """Return one confident prediction and record the rendered prompt."""

    def __init__(self, response: AnalysisResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def healthcheck(self) -> None:
        """No-op health check."""

    async def analyze(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        silent_player_ids: frozenset[int] = frozenset(),
    ) -> AnalysisResponse:
        """Delegate to the streaming fake."""
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
        """Record the request and return the canned response."""
        self.calls.append(
            {
                "messages": messages,
                "known_player_ids": known_player_ids,
                "role_names_ru": role_names_ru,
            }
        )
        return self._response

    async def aclose(self) -> None:
        """No-op close."""


@pytest.fixture
def registry() -> RoleRegistry:
    """Load the configured role catalogue."""
    return load_roles(ROLES_PATH)


@pytest.fixture
def jinja_env() -> jinja2.Environment:
    """Render the application's real prompt templates."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader("infra/llm/prompts"),
        autoescape=False,
    )


@pytest.fixture
def database() -> Iterator[tuple[Any, GamesRepository, PlayersRepository, PredictionsRepository]]:
    """Provide a migrated in-memory database with repositories."""
    connection = connect(":memory:")
    apply_migrations(connection)
    try:
        yield (
            connection,
            GamesRepository(connection),
            PlayersRepository(connection),
            PredictionsRepository(connection),
        )
    finally:
        connection.close()


async def test_club_snapshot_produces_16_role_predictions(
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    database: tuple[Any, GamesRepository, PlayersRepository, PredictionsRepository],
) -> None:
    """The reported failure case now yields exactly 16 addressed players."""
    _connection, games, players_repo, predictions_repo = database
    game_id = games.create_game(chat_id=-1001, player_count=16)

    parsed = parse_bot_message(CLUB_TEXT, registry=registry)
    assert parsed is not None
    assert parsed.kind is BotMessageKind.ALIVE_LIST
    assert parsed.alive_player_names is not None
    assert len(parsed.alive_player_names) == 16
    assert parsed.role_count is not None
    assert sum(parsed.role_count.values()) == 16

    state = GameState(game_id=game_id, registry=registry)
    state.apply(
        AliveListEvent(
            ts=TS,
            game_id=game_id,
            payload=AliveListPayload(
                alive_player_names=parsed.alive_player_names,
                role_count=parsed.role_count,
                total_players=parsed.total_players,
            ),
        )
    )

    ids = [_placeholder_player_id(name) for name in parsed.alive_player_names]
    for name, player_id in zip(parsed.alive_player_names, ids, strict=True):
        players_repo.upsert_player(player_id=player_id, username=None, display_name=name)
        state.apply(
            MessageEvent(
                ts=TS,
                game_id=game_id,
                payload=MessagePayload(
                    author_id=player_id,
                    author_name=name,
                    text=f"сообщение от {name}",
                ),
            )
        )

    first_id = ids[0]
    llm = RecordingLLM(
        AnalysisResponse(
            analysis=[
                PlayerPrediction(
                    player_id=first_id,
                    player_name=parsed.alive_player_names[0],
                    suspected_role="Мафия",
                    confidence=0.6,
                    reason="Активно давит в первый день.",
                    reasons=["активно давит в первый день"],
                )
            ],
            game_read="Состав партии получен от бота.",
            reasoning="Распределяю слоты по живым игрокам.",
        )
    )
    analyzer = GameAnalyzer(
        llm_client=llm,
        predictions_repo=predictions_repo,
        registry=registry,
        jinja_env=jinja_env,
    )

    result = await analyzer.analyze(state, persist=True)

    assert result.success and result.response is not None
    assert len(result.response.analysis) == 16
    assert set(llm.calls[0]["known_player_ids"]) == set(ids)

    used: dict[str, int] = {}
    for prediction in result.response.analysis:
        used[prediction.suspected_role] = used.get(prediction.suspected_role, 0) + 1
        assert prediction.player_id in ids
        assert prediction.reasons
    assert used == parsed.role_count
    assert sum("composition_slot_filled" in warning for warning in result.warnings) == 15

    persisted = predictions_repo.list_predictions(game_id)
    assert len(persisted) == 16
    assert sum(1 for row in persisted if str(row["predicted_role"]) == "Мафия") == 3

    prompt = llm.calls[0]["messages"][1]["content"]
    assert "Всего ролевых слотов (сумма счётчиков): 16" in prompt
    assert "Разных ролей: 14" in prompt
    assert "- Мафия — 3" in prompt
    assert "РОВНО 16 записей" in prompt
    assert "Живые игроки из сообщения бота (16)" in prompt
    assert "1 из 30" not in prompt
