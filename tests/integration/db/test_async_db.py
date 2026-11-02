"""Verify asyncio.to_thread keeps event loop responsive during DB writes."""

from __future__ import annotations

import asyncio
import time

import pytest

from infra.db.connection import close, connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository


@pytest.mark.asyncio
async def test_event_loop_not_blocked_during_insert() -> None:
    """While insert_prediction is running in a worker thread,
    asyncio.sleep on the main loop must return promptly
    (event loop is not blocked).
    """
    conn = connect(":memory:")
    apply_migrations(conn)
    try:
        games = GamesRepository(conn)
        players = PlayersRepository(conn)
        predictions = PredictionsRepository(conn)

        gid = await asyncio.to_thread(
            games.create_game,
            chat_id=-100,
            player_count=10,
        )
        await asyncio.to_thread(
            players.upsert_player,
            player_id=1,
            username="u1",
            display_name="u1",
        )

        async def slow_insert() -> int:
            # 20 iterations × ~0.5ms = ~10ms в потоке
            last_id = 0
            for i in range(20):
                last_id = await asyncio.to_thread(
                    predictions.insert_prediction,
                    game_id=gid,
                    player_id=1,
                    predicted_role="Мафия",
                    confidence=0.5,
                    reasoning=f"iter {i}",
                )
            return last_id

        # Параллельно: sleep на 10ms в main loop
        t0 = time.perf_counter()
        results = await asyncio.gather(
            slow_insert(),
            asyncio.sleep(0.01),
        )
        elapsed = max(0.0, time.perf_counter() - t0)

        # insert должен занять >= 10ms (20 × ~0.5ms), sleep — 10ms
        # Если event loop заблокирован — elapsed ≈ sum.
        # Если не заблокирован — elapsed ≈ max ≈ 10ms.
        # Assert: elapsed < 50ms (с запасом на CI-шум).
        assert elapsed < 0.05, (
            f"event loop appears blocked: total {elapsed * 1000:.1f}ms vs expected ~10ms"
        )
        assert results[0] > 0

        rows = await asyncio.to_thread(
            predictions.list_predictions,
            gid,
        )
        assert len(rows) == 1
        assert rows[0]["id"] == results[0]
        assert rows[0]["reasoning"] == "iter 19"
    finally:
        close(conn)
