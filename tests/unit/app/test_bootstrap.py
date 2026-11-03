"""Tests for bootstrap resource construction and teardown."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from loguru import logger

from app.bootstrap import Repositories, bootstrap, shutdown_result
from domain.exceptions import ConfigError
from domain.phases import Phase
from features.scanner.trigger_policy import TriggerMode
from infra.db.connection import close, connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.games import GamesRepository
from tests.unit.app.conftest import FakeLLMClient, FakeTelegram, create_settings_file


def _settings_with_patterns_file(tmp_path: Path, patterns_path: Path) -> Path:
    """Point temporary settings at an isolated patterns file."""
    config_path = create_settings_file(tmp_path)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["club_patterns_file"] = str(patterns_path)
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return config_path


async def test_bootstrap_loads_club_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The primary file wins over an example and flows all the way to the LLM."""
    patterns_path = tmp_path / "patterns.txt"
    patterns_path.write_text("  SYNTHETIC PRIMARY  \n", encoding="utf-8")
    (tmp_path / "patterns.example.txt").write_text("SYNTHETIC EXAMPLE", encoding="utf-8")
    llm_client = FakeLLMClient()
    analyze_stream = AsyncMock(wraps=llm_client.analyze_stream)
    monkeypatch.setattr(llm_client, "analyze_stream", analyze_stream)
    result = await bootstrap(
        _settings_with_patterns_file(tmp_path, patterns_path),
        telegram=FakeTelegram(),  # type: ignore[arg-type]
        llm_client=llm_client,
    )
    try:
        assert result.config.club_patterns_file == str(patterns_path)
        assert result.club_patterns == "SYNTHETIC PRIMARY"
        analysis = await result.analyzer.analyze(result.game_state, persist=False)
        assert analysis.success
        assert analyze_stream.await_args is not None
        prompt = analyze_stream.await_args.kwargs["messages"][1]["content"]
        assert "SYNTHETIC PRIMARY" in prompt
        assert "SYNTHETIC EXAMPLE" not in prompt
    finally:
        await shutdown_result(result)


async def test_bootstrap_falls_back_to_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An example is read with a warning when the configured file is absent."""
    patterns_path = tmp_path / "patterns.txt"
    example_path = tmp_path / "patterns.example.txt"
    example_path.write_text("  SYNTHETIC EXAMPLE  \n", encoding="utf-8")
    monkeypatch.setattr("app.bootstrap.setup_logging", lambda **kwargs: None)
    captured = io.StringIO()
    sink = logger.add(captured, level="WARNING", format="{level}: {message}")
    try:
        result = await bootstrap(
            _settings_with_patterns_file(tmp_path, patterns_path),
            telegram=FakeTelegram(),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
        )
        try:
            assert result.club_patterns == "SYNTHETIC EXAMPLE"
            assert result.analyzer._club_patterns == result.club_patterns
        finally:
            await shutdown_result(result)
    finally:
        logger.remove(sink)

    assert "WARNING: club_patterns: using example file" in captured.getvalue()
    assert str(example_path) in captured.getvalue()
    assert str(patterns_path) in captured.getvalue()
    assert not patterns_path.exists()


async def test_bootstrap_missing_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both files can be missing without preventing bootstrap or inventing patterns."""
    patterns_path = tmp_path / "absent" / "patterns.txt"
    monkeypatch.setattr("app.bootstrap.setup_logging", lambda **kwargs: None)
    captured = io.StringIO()
    sink = logger.add(captured, level="INFO", format="{level}: {message}")
    try:
        result = await bootstrap(
            _settings_with_patterns_file(tmp_path, patterns_path),
            telegram=FakeTelegram(),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
        )
        try:
            assert result.club_patterns == ""
            assert result.analyzer._club_patterns == ""
        finally:
            await shutdown_result(result)
    finally:
        logger.remove(sink)

    assert "INFO: club_patterns: file not found, using empty" in captured.getvalue()
    assert "club_patterns loaded: 0 chars" in captured.getvalue()
    assert not patterns_path.exists()


async def test_bootstrap_creates_all_resources(tmp_path: Path) -> None:
    """Bootstrap returns a fully wired result and applies database migrations."""
    config_path = create_settings_file(tmp_path)
    telegram = FakeTelegram()
    llm_client = FakeLLMClient()

    result = await bootstrap(
        config_path,
        telegram=telegram,  # type: ignore[arg-type]
        llm_client=llm_client,
    )
    try:
        assert result.conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 4
        assert isinstance(result.repos, Repositories)
        assert result.registry.all_ids()
        assert result.llm_client is llm_client
        assert result.telegram.me_id == telegram.me_id
        assert result.game_state.game_id > 0
        assert result.analyzer is not None
        assert result.club_patterns == ""
        assert result.trigger_policy.mode is TriggerMode.QUEUE
        assert result.jinja_env.get_template("system.jinja") is not None
        assert result.owns_llm is False
        assert result.owns_telegram is False
    finally:
        await shutdown_result(result)


async def test_bootstrap_creates_game_if_missing(tmp_path: Path) -> None:
    """An empty database gets a new active game for the configured chat."""
    result = await bootstrap(
        create_settings_file(tmp_path, chat_id=-100777),
        telegram=FakeTelegram(),  # type: ignore[arg-type]
        llm_client=FakeLLMClient(),
    )
    try:
        game = result.repos.games.get_game(result.game_state.game_id)
        assert game["chat_id"] == -100777
        assert game["ended_at"] is None
        assert result.game_state.game_id == game["id"]
    finally:
        await shutdown_result(result)


async def test_bootstrap_reuses_active_game(tmp_path: Path) -> None:
    """An unfinished latest game is reused instead of inserting another."""
    database_path = tmp_path / "mafia.db"
    connection = connect(database_path)
    try:
        apply_migrations(connection)
        existing_game_id = GamesRepository(connection).create_game(
            chat_id=-100778,
            player_count=2,
        )
    finally:
        close(connection)

    result = await bootstrap(
        create_settings_file(tmp_path, chat_id=-100778, db_path=str(database_path)),
        telegram=FakeTelegram(),  # type: ignore[arg-type]
        llm_client=FakeLLMClient(),
    )
    try:
        assert result.game_state.game_id == existing_game_id
        assert len(result.repos.games.list_games()) == 1
    finally:
        await shutdown_result(result)


async def test_bootstrap_skips_game_when_chat_id_none(tmp_path: Path) -> None:
    """Unattached startup uses game id zero until /attach creates a game."""
    result = await bootstrap(
        create_settings_file(tmp_path, chat_id=None),
        telegram=FakeTelegram(),  # type: ignore[arg-type]
        llm_client=FakeLLMClient(),
    )
    try:
        assert result.game_state.game_id == 0
        assert result.repos.games.list_games() == []
    finally:
        await shutdown_result(result)


async def test_bootstrap_validates_chat_when_set(tmp_path: Path) -> None:
    """An inaccessible configured chat is raised as a ConfigError."""
    telegram = FakeTelegram(get_chat_error=RuntimeError("private chat"))

    with pytest.raises(ConfigError, match="cannot access configured Telegram chat"):
        await bootstrap(
            create_settings_file(tmp_path, chat_id=-100779),
            telegram=telegram,  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
        )

    assert telegram.get_chat_calls == [-100779]


async def test_bootstrap_requires_active_game_when_creation_disabled(tmp_path: Path) -> None:
    """Disabling game creation rejects a chat with no unfinished game."""
    with pytest.raises(ConfigError, match="no active game"):
        await bootstrap(
            create_settings_file(tmp_path),
            telegram=FakeTelegram(),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            create_new_game_if_missing=False,
        )


async def test_shutdown_result_closes_owned_resources(tmp_path: Path) -> None:
    """Teardown invokes close only for resources marked as owned."""
    telegram = FakeTelegram()
    llm_client = FakeLLMClient()
    result = await bootstrap(
        create_settings_file(tmp_path),
        telegram=telegram,  # type: ignore[arg-type]
        llm_client=llm_client,
    )
    # The injected test doubles are used to exercise the ownership branch
    # without constructing a real Telethon or LM Studio client.
    result.owns_telegram = True
    result.owns_llm = True

    await shutdown_result(result)

    assert telegram.aclose_calls == 1
    assert llm_client.aclose_calls == 1
    with pytest.raises(sqlite3.ProgrammingError):
        result.conn.execute("SELECT 1")


async def test_shutdown_result_is_idempotent(tmp_path: Path) -> None:
    """Repeated teardown neither fails nor closes owned clients twice."""
    telegram = FakeTelegram()
    llm_client = FakeLLMClient()
    result = await bootstrap(
        create_settings_file(tmp_path),
        telegram=telegram,  # type: ignore[arg-type]
        llm_client=llm_client,
    )
    result.owns_telegram = True
    result.owns_llm = True

    await shutdown_result(result)
    await shutdown_result(result)

    assert telegram.aclose_calls == 1
    assert llm_client.aclose_calls == 1
    assert result.game_state.current_phase() is Phase.REGISTRATION
