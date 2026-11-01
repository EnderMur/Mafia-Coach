"""Build and tear down the application's long-lived resources."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
from loguru import logger

from app.config import AppConfig, load_config
from domain.exceptions import ConfigError
from domain.game_state import GameState
from domain.roles import RoleRegistry, load_roles
from features.analyzer.analyzer import GameAnalyzer
from features.scanner.trigger_policy import TriggerMode, TriggerPolicy
from infra.db.connection import close as close_db
from infra.db.connection import connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.events import EventsRepository
from infra.db.repositories.game_players import GamePlayersRepository
from infra.db.repositories.games import GamesRepository
from infra.db.repositories.players import PlayersRepository
from infra.db.repositories.predictions import PredictionsRepository
from infra.llm.client import LLMClient, LMStudioClient
from infra.logging.setup import setup_logging
from infra.telegram.client import TelegramClientWrapper, TelegramConfig


@dataclass
class Repositories:
    """All SQLite repositories sharing the application's connection."""

    games: GamesRepository
    players: PlayersRepository
    events: EventsRepository
    predictions: PredictionsRepository
    game_players: GamePlayersRepository


@dataclass
class BootstrapResult:
    """Initialized dependencies and resources for an application run."""

    config: AppConfig
    conn: sqlite3.Connection
    repos: Repositories
    registry: RoleRegistry
    llm_client: LLMClient
    telegram: TelegramClientWrapper
    game_state: GameState
    analyzer: GameAnalyzer
    trigger_policy: TriggerPolicy
    jinja_env: jinja2.Environment
    owns_telegram: bool
    owns_llm: bool
    _shutdown_complete: bool = field(default=False, init=False, repr=False)


async def bootstrap(
    config_path: str | Path = "configs/settings.yaml",
    *,
    telegram: TelegramClientWrapper | None = None,
    llm_client: LLMClient | None = None,
    create_new_game_if_missing: bool = True,
) -> BootstrapResult:
    """Load settings and initialize database, LLM, Telegram, and analysis.

    Injected clients are borrowed and remain the caller's responsibility. Any
    resource created here is cleaned up if a later initialization step fails.

    Args:
        config_path: Path to ``settings.yaml``.
        telegram: Optional already-created Telegram client for tests or callers.
        llm_client: Optional already-created LLM client for tests or callers.
        create_new_game_if_missing: Create a game if the configured chat has no
            unfinished game in the database.

    Returns:
        A :class:`BootstrapResult` ready for the headless orchestrator.

    Raises:
        ConfigError: If configuration, roles, Telegram session/chat, or active
            game initialization fails.
        LLMConnectionError: If the owned LM Studio client fails its healthcheck.
        TelegramNotConnectedError: If the owned Telegram client cannot connect.
    """
    config = load_config(config_path)
    setup_logging(
        level=config.logging.level,
        log_dir=config.logging.dir,
        rotation=config.logging.rotation,
        retention_days=config.logging.retention_days,
    )
    logger.bind(channel="bootstrap").info("config loaded")

    connection: sqlite3.Connection | None = None
    active_llm: LLMClient | None = None
    active_telegram: TelegramClientWrapper | None = None
    owns_llm = False
    owns_telegram = False

    try:
        registry = load_roles(config.roles_file)
        connection = await asyncio.to_thread(connect, config.db.path)
        if config.db.apply_migrations_on_start:
            await asyncio.to_thread(apply_migrations, connection)

        repos = Repositories(
            games=GamesRepository(connection),
            players=PlayersRepository(connection),
            events=EventsRepository(connection),
            predictions=PredictionsRepository(connection),
            game_players=GamePlayersRepository(connection),
        )

        if llm_client is None:
            active_llm = LMStudioClient(
                base_url=config.llm.base_url,
                model=config.llm.model,
                temperature=config.llm.temperature,
                top_p=config.llm.top_p,
                max_tokens=config.llm.max_tokens,
                request_timeout_sec=config.llm.request_timeout_sec,
                connect_timeout_sec=config.llm.connect_timeout_sec,
                max_retries=config.llm.max_retries,
                retry_backoff_sec=config.llm.retry_backoff_sec,
            )
            owns_llm = True
            await active_llm.healthcheck()
        else:
            active_llm = llm_client

        if telegram is None:
            telegram_config = TelegramConfig(
                session_file=config.telegram.session_file,
                api_id=config.telegram.api_id,
                api_hash=config.telegram.api_hash,
                chat_id=config.telegram.chat_id,
                bot_username=config.telegram.bot_username,
                bot_id=config.telegram.bot_id,
                ignore_own_messages=config.telegram.ignore_own_messages,
            )
            active_telegram = TelegramClientWrapper(config=telegram_config)
            owns_telegram = True
            await active_telegram.connect()
        else:
            active_telegram = telegram

        if config.telegram.chat_id is not None:
            try:
                await active_telegram.get_chat(config.telegram.chat_id)
            except Exception as error:
                raise ConfigError(
                    f"cannot access configured Telegram chat {config.telegram.chat_id}: "
                    f"{type(error).__name__}: {error}"
                ) from error

            recent_games = await asyncio.to_thread(repos.games.list_games, limit=1)
            if recent_games and recent_games[0]["ended_at"] is None:
                game_id = int(recent_games[0]["id"])
            elif create_new_game_if_missing:
                game_id = await asyncio.to_thread(
                    repos.games.create_game,
                    chat_id=config.telegram.chat_id,
                    player_count=0,
                )
            else:
                raise ConfigError("no active game")
        else:
            # The first /attach creates a real game and replaces this placeholder.
            game_id = 0

        game_state = GameState(game_id=game_id, registry=registry)
        jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("infra/llm/prompts"),
            autoescape=False,
        )
        trigger_policy = TriggerPolicy(
            mode=TriggerMode(config.llm.trigger_mode),
            batch_debounce_sec=config.llm.batch_debounce_sec,
            batch_max_messages=config.llm.batch_max_messages,
        )
        analyzer = GameAnalyzer(
            llm_client=active_llm,
            predictions_repo=repos.predictions,
            registry=registry,
            jinja_env=jinja_env,
            max_messages=config.llm.context_window_messages,
            reasoning_max_chars=config.llm.reasoning_max_chars,
        )

        logger.bind(channel="bootstrap").info(
            "bootstrap complete: game_id={}",
            game_id,
        )
        return BootstrapResult(
            config=config,
            conn=connection,
            repos=repos,
            registry=registry,
            llm_client=active_llm,
            telegram=active_telegram,
            game_state=game_state,
            analyzer=analyzer,
            trigger_policy=trigger_policy,
            jinja_env=jinja_env,
            owns_telegram=owns_telegram,
            owns_llm=owns_llm,
        )
    except BaseException:
        await _close_partial_resources(
            connection=connection,
            llm_client=active_llm if owns_llm else None,
            telegram=active_telegram if owns_telegram else None,
        )
        raise


async def shutdown_result(result: BootstrapResult) -> None:
    """Gracefully close resources owned by a bootstrap result, once."""
    if result._shutdown_complete:
        return
    result._shutdown_complete = True

    try:
        if result.owns_telegram:
            try:
                await result.telegram.aclose()
            except Exception as error:  # noqa: BLE001 — continue shutting down
                logger.bind(channel="bootstrap").error(
                    "telegram aclose failed: {}",
                    str(error),
                )
        if result.owns_llm:
            try:
                await result.llm_client.aclose()
            except Exception as error:  # noqa: BLE001 — continue shutting down
                logger.bind(channel="bootstrap").error(
                    "llm aclose failed: {}",
                    str(error),
                )
        try:
            await asyncio.to_thread(close_db, result.conn)
        except Exception as error:  # noqa: BLE001 — shutdown should remain defensive
            logger.bind(channel="bootstrap").error("db close failed: {}", str(error))
    finally:
        logger.bind(channel="bootstrap").info("shutdown complete")


async def _close_partial_resources(
    *,
    connection: sqlite3.Connection | None,
    llm_client: LLMClient | None,
    telegram: TelegramClientWrapper | None,
) -> None:
    """Close only resources created by a bootstrap attempt that failed."""
    if telegram is not None:
        try:
            await telegram.aclose()
        except Exception as error:  # noqa: BLE001 — preserve the startup error
            logger.bind(channel="bootstrap").error(
                "telegram aclose failed during bootstrap: {}",
                str(error),
            )
    if llm_client is not None:
        try:
            await llm_client.aclose()
        except Exception as error:  # noqa: BLE001 — preserve the startup error
            logger.bind(channel="bootstrap").error(
                "llm aclose failed during bootstrap: {}",
                str(error),
            )
    if connection is not None:
        try:
            await asyncio.to_thread(close_db, connection)
        except Exception as error:  # noqa: BLE001 — preserve the startup error
            logger.bind(channel="bootstrap").error(
                "db close failed during bootstrap: {}",
                str(error),
            )
