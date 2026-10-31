"""Read-only Telethon userbot wrapper for observing a Mafia game chat."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from telethon import TelegramClient, events  # type: ignore[import-untyped]
from telethon.errors import RPCError  # type: ignore[import-untyped]
from telethon.tl.types import User  # type: ignore[import-untyped]

from domain.exceptions import ConfigError, TelegramNotConnectedError
from infra.logging.setup import get_logger
from infra.telegram.session import ensure_session_dir, session_exists


class NewMessageHandler(Protocol):
    """Asynchronous callback invoked for each incoming new-message event."""

    def __call__(self, event: events.NewMessage.Event, /) -> Awaitable[None]:
        """Handle a single Telethon NewMessage event."""


class TelegramConfig(BaseModel):
    """Immutable validated configuration for the read-only Telegram observer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_file: str = Field(min_length=1)
    api_id: int = Field(gt=0)
    api_hash: str = Field(min_length=16)
    chat_id: int | None = None
    bot_username: str | None = None
    bot_id: int | None = Field(default=None, gt=0)
    ignore_own_messages: bool = True

    @model_validator(mode="after")
    def _at_least_one_bot_identifier(self) -> Self:
        """Require either bot_username or bot_id to identify the game bot."""
        has_username = self.bot_username is not None and bool(self.bot_username.strip())
        if not has_username and self.bot_id is None:
            raise ValueError("telegram.bot_username or telegram.bot_id required")
        if not self.session_file.strip():
            raise ValueError("telegram.session_file must not be blank")
        if not self.api_hash.strip():
            raise ValueError("telegram.api_hash must not be blank")
        return self

    @model_validator(mode="after")
    def _chat_or_none(self) -> Self:
        """Allow chat_id to remain None until attached by the orchestrator at runtime."""
        return self


class TelegramClientWrapper:
    """Read-only Telethon userbot wrapper.

    THIS CLASS MUST NEVER EXPOSE outgoing messaging, editing, deletion,
    polling, or reaction operations. The read-only audit (unit tests and AST
    scanner) enforces that this wrapper strictly observes incoming data.
    """

    def __init__(
        self,
        *,
        config: TelegramConfig,
        client: TelegramClient | None = None,
    ) -> None:
        """Initialize the read-only wrapper and verify that the session file exists.

        Args:
            config: Validated Telegram configuration.
            client: Optional pre-built Telethon client (used in unit tests).

        Raises:
            ConfigError: If the configured session file does not exist on disk.
        """
        self._config = config
        ensure_session_dir(config.session_file)
        if not session_exists(config.session_file):
            raise ConfigError(
                f"session file missing: {config.session_file}. "
                "Run `python scripts/make_session.py` first."
            )
        self._client: TelegramClient = client or TelegramClient(
            session=str(config.session_file),
            api_id=config.api_id,
            api_hash=config.api_hash,
        )
        self._owns_client = client is None
        self._connected = False
        self._closed = False
        self._me_id: int | None = None
        self._bot_id: int | None = None
        self._handlers: list[Any] = []

    @property
    def config(self) -> TelegramConfig:
        """Return the immutable Telegram configuration."""
        return self._config

    @property
    def is_connected(self) -> bool:
        """Return True when the wrapper has an active authorized connection."""
        return self._connected

    @property
    def me_id(self) -> int | None:
        """Return the resolved Telegram user id of the userbot account, if known."""
        return self._me_id

    @property
    def bot_id(self) -> int | None:
        """Return the resolved Telegram user id of the game bot, if connected."""
        return self._bot_id

    async def connect(self) -> None:
        """Connect to Telegram, verify session authorization, and resolve user/bot IDs.

        Raises:
            TelegramNotConnectedError: If transport fails or the session is unauthorized.
            ConfigError: If the configured game bot cannot be resolved.
        """
        if self._connected:
            return
        try:
            await self._client.connect()
            authorized = bool(await self._client.is_user_authorized())
        except (RPCError, OSError) as exc:
            raise TelegramNotConnectedError(f"failed to connect to Telegram: {exc}") from exc

        if not authorized:
            raise TelegramNotConnectedError(
                "session is not authorized. Run `python scripts/make_session.py`."
            )

        self._me_id = await self._resolve_me_id()
        self._bot_id = await self._resolve_bot_id()
        if self._bot_id is None:
            raise ConfigError("cannot resolve bot_username/bot_id; check settings.yaml")

        self._connected = True
        self._closed = False
        get_logger("telegram").bind(channel="telegram").info("connected to Telegram (read-only)")

    async def disconnect(self) -> None:
        """Disconnect the underlying Telethon client if currently connected."""
        if self._connected:
            await self._disconnect_transport()
            self._connected = False

    async def get_chat(self, chat_ref: int | str) -> Any:
        """Resolve a chat entity by numeric id or ``@username``.

        Args:
            chat_ref: Numeric chat identifier or username string.

        Returns:
            The resolved Telethon entity.

        Raises:
            TelegramNotConnectedError: If called before :meth:`connect`.
            ConfigError: If Telethon cannot resolve the target chat.
        """
        if not self._connected:
            raise TelegramNotConnectedError("Telegram client is not connected")
        try:
            return await self._client.get_entity(chat_ref)
        except (RPCError, ValueError) as exc:
            raise ConfigError(f"cannot resolve chat {chat_ref}: {exc}") from exc

    def add_message_handler(
        self,
        handler: NewMessageHandler,
        *,
        chat_id: int,
        bot_only: bool = False,
    ) -> None:
        """Register a new-message handler for a specific chat.

        Only ``events.NewMessage`` is subscribed; edited or deleted messages are
        intentionally ignored per project rules.

        Args:
            handler: Async callback invoked on each matching new message.
            chat_id: Target chat identifier to observe.
            bot_only: When True, restrict events to messages sent by the game bot.

        Raises:
            TelegramNotConnectedError: If called before :meth:`connect`.
        """
        if not self._connected:
            raise TelegramNotConnectedError("Telegram client is not connected")

        from_users = [self._bot_id] if bot_only and self._bot_id is not None else None
        if self._config.ignore_own_messages:
            builder = events.NewMessage(
                chats=[chat_id],
                incoming=True,
                from_users=from_users,
                func=self._accept_event,
            )
        else:
            builder = events.NewMessage(
                chats=[chat_id],
                from_users=from_users,
            )

        self._client.add_event_handler(handler, builder)
        self._handlers.append(builder)

    async def list_recent_messages(
        self,
        *,
        chat_id: int,
        limit: int = 50,
    ) -> list[Any]:
        """Fetch up to ``limit`` recent messages from ``chat_id`` (read-only backfill).

        Args:
            chat_id: Target chat identifier to read from.
            limit: Maximum number of messages to retrieve.

        Returns:
            List of Telethon message objects ordered as yielded by Telethon.

        Raises:
            TelegramNotConnectedError: If called before :meth:`connect` or on RPC failure.
        """
        if not self._connected:
            raise TelegramNotConnectedError("Telegram client is not connected")
        if limit <= 0:
            return []

        messages: list[Any] = []
        try:
            async for message in self._client.iter_messages(chat_id, limit=limit):
                messages.append(message)
        except RPCError as exc:
            raise TelegramNotConnectedError(f"failed to list recent messages: {exc}") from exc
        return messages

    async def aclose(self) -> None:
        """Close the underlying Telethon client if owned by this wrapper."""
        if self._closed:
            return
        self._closed = True
        self._connected = False
        if self._owns_client:
            await self._disconnect_transport()

    async def _disconnect_transport(self) -> None:
        """Await Telethon's disconnect result when it returns an awaitable."""
        result = self._client.disconnect()
        if inspect.isawaitable(result):
            await result

    async def _resolve_me_id(self) -> int | None:
        """Resolve the userbot's own Telegram user id, returning None on RPC failure."""
        try:
            me = await self._client.get_me()
        except RPCError:
            return None
        if me is not None and isinstance(me.id, int):
            return me.id
        return None

    async def _resolve_bot_id(self) -> int | None:
        """Resolve the game bot's numeric user id from config or @username."""
        if self._config.bot_id is not None:
            return self._config.bot_id
        bot_username = self._config.bot_username
        if bot_username is None:  # pragma: no cover
            return None
        try:
            entity = await self._client.get_entity(bot_username)
        except (RPCError, ValueError):
            return None
        if isinstance(entity, User) and isinstance(entity.id, int):
            return entity.id
        return None

    def _accept_event(self, event: Any) -> bool:
        """Return False for userbot-authored messages when ignore_own_messages is enabled."""
        if getattr(event, "out", False):
            return False
        return not (self._me_id is not None and getattr(event, "sender_id", None) == self._me_id)
