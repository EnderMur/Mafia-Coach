"""Read-only Telethon userbot wrapper for observing a Mafia game chat."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from telethon import TelegramClient, events  # type: ignore[import-untyped]
from telethon.errors import (  # type: ignore[import-untyped]
    ChannelInvalidError,
    ChannelPrivateError,
    ChatAdminRequiredError,
    FloodWaitError,
    RPCError,
    UserNotParticipantError,
)
from telethon.tl.types import User  # type: ignore[import-untyped]

from domain.exceptions import (
    ConfigError,
    TelegramAccessError,
    TelegramFloodWaitError,
    TelegramNotConnectedError,
)
from infra.logging.setup import get_logger
from infra.telegram.session import ensure_session_dir, session_exists

#: Upper bound for the in-memory de-duplication window of ``(chat_id, msg_id)`` keys.
_SEEN_IDS_LIMIT = 10_000

#: Telethon access errors that mean "this account may not read that chat".
_ACCESS_ERRORS = (
    ChannelPrivateError,
    ChatAdminRequiredError,
    ChannelInvalidError,
    UserNotParticipantError,
)


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
        self._handlers: list[tuple[Any, Any]] = []
        self._seen_ids: set[tuple[int, int]] = set()
        self._edited_warn_handler: Any | None = None

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

        Also registers a single log-only watcher for edited messages: the parser
        never consumes them, we only warn that they were ignored.

        Raises:
            TelegramNotConnectedError: If transport fails or the session is unauthorized.
            TelegramFloodWaitError: If Telegram rate-limits the connection attempt.
            TelegramAccessError: If Telegram refuses access to the authorized account.
            ConfigError: If the configured game bot cannot be resolved.
        """
        if self._connected:
            return
        try:
            await self._client.connect()
            authorized = bool(await self._client.is_user_authorized())
        except FloodWaitError as exc:
            wait = int(getattr(exc, "seconds", 0) or 0)
            get_logger("telegram").bind(channel="telegram").warning("flood wait: {} seconds", wait)
            raise TelegramFloodWaitError(wait) from exc
        except _ACCESS_ERRORS as exc:
            raise TelegramAccessError(f"access denied: {type(exc).__name__}: {exc}") from exc
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
        self._register_edited_warning_handler()
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
            TelegramFloodWaitError: If Telegram rate-limits the resolution request.
            TelegramAccessError: If the chat is private or the account lacks access.
            ConfigError: If Telethon cannot resolve the target chat.
        """
        if not self._connected:
            raise TelegramNotConnectedError("Telegram client is not connected")
        try:
            return await self._client.get_entity(chat_ref)
        except FloodWaitError as exc:
            wait = int(getattr(exc, "seconds", 0) or 0)
            get_logger("telegram").bind(channel="telegram").warning("flood wait: {} seconds", wait)
            raise TelegramFloodWaitError(wait) from exc
        except _ACCESS_ERRORS as exc:
            raise TelegramAccessError(f"access denied: {type(exc).__name__}: {exc}") from exc
        except (RPCError, ValueError) as exc:
            raise ConfigError(f"cannot resolve chat {chat_ref}: {exc}") from exc

    async def iter_dialogs(
        self,
        *,
        dialogs_limit: int = 200,
        groups_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Return recent dialogs as plain dicts (id, title, kind).

        Used by the /attach command to let the user choose a chat.
        ``groups_only=True`` keeps only groups/channels/supergroups; bots and
        private user chats are excluded from attach.

        Read-only: does not join, send, or modify anything.

        Args:
            dialogs_limit: Maximum number of dialogs to scan from Telegram.
            groups_only: When True, drop anything that is neither group nor channel.

        Returns:
            List of ``{"id": int, "title": str, "kind": str}`` dicts, where
            ``kind`` is one of ``"group"``, ``"channel"`` or ``"other"``.

        Raises:
            TelegramNotConnectedError: If called before :meth:`connect`.
            TelegramFloodWaitError: If Telegram rate-limits the dialog listing.
            TelegramAccessError: If Telegram refuses access to the dialog list.
        """
        if not self._connected:
            raise TelegramNotConnectedError("Telegram client is not connected")

        out: list[dict[str, Any]] = []
        try:
            async for dialog in self._client.iter_dialogs(limit=dialogs_limit):
                is_group = bool(getattr(dialog, "is_group", False))
                is_channel = bool(getattr(dialog, "is_channel", False))
                kind = "channel" if is_channel else ("group" if is_group else "other")
                if groups_only and kind == "other":
                    continue
                out.append(
                    {
                        "id": int(dialog.id),
                        "title": str(getattr(dialog, "title", "") or ""),
                        "kind": kind,
                    }
                )
        except FloodWaitError as exc:
            wait = int(getattr(exc, "seconds", 0) or 0)
            get_logger("telegram").bind(channel="telegram").warning("flood wait: {} seconds", wait)
            raise TelegramFloodWaitError(wait) from exc
        except _ACCESS_ERRORS as exc:
            raise TelegramAccessError(f"access denied: {type(exc).__name__}: {exc}") from exc
        except RPCError as exc:
            raise ConfigError(f"telegram RPC error: {type(exc).__name__}: {exc}") from exc

        get_logger("telegram").bind(channel="telegram").info("iter_dialogs: {} found", len(out))
        return out

    def add_message_handler(
        self,
        handler: NewMessageHandler,
        *,
        chat_id: int,
        bot_only: bool = False,
        deduplicate: bool = True,
    ) -> None:
        """Register a new-message handler for a specific chat.

        Only ``events.NewMessage`` is subscribed; edited or deleted messages are
        intentionally ignored per project rules (edited events are logged only).
        By default the handler is wrapped with a ``(chat_id, message_id)``
        de-duplicator so Telegram redeliveries never reach the parser twice.

        Args:
            handler: Async callback invoked on each matching new message.
            chat_id: Target chat identifier to observe.
            bot_only: When True, restrict events to messages sent by the game bot.
            deduplicate: When True, skip events already seen for this chat.

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

        wrapped = self._make_deduplicating_handler(handler, deduplicate=deduplicate)
        self._client.add_event_handler(wrapped, builder)
        self._handlers.append((handler, wrapped))

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
            TelegramNotConnectedError: If called before :meth:`connect` or on generic RPC failure.
            TelegramFloodWaitError: If Telegram rate-limits the history request.
            TelegramAccessError: If the account may not read that chat.
        """
        if not self._connected:
            raise TelegramNotConnectedError("Telegram client is not connected")
        if limit <= 0:
            return []

        messages: list[Any] = []
        try:
            async for message in self._client.iter_messages(chat_id, limit=limit):
                messages.append(message)
        except FloodWaitError as exc:
            wait = int(getattr(exc, "seconds", 0) or 0)
            get_logger("telegram").bind(channel="telegram").warning("flood wait: {} seconds", wait)
            raise TelegramFloodWaitError(wait) from exc
        except _ACCESS_ERRORS as exc:
            raise TelegramAccessError(f"access denied: {type(exc).__name__}: {exc}") from exc
        except RPCError as exc:
            raise TelegramNotConnectedError(f"failed to list recent messages: {exc}") from exc
        return messages

    async def remove_all_handlers(self) -> None:
        """Remove all handlers previously registered via add_message_handler.

        Also removes the internal edited-message warning handler and clears the
        de-duplication window. Safe to call multiple times, and safe to call
        while disconnected.

        Cleanup failures are swallowed so shutdown never breaks.
        """
        for _user_handler, wrapped in self._handlers:
            await self._safe_remove_handler(wrapped)
        self._handlers.clear()
        if self._edited_warn_handler is not None:
            await self._safe_remove_handler(self._edited_warn_handler)
            self._edited_warn_handler = None
        self._seen_ids.clear()

    async def aclose(self) -> None:
        """Close the underlying Telethon client if owned by this wrapper."""
        if self._closed:
            return
        self._closed = True
        self._connected = False
        if self._owns_client:
            await self._disconnect_transport()

    async def _safe_remove_handler(self, handler: Any) -> None:
        """Detach a single Telethon event handler, never propagating cleanup errors.

        Args:
            handler: Previously registered handler (or wrapper) to remove.
        """
        try:
            result = self._client.remove_event_handler(handler)
        except Exception:  # noqa: BLE001 - cleanup must never raise
            return
        if inspect.isawaitable(result):
            await result

    def _register_edited_warning_handler(self) -> None:
        """Attach the log-only watcher for edited messages (never parsed).

        Idempotent: reconnecting does not stack a second warning handler.
        """
        if self._edited_warn_handler is not None:
            return

        async def _warn_edited(event: Any) -> None:
            get_logger("telegram").bind(channel="telegram").warning(
                "edited message received (ignored by parser): chat_id={} msg_id={}",
                getattr(event, "chat_id", None),
                getattr(getattr(event, "message", None), "id", None),
            )

        self._client.add_event_handler(_warn_edited, events.MessageEdited)
        self._edited_warn_handler = _warn_edited

    def _make_deduplicating_handler(
        self,
        handler: NewMessageHandler,
        *,
        deduplicate: bool,
    ) -> Any:
        """Build a Telethon-compatible wrapper that skips already-seen messages.

        Telethon awaits every registered callback unconditionally, so the wrapper
        itself is a coroutine function: duplicates resolve to ``None`` *without*
        invoking the user handler, first occurrences delegate to it.

        Args:
            handler: User callback to invoke for the first occurrence of an event.
            deduplicate: When False, every event is forwarded without bookkeeping.

        Returns:
            An async callable that never passes duplicates to ``handler``.
        """

        async def _wrapped(event: Any) -> None:
            msg_id = getattr(getattr(event, "message", None), "id", None)
            event_chat_id = getattr(event, "chat_id", None)
            if msg_id is None or event_chat_id is None:
                await self._call_handler(handler, event)
                return
            key = (int(event_chat_id), int(msg_id))
            if deduplicate:
                if key in self._seen_ids:
                    return  # duplicate: handler is NOT called
                self._seen_ids.add(key)
                if len(self._seen_ids) > _SEEN_IDS_LIMIT:
                    self._seen_ids.clear()
                    self._seen_ids.add(key)
            await self._call_handler(handler, event)

        return _wrapped

    @staticmethod
    async def _call_handler(handler: NewMessageHandler, event: Any) -> None:
        """Invoke a user handler and await its result when it is awaitable.

        Args:
            handler: User callback registered via :meth:`add_message_handler`.
            event: Telethon event forwarded by the de-duplicating wrapper.
        """
        result = handler(event)
        if inspect.isawaitable(result):
            await result

    async def _disconnect_transport(self) -> None:
        """Await Telethon's disconnect result when it returns an awaitable."""
        result = self._client.disconnect()
        if inspect.isawaitable(result):
            await result

    async def _resolve_me_id(self) -> int | None:
        """Resolve the userbot's own Telegram user id, returning None on RPC failure.

        Rate-limit and access errors are re-raised as domain errors so the
        orchestrator can react; any other RPC failure degrades to ``None``.

        Raises:
            TelegramFloodWaitError: If Telegram rate-limits the ``get_me`` call.
            TelegramAccessError: If Telegram refuses access to the own account.
        """
        try:
            me = await self._client.get_me()
        except FloodWaitError as exc:
            wait = int(getattr(exc, "seconds", 0) or 0)
            get_logger("telegram").bind(channel="telegram").warning("flood wait: {} seconds", wait)
            raise TelegramFloodWaitError(wait) from exc
        except _ACCESS_ERRORS as exc:
            raise TelegramAccessError(f"access denied: {type(exc).__name__}: {exc}") from exc
        except RPCError:
            return None
        if me is not None and isinstance(me.id, int):
            return me.id
        return None

    async def _resolve_bot_id(self) -> int | None:
        """Resolve the game bot's numeric user id from config or @username.

        Rate-limit and access errors are re-raised as domain errors; other
        resolution failures degrade to ``None`` (caller raises ConfigError).

        Raises:
            TelegramFloodWaitError: If Telegram rate-limits the entity lookup.
            TelegramAccessError: If Telegram refuses access to the bot entity.
        """
        if self._config.bot_id is not None:
            return self._config.bot_id
        bot_username = self._config.bot_username
        if bot_username is None:  # pragma: no cover
            return None
        try:
            entity = await self._client.get_entity(bot_username)
        except FloodWaitError as exc:
            wait = int(getattr(exc, "seconds", 0) or 0)
            get_logger("telegram").bind(channel="telegram").warning("flood wait: {} seconds", wait)
            raise TelegramFloodWaitError(wait) from exc
        except _ACCESS_ERRORS as exc:
            raise TelegramAccessError(f"access denied: {type(exc).__name__}: {exc}") from exc
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
