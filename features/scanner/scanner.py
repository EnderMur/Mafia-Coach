"""Async orchestration: Telegram → GameState → analyzer → database.

The scanner is the only place in ``features/scanner`` that is allowed to be
stateful and asynchronous. It glues the pure pieces of the pipeline together:

1. :mod:`features.parser.classifier` decides what a raw message is;
2. :mod:`features.parser.bot_messages` + :mod:`features.scanner.bot_resolver`
   turn bot announcements into domain events;
3. :class:`domain.game_state.GameState` validates and applies every event;
4. :class:`features.scanner.trigger_policy.TriggerPolicy` decides *when* the
   LLM should look at the game, and :class:`features.analyzer.analyzer.GameAnalyzer`
   performs the analysis and persists predictions.

Event persistence is intentionally **not** done here: raw events are written to
SQLite by the orchestrator (step 10), which owns repository lifetimes and the
``asyncio.to_thread`` offloading of the synchronous ``sqlite3`` calls
(AI_INSTRUCTIONS.md §4, ``docs/llm_contract.md``).

Concurrency model
-----------------
One instance per chat, created and used inside a single asyncio event loop.
The object is not thread-safe. ``_flush`` is guarded by ``_flushing`` so that
overlapping triggers never reach the analyzer twice. The debounce timer is a
plain :class:`asyncio.Task`; it is cancelled on ``shutdown`` and before every
forced flush.
"""

from __future__ import annotations

import asyncio
from typing import Any, Final

from loguru import logger

from domain.events import (
    AnyEvent,
    DeathEvent,
    MessageEvent,
    MessagePayload,
    RoleRevealEvent,
)
from domain.game_state import GameState
from domain.roles import RoleRegistry
from features.analyzer.analyzer import GameAnalyzer
from features.parser.bot_messages import parse_bot_message
from features.parser.classifier import MessageKind, classify_message
from features.scanner.bot_resolver import resolve_bot_message
from features.scanner.trigger_policy import TriggerPolicy
from infra.telegram.client import TelegramClientWrapper

_LOGGER_CHANNEL: Final[str] = "scanner"

#: Prefix for synthetic display names when Telethon hides the sender entity
#: (e.g. the sender left the chat and the entity is no longer resolvable).
_FALLBACK_NAME_PREFIX: Final[str] = "user_"


def _sender_display_name(sender: Any) -> str | None:
    """Extract a human-readable name from a Telethon sender entity.

    The bot writes nicknames as plain text, so the scanner needs the same
    string a human would see. Priority:

    1. ``first_name`` + ``last_name`` (``User``) → ``"Иван Петров"``;
    2. ``title`` (``Chat``/``Channel``) → ``"Название"``;
    3. ``username`` (``User``) → ``"ivan"`` (without the ``@``);
    4. ``None`` when nothing is available.

    Args:
        sender: Telethon sender entity, ``None`` when unavailable.

    Returns:
        Display name or ``None``.
    """
    if sender is None:
        return None
    first_name = getattr(sender, "first_name", None)
    if first_name:
        last_name = getattr(sender, "last_name", None) or ""
        return f"{first_name} {last_name}".strip()
    title = getattr(sender, "title", None)
    if title:
        return str(title)
    username = getattr(sender, "username", None)
    if username:
        return str(username)
    return None


def _event_player(event: AnyEvent) -> tuple[int, str] | None:
    """Return ``(player_id, display_name)`` referenced by an event, if any.

    Deaths and role reveals must reference a player already known to
    :class:`~domain.game_state.GameState`; messages create players on the fly.
    Votes are not player-introducing (a voter may abstain or be unknown), so
    they return ``None``.

    Args:
        event: domain event produced by :func:`resolve_bot_message`.

    Returns:
        ``(player_id, display_name)`` or ``None`` when the event introduces no
        player.
    """
    if isinstance(event, MessageEvent):
        return event.payload.author_id, event.payload.author_name
    if isinstance(event, DeathEvent):
        return event.payload.victim_id, event.payload.victim_name
    if isinstance(event, RoleRevealEvent):
        return event.payload.player_id, event.payload.player_name
    return None


class MessageScanner:
    """Async orchestration: Telegram → GameState → analyzer → DB.

    Subscribes to the new messages of one chat, parses bot announcements,
    classifies player speech, applies everything to :class:`GameState`, and
    runs the analysis according to :class:`TriggerPolicy`.

    Not thread-safe and not shareable between chats: one instance per game.
    """

    def __init__(
        self,
        *,
        telegram: TelegramClientWrapper,
        game_state: GameState,
        analyzer: GameAnalyzer,
        registry: RoleRegistry,
        trigger_policy: TriggerPolicy,
        chat_id: int,
        bot_id: int,
    ) -> None:
        """Retain dependencies and initialize the empty scan state.

        Args:
            telegram: read-only Telethon wrapper used for subscription.
            game_state: aggregate the scanner feeds with events.
            analyzer: analyzer invoked on every flush.
            registry: role registry for bot-message parsing.
            trigger_policy: when-to-flush decisions.
            chat_id: chat observed by this scanner.
            bot_id: Telegram id of the game bot.
        """
        self._telegram = telegram
        self._game_state = game_state
        self._analyzer = analyzer
        self._registry = registry
        self._policy = trigger_policy
        self._chat_id = chat_id
        self._bot_id = bot_id

        self._name_to_id: dict[str, int] = {}
        self._registered_ids: set[int] = set()
        self._buffer: list[AnyEvent] = []
        self._debounce_task: asyncio.Task[None] | None = None
        self._flushing: bool = False
        self._started: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to the chat's new messages.

        Idempotent: a second call is a no-op and never re-registers the
        handler. Only ``events.NewMessage`` is subscribed by the wrapper
        (AI_INSTRUCTIONS.md §13.1 п.7), edited messages are ignored.
        """
        if self._started:
            return
        self._telegram.add_message_handler(
            self._on_event,
            chat_id=self._chat_id,
        )
        self._started = True
        logger.bind(channel=_LOGGER_CHANNEL).info(
            "scanner started: chat_id={} mode={}",
            self._chat_id,
            self._policy.mode.value,
        )

    async def shutdown(self) -> None:
        """Cancel the debounce timer and detach handlers. Idempotent.

        An in-flight flush is *not* awaited: the orchestrator (step 10) calls
        ``telegram.aclose()`` right after ``scanner.shutdown()`` and owns the
        remaining lifecycle. The timer task is cancelled and given a single
        event-loop turn to unwind so no pending task is left behind.
        """
        self._cancel_debounce()
        await asyncio.sleep(0)
        if self._started:
            await self._telegram.remove_all_handlers()
            self._started = False
        logger.bind(channel=_LOGGER_CHANNEL).info("scanner stopped")

    # ------------------------------------------------------------------
    # Telegram handler
    # ------------------------------------------------------------------

    async def _on_event(self, event: Any) -> None:
        """Telegram ``NewMessage`` handler. Never raises.

        A Telethon callback runs inside the client's dispatch loop, so an
        exception escaping it would surface as an unhandled task error and
        could stall the whole pipeline. Every failure is logged instead.

        Args:
            event: Telethon new-message event.
        """
        try:
            await self._process_event(event)
        except Exception as e:  # noqa: BLE001 — handler is a protective boundary
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "handler failed: {}: {}",
                type(e).__name__,
                str(e),
            )

    async def _process_event(self, event: Any) -> None:
        """Classify one raw message and route it to the matching handler.

        Args:
            event: Telethon new-message event (or a compatible fake in tests).
        """
        sender_id = getattr(event, "sender_id", None)
        if sender_id is None:
            # Anonymous admin, channel post without a signature, service event.
            return

        display_name = _sender_display_name(getattr(event, "sender", None))
        if (
            sender_id != self._bot_id
            and display_name is not None
            and display_name not in self._name_to_id
        ):
            # Only players are addressable by nickname: the bot's own title must
            # never shadow a player name. First observed spelling wins — the bot
            # writes the nickname it saw at registration, so later renames of a
            # player must not break death/vote resolution.
            self._name_to_id[display_name] = sender_id

        message = getattr(event, "message", None)
        text = getattr(message, "text", None) if message is not None else None

        kind = classify_message(
            text=text,
            from_id=sender_id,
            bot_id=self._bot_id,
        )
        if kind is MessageKind.IRRELEVANT:
            return

        ts = getattr(event, "date", None)
        if ts is None:
            return

        if kind is MessageKind.BOT_MESSAGE:
            await self._handle_bot_message(text, ts)
        else:
            # MessageKind has exactly three members and IRRELEVANT returned
            # above, so PLAYER_SPEECH is the only remaining possibility.
            applied = self._handle_player_speech(
                author_id=sender_id,
                author_name=display_name or f"{_FALLBACK_NAME_PREFIX}{sender_id}",
                text=text or "",
                ts=ts,
            )
            if applied is not None:
                await self._maybe_flush(triggered_by=[applied])

    # ------------------------------------------------------------------
    # Event production
    # ------------------------------------------------------------------

    async def _handle_bot_message(self, text: str | None, ts: Any) -> None:
        """Parse a game-bot message, apply its events, maybe flush.

        Args:
            text: raw bot message text.
            ts: timezone-aware timestamp of the source message.
        """
        if not text:
            return
        parse = parse_bot_message(text, registry=self._registry)
        if parse is None:
            return
        events = resolve_bot_message(
            parse=parse,
            name_to_id=self._name_to_id,
            registry=self._registry,
            game_id=self._game_state.game_id,
            ts=ts,
            current_phase=self._game_state.current_phase(),
        )
        if not events:
            return
        applied: list[AnyEvent] = []
        for event in events:
            self._ensure_event_player(event)
            if self._apply_safely(event):
                applied.append(event)
        if applied:
            self._buffer.extend(applied)
            await self._maybe_flush(triggered_by=applied)

    def _handle_player_speech(
        self,
        *,
        author_id: int,
        author_name: str,
        text: str,
        ts: Any,
    ) -> AnyEvent | None:
        """Turn a player message into a ``MessageEvent`` and buffer it.

        Args:
            author_id: Telegram id of the author.
            author_name: nickname shown by Telegram.
            text: message text.
            ts: timezone-aware timestamp of the source message.

        Returns:
            The applied event, or ``None`` when it was dropped. The caller
            awaits :meth:`_maybe_flush` because this method stays synchronous.
        """
        if not text.strip():
            return None
        self._register_player(author_id, author_name)
        event = MessageEvent(
            ts=ts,
            game_id=self._game_state.game_id,
            payload=MessagePayload(
                author_id=author_id,
                author_name=author_name,
                text=text,
            ),
        )
        if not self._apply_safely(event):
            return None
        self._buffer.append(event)
        return event

    # ------------------------------------------------------------------
    # GameState plumbing
    # ------------------------------------------------------------------

    def _register_player(self, player_id: int, display_name: str) -> None:
        """Introduce a player to ``GameState`` once.

        ``GameState`` starts empty and rejects deaths/role reveals for unknown
        players, so the scanner registers every player it learns about.

        Args:
            player_id: Telegram user id.
            display_name: nickname the bot writes in game.
        """
        if player_id in self._registered_ids:
            return
        if not display_name.strip():
            return
        self._game_state.register_player(player_id, display_name)
        self._registered_ids.add(player_id)
        logger.bind(channel=_LOGGER_CHANNEL).debug(
            "player registered: id={} name={!r}",
            player_id,
            display_name,
        )

    def _ensure_event_player(self, event: AnyEvent) -> None:
        """Register the player an event refers to, before applying it.

        Args:
            event: event resolved from a bot message.
        """
        referenced = _event_player(event)
        if referenced is None:
            return
        player_id, display_name = referenced
        self._register_player(player_id, display_name)

    def _apply_safely(self, event: AnyEvent) -> bool:
        """Apply an event, downgrading domain rejections to a log record.

        Args:
            event: event to apply to ``GameState``.

        Returns:
            ``True`` when the event was applied, ``False`` when rejected.
        """
        try:
            self._game_state.apply(event)
        except Exception as e:  # noqa: BLE001 — one bad event must not stop the game
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "GameState.apply({}) failed: {}: {}",
                event.kind,
                type(e).__name__,
                str(e),
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Flush triggering
    # ------------------------------------------------------------------

    async def _maybe_flush(self, *, triggered_by: list[AnyEvent]) -> None:
        """Flush, schedule a debounce, or do nothing — delegated to the policy.

        Args:
            triggered_by: events produced by the current Telegram message.
        """
        policy = self._policy
        if any(policy.should_force_now(event) for event in triggered_by):
            await self._flush()
            return
        if policy.should_force_on_size(len(self._buffer)):
            await self._flush()
            return
        if policy.is_batch():
            self._schedule_debounce()

    def _schedule_debounce(self) -> None:
        """(Re)start the debounce timer, replacing any pending one."""
        self._cancel_debounce()
        self._debounce_task = asyncio.create_task(self._debounce_and_flush())

    async def _debounce_and_flush(self) -> None:
        """Sleep for ``batch_debounce_sec`` and flush when the chat goes quiet."""
        try:
            await asyncio.sleep(self._policy.batch_debounce_sec)
        except asyncio.CancelledError:
            return
        # Drop the reference first: the flush below must not try to cancel the
        # very task that is running it, and shutdown() must not cancel an
        # in-flight analysis.
        self._debounce_task = None
        await self._flush()

    def _cancel_debounce(self) -> None:
        """Cancel the pending debounce timer, if any."""
        task = self._debounce_task
        self._debounce_task = None
        if task is None or task.done():
            return
        task.cancel()

    async def _flush(self) -> None:
        """Hand the current game state to the analyzer and drain the buffer.

        Re-entrancy is guarded by ``_flushing``: a trigger that arrives while
        the LLM is thinking is dropped instead of queueing a second identical
        request. Events applied during the analysis are kept in the buffer and
        a new debounce timer is scheduled for ``batch_debounce`` mode so they
        are never starved.
        """
        if self._flushing:
            return
        self._flushing = True
        try:
            self._cancel_debounce()
            self._buffer.clear()
            result = await self._analyzer.analyze(self._game_state, persist=True)
            response = result.response
            if not result.success or response is None:
                logger.bind(channel=_LOGGER_CHANNEL).warning(
                    "flush failed: {}",
                    result.error,
                )
            else:
                logger.bind(channel=_LOGGER_CHANNEL).info(
                    "flush ok: {} predictions, {}s",
                    len(response.analysis),
                    round(result.duration_sec, 2),
                )
        finally:
            self._flushing = False
            if self._started and self._policy.is_batch() and self._buffer:
                self._schedule_debounce()

    # ------------------------------------------------------------------
    # Read-only introspection (tests, orchestrator step 10)
    # ------------------------------------------------------------------

    @property
    def name_to_id(self) -> dict[str, int]:
        """Return a copy of the nickname → player id mapping."""
        return dict(self._name_to_id)

    @property
    def buffer_size(self) -> int:
        """Return the number of events waiting for the next analysis."""
        return len(self._buffer)

    @property
    def is_started(self) -> bool:
        """Return whether the scanner is subscribed to the chat."""
        return self._started
