"""Headless command loop connecting the scanner to application resources."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import threading
from collections.abc import Callable
from typing import Final, TextIO

from loguru import logger

from app.bootstrap import BootstrapResult, shutdown_result
from domain.events import AnyEvent, EndGameEvent
from domain.exceptions import ConfigError
from domain.game_state import GameState, Player
from domain.names import name_match_keys, names_match
from features.scanner.scanner import MessageScanner
from features.ui.protocol import ReasoningMessage, StatsMessage, StatsRow
from features.ui.window_host import WindowHost
from infra.db.connection import transaction

_LOGGER_CHANNEL: str = "orchestrator"
_ATTACH_PROMPT: str = "Введите номер чата для подключения (или пустую строку для отмены): "

#: Grace period before the automatic END_GAME retro. The final analysis
#: (a backfill catch-up flush or an in-flight LLM request) may still be
#: writing predictions, and retro has to label those rows.
_RETRO_DELAY_SEC: Final[float] = 5.0


class _StdinReader:
    """Dedicated thread that reads stdin lines and pushes them to an asyncio queue.

    ``asyncio.to_thread(sys.stdin.readline)`` on Windows has a race after a
    prior read: the worker thread can lose the console input handle state and
    block forever. A single dedicated reader thread that hands lines over with
    ``call_soon_threadsafe`` is reliable, because exactly one thread ever
    touches the stream.

    EOF is signalled by pushing ``None`` onto the queue.
    """

    def __init__(self, stream: TextIO) -> None:
        """Retain the stream to read from and prepare the handoff queue."""
        self._stream = stream
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the reader thread bound to ``loop``. Idempotent."""
        if self._thread is not None:
            return
        self._loop = loop
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="stdin-reader",
        )
        self._thread.start()

    def _run(self) -> None:
        """Read the stream until EOF or :meth:`stop` and forward each line."""
        loop = self._loop
        while not self._stopped:
            try:
                line = self._stream.readline()
            except Exception:  # noqa: BLE001 — a broken stream behaves like EOF
                line = ""
            if loop is None:
                return
            if not line:
                loop.call_soon_threadsafe(self._queue.put_nowait, None)
                return
            loop.call_soon_threadsafe(self._queue.put_nowait, line)

    async def get(self) -> str | None:
        """Await the next line, or ``None`` at EOF."""
        return await self._queue.get()

    def stop(self) -> None:
        """Ask the reader thread to exit after its current ``readline``."""
        self._stopped = True


class Orchestrator:
    """Main async loop that owns the scanner and application lifecycle.

    The primary console accepts commands while two output-only child windows
    display current stats and analyzer reasoning through lightweight IPC.
    """

    def __init__(
        self,
        *,
        bootstrap_result: BootstrapResult,
        scanner_factory: Callable[[int], MessageScanner] | None = None,
        stdin: TextIO | None = None,
        window_host: WindowHost | None = None,
        enable_ui: bool | None = None,
    ) -> None:
        """Retain initialized resources and injectable console/scanner seams.

        Args:
            bootstrap_result: Dependencies created by :func:`app.bootstrap.bootstrap`.
            scanner_factory: Optional factory used to create a scanner per chat.
            stdin: Optional input stream, primarily for tests.
            window_host: Optional injected owner for the two output windows.
            enable_ui: Override UI config; ``None`` uses configured windows.
        """
        self._result = bootstrap_result
        self._scanner_factory = scanner_factory or self._default_scanner
        self._stdin = stdin if stdin is not None else sys.stdin
        self._scanner: MessageScanner | None = None
        ui_cfg = bootstrap_result.config.ui
        ui_enabled = (
            ui_cfg.stats_window or ui_cfg.reasoning_window if enable_ui is None else enable_ui
        )
        self._ui: WindowHost | None = window_host
        if ui_enabled and self._ui is None:
            self._ui = WindowHost(
                stats_enabled=ui_cfg.stats_window,
                reasoning_enabled=ui_cfg.reasoning_window,
            )
        self._chat_id = bootstrap_result.config.telegram.chat_id
        self._quit_event = asyncio.Event()
        self._command_task: asyncio.Task[None] | None = None
        self._shutdown_complete = False
        self._player_persistence_lock = asyncio.Lock()
        # Serializes every sqlite3 access handed to ``asyncio.to_thread``: a
        # single ``sqlite3.Connection`` is not usable from two threads at once
        # (it raises ``InterfaceError`` and the write is lost).
        self._db_lock = asyncio.Lock()
        self._stdin_reader = _StdinReader(self._stdin)
        self._backfill_task: asyncio.Task[None] | None = None
        # END_GAME retro is deferred, not run inline with event persistence.
        self._pending_retro: bool = False
        self._pending_retro_task: asyncio.Task[None] | None = None

    @property
    def _registered_player_ids(self) -> set[int]:
        """Per-game cache of persisted players."""
        return self._result.game_state.persisted_ids

    def _default_scanner(self, chat_id: int) -> MessageScanner:
        """Build the production scanner with persistence and UI callbacks wired in."""
        bot_id = self._result.config.telegram.bot_id
        if bot_id is None:
            bot_id = self._result.telegram.bot_id
        return MessageScanner(
            telegram=self._result.telegram,
            game_state=self._result.game_state,
            analyzer=self._result.analyzer,
            registry=self._result.registry,
            trigger_policy=self._result.trigger_policy,
            chat_id=chat_id,
            bot_id=bot_id or 0,
            on_player_registered=self._persist_player,
            on_event=self._persist_event,
            on_reasoning=self._send_reasoning,
            on_flush_complete=self._send_stats,
            on_new_game=self._new_game_from_start,
        )

    async def _new_game_from_start(self) -> GameState:
        """Finish pending work for the old game and create a fresh aggregate."""
        pending = self._pending_retro_task
        if pending is not None and not pending.done():
            await pending
        scanner = self._scanner
        while scanner is not None and scanner.is_flushing:
            await asyncio.sleep(0.05)
        if self._chat_id is None:
            raise RuntimeError("Невозможно создать партию без подключённого чата")
        async with self._db_lock:
            await asyncio.to_thread(
                self._result.repos.games.close_game, game_id=self._result.game_state.game_id
            )
            game_id = await asyncio.to_thread(
                self._result.repos.games.create_game, chat_id=self._chat_id, player_count=0
            )
        self._result.game_state = self._result.game_state.reset_for_new_game(game_id)
        return self._result.game_state

    async def run(self) -> None:
        """Start the configured/selected chat and wait for quit or stdin EOF.

        A scanner start failure escapes to the entry point, which always runs
        :meth:`shutdown`. EOF is treated as a graceful end, just like ``/quit``.
        """
        if self._quit_event.is_set():
            return

        self._stdin_reader.start(asyncio.get_running_loop())

        if self._ui is not None:
            await self._ui.start()

        if self._chat_id is None:
            await self._handle_attach(None)
        else:
            await self._start_scanner(self._chat_id)

        # Backfill runs in the background so it never blocks the command loop.
        if self._scanner is not None and self._result.game_state.game_id > 0:
            self._backfill_task = asyncio.create_task(self._backfill_safe())
        else:
            self._backfill_task = None

        if self._quit_event.is_set():
            return

        self._command_task = asyncio.create_task(self._read_commands())
        quit_task = asyncio.create_task(self._quit_event.wait())
        try:
            await asyncio.wait(
                {self._command_task, quit_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._command_task.done() and not self._quit_event.is_set():
                self._quit_event.set()
        finally:
            if not quit_task.done():
                quit_task.cancel()
                try:
                    await quit_task
                except asyncio.CancelledError:
                    pass
            await self._cancel_backfill()
            command_task = self._command_task
            if command_task is not None:
                if not command_task.done():
                    command_task.cancel()
                try:
                    await command_task
                except asyncio.CancelledError:
                    pass
                except Exception as error:  # noqa: BLE001 — cleanup must still finish
                    logger.bind(channel=_LOGGER_CHANNEL).error(
                        "command reader failed while run was stopping: {}: {}",
                        type(error).__name__,
                        str(error),
                    )
                self._command_task = None

    async def shutdown(self) -> None:
        """Cancel command input, stop scanner/UI children, and close resources."""
        if self._shutdown_complete:
            return
        self._shutdown_complete = True

        self._stdin_reader.stop()

        # A deferred END_GAME retro still sleeping on its grace period must not
        # outlive the loop (or touch the DB after it was closed).
        pending_retro = self._pending_retro_task
        self._pending_retro_task = None
        self._pending_retro = False
        if pending_retro is not None and not pending_retro.done():
            pending_retro.cancel()
            try:
                await pending_retro
            except asyncio.CancelledError:
                pass
            except Exception as error:  # noqa: BLE001 — continue resource cleanup
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "scheduled retro failed during shutdown: {}: {}",
                    type(error).__name__,
                    str(error),
                )

        if self._backfill_task is not None and not self._backfill_task.done():
            self._backfill_task.cancel()
            try:
                await self._backfill_task
            except asyncio.CancelledError:
                pass
            except Exception as error:  # noqa: BLE001 — continue resource cleanup
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "backfill failed during shutdown: {}: {}",
                    type(error).__name__,
                    str(error),
                )
            self._backfill_task = None

        if self._command_task is not None and not self._command_task.done():
            self._command_task.cancel()
            try:
                await self._command_task
            except asyncio.CancelledError:
                pass
            except Exception as error:  # noqa: BLE001 — continue resource cleanup
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "command reader failed during shutdown: {}",
                    str(error),
                )

        scanner = self._scanner
        self._scanner = None
        try:
            if scanner is not None:
                await scanner.shutdown()
        except Exception as error:  # noqa: BLE001 — still close bootstrap resources
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "scanner shutdown failed: {}: {}",
                type(error).__name__,
                str(error),
            )
        finally:
            try:
                if self._ui is not None:
                    await self._ui.shutdown()
            except Exception as error:  # noqa: BLE001 — continue resource cleanup
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "UI shutdown failed: {}: {}",
                    type(error).__name__,
                    str(error),
                )
            await shutdown_result(self._result)

    async def _persist_player(self, player_id: int, name: str) -> None:
        """Insert a player once, before persisting predictions that reference it.

        Two tables are involved and both are needed: ``players`` is the global
        identity table the ``predictions`` foreign key points at, and
        ``game_players`` records participation in the current game — without
        that row :meth:`_run_retro` cannot store the final role and every
        prediction of the player stays "unknown".

        Order is load-bearing: ``players`` must exist before the
        ``game_players`` foreign key can be satisfied. Both writes are
        idempotent, so a repeated attach or a backfill only refreshes data.
        """
        async with self._player_persistence_lock:
            if player_id in self._registered_player_ids:
                return
            try:
                async with self._db_lock:
                    await asyncio.to_thread(
                        self._result.repos.players.upsert_player,
                        player_id=player_id,
                        username=name,
                        display_name=name,
                    )
                    game_id = self._result.game_state.game_id
                    if game_id > 0:
                        await asyncio.to_thread(
                            self._result.repos.game_players.add_player,
                            game_id=game_id,
                            player_id=player_id,
                            final_role=None,
                            survived=None,
                            is_winner=None,
                        )
                self._registered_player_ids.add(player_id)
            except Exception as error:  # noqa: BLE001 — best-effort persistence
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "persist_player failed: {}: {}", type(error).__name__, str(error)
                )

    async def _persist_event(self, event: AnyEvent) -> None:
        """Persist one applied scanner event without blocking the event loop.

        The bot's final report triggers the automatic retrospective, but it is
        *scheduled* in the background instead of running inline: the final
        flush (backfill catch-up or an in-flight analysis) may still be
        persisting predictions, and retro must label those rows. Re-running
        retro is idempotent (``upsert final_role`` + ``retro_analyze``).
        """
        try:
            async with self._db_lock:
                if (
                    isinstance(event, EndGameEvent)
                    and event.game_id != self._result.game_state.game_id
                ):
                    logger.bind(channel=_LOGGER_CHANNEL).warning(
                        "late END_GAME ignored: event game_id={}, current game_id={}",
                        event.game_id,
                        self._result.game_state.game_id,
                    )
                    return
                await asyncio.to_thread(self._result.repos.events.insert_event, event)
                if isinstance(event, EndGameEvent):
                    roles = [
                        self._result.registry.resolve_alias(role_name.strip())
                        for _, role_name in event.payload.winners
                    ]
                    if not roles or any(role is None for role in roles):
                        winner_faction = "unknown"
                    else:
                        factions = {role.faction.value for role in roles if role is not None}
                        winner_faction = next(iter(factions)) if len(factions) == 1 else "mixed"
                    await asyncio.to_thread(
                        self._result.repos.games.finish_game,
                        game_id=event.game_id,
                        winner_faction=winner_faction,
                    )
        except Exception as error:  # noqa: BLE001 — defensive callback boundary
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "persist event failed: {}: {}",
                type(error).__name__,
                str(error),
            )
            return
        # Schedule retro after END_GAME with a 5.0 s delay (flush may be in
        # progress); the final analysis must persist predictions before retro.
        if isinstance(event, EndGameEvent):
            self._pending_retro = True
            pending = self._pending_retro_task
            if pending is None or pending.done():
                self._pending_retro_task = asyncio.create_task(
                    self._schedule_retro(delay=_RETRO_DELAY_SEC)
                )

    async def _schedule_retro(self, delay: float = 5.0) -> None:
        """Run the retrospective after ``delay`` seconds, off the event path.

        The default is the production grace period (:data:`_RETRO_DELAY_SEC`);
        tests may pass a shorter delay explicitly.

        Args:
            delay: Grace period for the final analysis flush, in seconds.
        """
        try:
            await asyncio.sleep(delay)
            scanner = self._scanner
            if scanner is not None:
                waited = 0.0
                while scanner.is_flushing and waited < 60.0:
                    await asyncio.sleep(0.5)
                    waited += 0.5
                if waited > 0:
                    logger.bind(channel=_LOGGER_CHANNEL).info(
                        "retro: waited {:.1f}s for flush to complete", waited
                    )
            await self._run_retro()
        except Exception as error:  # noqa: BLE001 — retro must not break scanning
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "scheduled retro failed: {}: {}",
                type(error).__name__,
                str(error),
            )
        finally:
            self._pending_retro = False

    async def _merge_placeholder_players(self, game_state: GameState) -> None:
        """Merge unambiguous normalized name groups, preserving ground truth."""
        players = {player.player_id: player for player in game_state.players()}
        names = {pid: player.display_name for pid, player in players.items()}

        # A placeholder can already have disappeared from GameState when its
        # real player spoke, while the older participation row remains in SQLite.
        async with self._db_lock:
            rows = await asyncio.to_thread(
                self._result.repos.game_players.list_game_players, game_state.game_id
            )
            for row in rows:
                pid = int(row["player_id"])
                if pid in names:
                    continue
                db_player = await asyncio.to_thread(self._result.repos.players.get_player, pid)
                if db_player is not None and db_player["username"]:
                    names[pid] = str(db_player["username"])

        # The comparison is not necessarily transitive: connect all overlapping
        # key sets and reject the entire component if it contains multiple reals.
        remaining = set(names)
        while remaining:
            first = min(remaining)
            group = {first}
            remaining.remove(first)
            keys = set(name_match_keys(names[first]))
            while True:
                adjacent = {pid for pid in remaining if keys & name_match_keys(names[pid])}
                if not adjacent:
                    break
                for pid in adjacent:
                    keys.update(name_match_keys(names[pid]))
                group.update(adjacent)
                remaining.difference_update(adjacent)
            reals = [pid for pid in group if pid > 0]
            placeholders = [pid for pid in group if pid < 0]
            if not reals or not placeholders:
                continue
            if len(reals) != 1:
                logger.bind(channel=_LOGGER_CHANNEL).warning(
                    "ambiguous normalized names: placeholders={} real_ids={}",
                    sorted(placeholders),
                    sorted(reals),
                )
                continue
            keeper = reals[0]
            keeper_player = players.get(keeper)
            if keeper_player is not None:
                await self._persist_player(keeper, keeper_player.display_name)
            for ph in sorted(placeholders):
                placeholder = players.get(ph)
                if placeholder is not None:
                    await self._persist_player(ph, placeholder.display_name)

                def merge(ph: int, keeper: int, placeholder: Player | None) -> None:
                    with transaction(self._result.conn):
                        repo = self._result.repos.game_players
                        old = repo.get_player_in_game(game_id=game_state.game_id, player_id=ph)
                        real = repo.get_player_in_game(game_id=game_state.game_id, player_id=keeper)
                        if old is None or real is None:
                            raise ConfigError("missing participation row for merge")
                        final_role = (
                            placeholder.final_role.name_ru
                            if placeholder is not None and placeholder.final_role is not None
                            else old["final_role"]
                        )
                        if final_role is not None:
                            if real["final_role"] is not None and real["final_role"] != final_role:
                                raise ConfigError("conflicting final roles for merge")
                            repo.set_final_role(
                                game_id=game_state.game_id,
                                player_id=keeper,
                                final_role=str(final_role),
                                survived=(
                                    None
                                    if real["survived"] is not None
                                    else bool(old["survived"])
                                    if old["survived"] is not None
                                    else placeholder.alive
                                    if placeholder is not None
                                    and placeholder.final_role is not None
                                    else None
                                ),
                                is_winner=(
                                    None
                                    if real["is_winner"] is not None
                                    else bool(old["is_winner"])
                                    if old["is_winner"] is not None
                                    else placeholder.is_winner
                                    if placeholder is not None
                                    else None
                                ),
                            )
                        self._result.repos.predictions.merge_player(
                            from_player_id=ph, to_player_id=keeper, game_id=game_state.game_id
                        )
                        repo.delete_player(game_id=game_state.game_id, player_id=ph)
                        self._result.repos.players.delete_player(player_id=ph)

                try:
                    async with self._db_lock:
                        await asyncio.to_thread(merge, ph, keeper, placeholder)
                except Exception as error:  # noqa: BLE001 — best-effort merge
                    logger.bind(channel=_LOGGER_CHANNEL).warning(
                        "merge placeholder {} → {} ({}) failed: {}: {}",
                        ph,
                        keeper,
                        names[ph],
                        type(error).__name__,
                        str(error),
                    )
                else:
                    self._registered_player_ids.add(ph)

    async def _persist_final_roles(self, game_state: GameState, game_id: int) -> None:
        """Write every known final role into ``game_players``.

        Must be called while holding :attr:`_db_lock`. A player missing from
        ``game_players`` is expected (registered before 10c-fix-12, or the
        backfill has not reached it yet) and only logged as a warning.

        Args:
            game_state: State holding the final roles parsed from the report.
            game_id: Game whose participation rows are updated.
        """
        for player in game_state.players():
            if player.final_role is None:
                continue
            try:
                await asyncio.to_thread(
                    self._result.repos.game_players.set_final_role,
                    game_id=game_id,
                    player_id=player.player_id,
                    final_role=player.final_role.name_ru,
                    survived=player.alive,
                    is_winner=player.is_winner,
                )
            except (ConfigError, sqlite3.IntegrityError) as error:
                # IntegrityError: the participation row still points at a
                # placeholder that the merge cleanup could not delete — skip
                # this player instead of aborting the whole retro.
                logger.bind(channel=_LOGGER_CHANNEL).warning(
                    "final_role skipped for {} (not in game_players): {}",
                    player.player_id,
                    str(error),
                )
            except Exception as error:  # noqa: BLE001 — keep processing players
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "persist final_role failed for {}: {}: {}",
                    player.player_id,
                    type(error).__name__,
                    str(error),
                )

    async def _run_retro(self) -> None:
        """Persist the final standings and run the end-of-game retrospective.

        Order is load-bearing: ``retro_analyze`` reads
        ``game_players.final_role``, so the ground truth from the bot's final
        report must be written first. Both steps are idempotent — running retro
        twice only rewrites the same values.

        Everything is best-effort: a failing player update is logged and the
        remaining players are still persisted, and a failing ``retro_analyze``
        ends the method without touching the stats window.
        """
        game_state = self._result.game_state
        game_id = game_state.game_id
        if game_id == 0:
            return

        # Merge placeholder + real с одним display_name в game_players /
        # players: real приоритетнее, placeholder удаляется. Иначе один и тот
        # же человек висит в таблице дважды, а final_role достаётся случайной
        # (первой по порядку регистрации) записи.
        await self._merge_placeholder_players(game_state)

        # Persist ALL players (including END_GAME placeholders) to
        # game_players before retro. Without this, placeholders
        # registered by _apply_end_game never reach game_players →
        # retro drops them as unknown.
        for player in game_state.players():
            if player.player_id in self._registered_player_ids:
                continue
            try:
                await self._persist_player(player.player_id, player.display_name)
            except Exception as e:  # noqa: BLE001 — defensive
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "persist placeholder failed for {}: {}: {}",
                    player.player_id,
                    type(e).__name__,
                    str(e),
                )

        # One lock for the whole loop keeps the ground truth atomic: no other
        # sqlite call can interleave between two players of the same game.
        async with self._db_lock:
            await self._persist_final_roles(game_state, game_id)

        from features.memory.retro import retro_analyze

        try:
            async with self._db_lock:
                report = await asyncio.to_thread(
                    retro_analyze,
                    game_id=game_id,
                    registry=self._result.registry,
                    predictions_repo=self._result.repos.predictions,
                    game_players_repo=self._result.repos.game_players,
                )
        except Exception as error:  # noqa: BLE001 — retro is best-effort
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "retro_analyze failed: {}: {}",
                type(error).__name__,
                str(error),
            )
            return
        logger.bind(channel="retro").info(
            "game {} retro: total={} correct={} wrong={} unknown={}",
            game_id,
            report.total_predictions,
            report.correct,
            report.wrong,
            report.unknown,
        )

        try:
            await self._send_stats()
        except Exception as error:  # noqa: BLE001 — UI is best-effort
            logger.bind(channel=_LOGGER_CHANNEL).debug(
                "stats refresh after retro failed: {}: {}",
                type(error).__name__,
                str(error),
            )

    async def _send_stats(self) -> None:
        """Send all real players with their current predictions, if confident enough."""
        if self._ui is None:
            return
        game_state = self._result.game_state
        if game_state.game_id == 0:
            return
        try:
            async with self._db_lock:
                predictions = await asyncio.to_thread(
                    self._result.repos.predictions.list_predictions,
                    game_state.game_id,
                )
            min_confidence = self._result.config.analysis.min_confidence_to_display
            show_dead = self._result.config.analysis.show_dead
            all_players = game_state.players()
            eligible_ids = {player.player_id for player in all_players if show_dead or player.alive}
            predictions_by_player = {
                int(prediction["player_id"]): prediction
                for prediction in predictions
                if int(prediction["player_id"]) in eligible_ids
                and float(prediction["confidence"]) >= min_confidence
            }
            # ALIVE_LIST whitelist: utility bots that post in the game chat
            # (EzzProd & co.) are not players and must not get a table row.
            # An empty set means the bot has not sent ALIVE_LIST yet — show all
            # real players rather than nothing.
            whitelist = game_state.last_alive_list_names
            rows: list[StatsRow] = []
            for player in all_players:
                whitelisted = not whitelist or (
                    any(names_match(player.display_name, listed) for listed in whitelist)
                    or (
                        player.username is not None
                        and any(names_match(player.username, listed) for listed in whitelist)
                    )
                )
                # Placeholders are shown only after an ALIVE_LIST confirms the
                # nickname: before that they are just unmatched bot noise.
                if player.player_id <= 0 and not (whitelist and whitelisted):
                    continue
                if not whitelisted:
                    continue
                if not show_dead and not player.alive:
                    continue
                prediction = predictions_by_player.get(player.player_id)
                if prediction is not None:
                    role_str = str(prediction["predicted_role"])
                    if not player.alive:
                        role_str = f"[мёртв] {role_str}"
                    rows.append(
                        StatsRow(
                            player_name=player.display_name,
                            role=role_str,
                            confidence=float(prediction["confidence"]),
                        )
                    )
                else:
                    role_dash = "—" if player.alive else "[мёртв] —"
                    rows.append(
                        StatsRow(player_name=player.display_name, role=role_dash, confidence=0.0)
                    )
            rows.sort(key=lambda row: (-row.confidence, row.player_name))
            logger.bind(channel=_LOGGER_CHANNEL).debug(
                "send_stats: {} rows",
                len(rows),
            )
            # Send even an empty frame so the UI clears stale rows.
            message = StatsMessage(
                game_id=game_state.game_id,
                phase=game_state.current_phase().value,
                day=game_state.current_day(),
                rows=rows,
            )
            await self._ui.send_stats(message)
        except Exception:  # noqa: BLE001 — UI is best-effort
            logger.bind(channel=_LOGGER_CHANNEL).opt(exception=True).error("send_stats failed")

    async def _send_reasoning(self, text: str) -> None:
        """Send one reasoning result, clipping the IPC frame defensively."""
        logger.bind(channel=_LOGGER_CHANNEL).debug(
            "send_reasoning: text_len={}",
            len(text or ""),
        )
        if self._ui is None:
            return
        if not text:
            text = "(streaming reasoning unavailable — non-thinking model)"
        truncated = len(text) > 4096
        message = ReasoningMessage(
            text=text[:4096],
            truncated=truncated,
            total_chars=len(text),
        )
        try:
            await self._ui.send_reasoning(message)
        except Exception as error:  # noqa: BLE001 — UI is best-effort
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "send_reasoning failed: {}",
                str(error),
            )

    async def _read_commands(self) -> None:
        """Read commands from the stdin reader thread until ``/quit`` or EOF."""
        while not self._quit_event.is_set():
            line = await self._stdin_reader.get()
            if line is None:
                return
            await self._handle_command(line.strip())

    async def _handle_command(self, line: str) -> None:
        """Dispatch ``/attach``, ``/status``, and ``/quit`` commands."""
        if not line:
            return
        parts = line.split(maxsplit=1)
        command = parts[0]
        argument = parts[1] if len(parts) == 2 else None

        if command == "/status" and argument is None:
            await self._handle_status()
        elif command == "/quit" and argument is None:
            await self._handle_quit()
        elif command == "/reset" and argument is None:
            await self._handle_reset()
        elif command == "/retro" and argument is None:
            await self._handle_retro()
        elif command == "/attach":
            await self._handle_attach(argument)
        else:
            print(
                f"Unknown command: {line}. Available: /status, /quit, /attach [id], /reset, /retro",
                flush=True,
            )

    async def _handle_reset(self) -> None:
        """Handle /reset by starting a fresh game and scanner in the attached chat."""
        chat_id = self._chat_id
        if chat_id is None:
            print("No chat attached. Use /attach first.", flush=True)
            return
        await self._cancel_backfill()
        if self._scanner is not None:
            await self._scanner.shutdown()
            self._scanner = None
        try:
            async with self._db_lock:
                gid = await asyncio.to_thread(
                    self._result.repos.games.create_game, chat_id=chat_id, player_count=0
                )
            self._result.game_state = GameState(game_id=gid, registry=self._result.registry)
            await self._start_scanner(chat_id)
        except Exception as error:  # noqa: BLE001 — command boundary
            print(f"Reset failed: {error}", flush=True)
            return
        print(f"Reset: new game {gid}.", flush=True)

    async def _handle_retro(self) -> None:
        """Handle /retro by manually running retrospective analysis."""
        try:
            await self._run_retro()
            print("Retro: done (see logs).", flush=True)
        except Exception as error:  # noqa: BLE001 — command boundary
            print(f"Retro failed: {error}", flush=True)

    async def _handle_attach(self, arg: str | None) -> None:
        """Select a chat interactively or attach directly to a numeric id.

        History backfill is *not* run here: it is started in the background by
        :meth:`run` (see :meth:`_backfill_safe`) so an ``/attach`` command never
        blocks the command loop.
        """
        self._stdin_reader.start(asyncio.get_running_loop())
        dialogs: list[dict[str, object]] = []
        if arg is None:
            try:
                dialogs = await self._result.telegram.iter_dialogs(
                    groups_only=True,
                    dialogs_limit=50,
                )
            except Exception as error:  # noqa: BLE001 — surface failure in console/log
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "cannot list Telegram dialogs: {}: {}",
                    type(error).__name__,
                    str(error),
                )
                print(f"Не удалось получить список чатов: {error}", flush=True)
                return

            if not dialogs:
                print("Не найдено доступных групповых чатов.", flush=True)
                return
            for index, dialog in enumerate(dialogs, start=1):
                title = str(dialog.get("title", ""))
                kind = str(dialog.get("kind", ""))
                print(f"{index}. {title} ({kind})", flush=True)
            print(_ATTACH_PROMPT, end="", flush=True)
            # Read through the shared reader thread instead of
            # ``asyncio.to_thread(stdin.readline)``: on Windows the worker-thread
            # read races with the command reader and can block forever.
            selection = await self._stdin_reader.get()
            if not selection or not selection.strip():
                print("Выбор чата отменён.", flush=True)
                return
            try:
                dialog_index = int(selection.strip())
            except ValueError:
                print("Некорректный номер чата.", flush=True)
                return
            if not 1 <= dialog_index <= len(dialogs):
                print("Номер чата вне списка.", flush=True)
                return
            try:
                dialog_chat_id = dialogs[dialog_index - 1]["id"]
                if isinstance(dialog_chat_id, bool) or not isinstance(dialog_chat_id, int):
                    raise TypeError("dialog id must be an integer")
                chat_id = dialog_chat_id
            except (KeyError, TypeError) as error:
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "invalid dialog entry: {}",
                    str(error),
                )
                print("Не удалось определить id выбранного чата.", flush=True)
                return
        else:
            try:
                chat_id = int(arg)
            except ValueError:
                print("chat_id должен быть числом.", flush=True)
                return

        try:
            await self._result.telegram.get_chat(chat_id)
        except Exception as error:  # noqa: BLE001 — access errors are user-visible
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "cannot access Telegram chat {}: {}: {}",
                chat_id,
                type(error).__name__,
                str(error),
            )
            print(f"Не удалось подключиться к чату {chat_id}: {error}", flush=True)
            return

        if self._scanner is not None:
            await self._scanner.shutdown()
            self._scanner = None

        # Bootstrap uses game_id=0 as a placeholder while no chat is selected.
        # The first attach replaces that placeholder; later attaches intentionally
        # keep the current game/state in v1 rather than splitting it per chat.
        if self._result.game_state.game_id == 0:
            try:
                async with self._db_lock:
                    game_id = await asyncio.to_thread(
                        self._result.repos.games.create_game,
                        chat_id=chat_id,
                        player_count=0,
                    )
            except Exception as error:  # noqa: BLE001 — database failures are logged
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "cannot create game for attached chat: {}: {}",
                    type(error).__name__,
                    str(error),
                )
                print(f"Не удалось создать запись партии: {error}", flush=True)
                return
            self._result.game_state = GameState(
                game_id=game_id,
                registry=self._result.registry,
            )
            logger.bind(channel=_LOGGER_CHANNEL).info(
                "created game {} for chat_id={}", game_id, chat_id
            )

        await self._start_scanner(chat_id)
        print(f"Подключён чат {chat_id}.", flush=True)

    async def _start_scanner(self, chat_id: int) -> None:
        """Construct and start a scanner for the selected chat."""
        scanner = self._scanner_factory(chat_id)
        await scanner.start()
        self._scanner = scanner
        self._chat_id = chat_id

    async def _backfill_safe(self) -> None:
        """Run the history backfill in the background with full diagnostics.

        Called as a task from :meth:`run`; every outcome (success, empty
        history, timeout, failure) is printed to the console *and* logged, so a
        silent backfill can never go undiagnosed again. The 300 s timeout keeps a
        wedged Telegram call from pinning the task forever.
        """
        scanner = self._scanner
        if scanner is None:
            return
        try:
            applied = await asyncio.wait_for(scanner.backfill(), timeout=300.0)
            if applied > 0:
                print(f"Backfilled {applied} messages.", flush=True)
                logger.bind(channel=_LOGGER_CHANNEL).info("backfilled {} messages", applied)
            else:
                print(
                    "Backfill: no messages applied (GAME_START not found or no history).",
                    flush=True,
                )
                logger.bind(channel=_LOGGER_CHANNEL).warning(
                    "backfill: 0 applied (GAME_START not found?)"
                )
        except TimeoutError:
            print("Backfill: timed out after 300s.", flush=True)
            logger.bind(channel=_LOGGER_CHANNEL).warning("backfill timeout (300s)")
        except Exception as e:  # noqa: BLE001 — background task boundary
            print(f"Backfill failed: {type(e).__name__}: {e}", flush=True)
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "backfill failed: {}: {}", type(e).__name__, str(e)
            )

    async def _cancel_backfill(self) -> None:
        """Cancel and await the background backfill task, if any."""
        task = self._backfill_task
        self._backfill_task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as error:  # noqa: BLE001 — cleanup must finish
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "backfill failed while stopping: {}: {}",
                type(error).__name__,
                str(error),
            )

    async def _handle_status(self) -> None:
        """Print the current game snapshot and the five strongest predictions."""
        game_state = self._result.game_state
        alive = game_state.alive_players()
        alive_count = len(alive)
        bot_total = game_state.total_players
        # NEVER len(players()): placeholders and already-merged duplicates
        # inflate it (35 players for a 30-player game). The bot's headcount is
        # authoritative while it is consistent; otherwise it is stale (e.g. it
        # still reports 4 when 7 players are alive) and alive_count is used.
        if bot_total is not None and bot_total >= alive_count >= 1:
            display_total = bot_total
        else:
            display_total = alive_count
        print(
            f"Game #{game_state.game_id}, "
            f"phase={game_state.current_phase().value}, "
            f"current_day={game_state.current_day()}, "
            f"alive={len(alive)}/{display_total}",
            flush=True,
        )

        async with self._db_lock:
            predictions = await asyncio.to_thread(
                self._result.repos.predictions.list_predictions,
                game_state.game_id,
            )
        top_predictions = sorted(
            predictions,
            key=lambda row: float(row["confidence"]),
            reverse=True,
        )[:5]
        for prediction in top_predictions:
            print(
                f"player {prediction['player_id']}: {prediction['predicted_role']} "
                f"({float(prediction['confidence']):.0%})",
                flush=True,
            )
        if not top_predictions:
            print("Предсказаний пока нет.", flush=True)

        buffer_size = self._scanner.buffer_size if self._scanner is not None else 0
        print(f"messages in buffer: {buffer_size}", flush=True)

    async def _handle_quit(self) -> None:
        """Signal the main loop to stop."""
        self._quit_event.set()
