"""Headless command loop connecting the scanner to application resources."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import TextIO

from loguru import logger

from app.bootstrap import BootstrapResult, shutdown_result
from domain.events import AnyEvent, EndGameEvent
from domain.game_state import GameState
from features.scanner.scanner import MessageScanner
from features.ui.protocol import ReasoningMessage, StatsMessage, StatsRow
from features.ui.window_host import WindowHost

_LOGGER_CHANNEL: str = "orchestrator"
_ATTACH_PROMPT: str = "Введите номер чата для подключения (или пустую строку для отмены): "


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
        self._registered_player_ids: set[int] = set()
        self._player_persistence_lock = asyncio.Lock()

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
        )

    async def run(self) -> None:
        """Start the configured/selected chat and wait for quit or stdin EOF.

        A scanner start failure escapes to the entry point, which always runs
        :meth:`shutdown`. EOF is treated as a graceful end, just like ``/quit``.
        """
        if self._quit_event.is_set():
            return

        if self._ui is not None:
            await self._ui.start()

        if self._chat_id is None:
            await self._handle_attach(None)
        else:
            await self._start_scanner(self._chat_id)

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
        """Insert a player once, before persisting predictions that reference it."""
        async with self._player_persistence_lock:
            if player_id in self._registered_player_ids:
                return
            try:
                await asyncio.to_thread(
                    self._result.repos.players.upsert_player,
                    player_id=player_id,
                    username=name,
                    display_name=name,
                )
                self._registered_player_ids.add(player_id)
            except Exception as error:  # noqa: BLE001 — best-effort persistence
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "persist_player failed: {}: {}", type(error).__name__, str(error)
                )

    async def _persist_event(self, event: AnyEvent) -> None:
        """Persist one applied scanner event without blocking the event loop.

        The bot's final report is the trigger for the automatic retrospective:
        once it is stored, :meth:`_run_retro` writes the ground truth into
        ``game_players`` and labels the game's predictions.
        """
        try:
            await asyncio.to_thread(self._result.repos.events.insert_event, event)
        except Exception as error:  # noqa: BLE001 — defensive callback boundary
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "persist event failed: {}: {}",
                type(error).__name__,
                str(error),
            )
            return
        if isinstance(event, EndGameEvent):
            try:
                await self._run_retro()
            except Exception as error:  # noqa: BLE001 — retro must not break scanning
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "auto-retro failed: {}: {}",
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
            except Exception as error:  # noqa: BLE001 — keep processing players
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "persist final_role failed for {}: {}: {}",
                    player.player_id,
                    type(error).__name__,
                    str(error),
                )

        from features.memory.retro import retro_analyze

        try:
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
        """Send persisted predictions after a flush, or empty rows to clear stale stats."""
        if self._ui is None:
            return
        game_state = self._result.game_state
        if game_state.game_id == 0:
            return
        try:
            predictions = await asyncio.to_thread(
                self._result.repos.predictions.list_predictions,
                game_state.game_id,
            )
            min_confidence = self._result.config.analysis.min_confidence_to_display
            sorted_predictions = sorted(
                predictions,
                key=lambda row: float(row["confidence"]),
                reverse=True,
            )[:20]
            players = game_state.players()
            rows: list[StatsRow] = []
            for prediction in sorted_predictions:
                confidence = float(prediction["confidence"])
                if confidence < min_confidence:
                    continue
                player = next(
                    (item for item in players if item.player_id == prediction["player_id"]),
                    None,
                )
                if player is None:
                    continue
                rows.append(
                    StatsRow(
                        player_name=player.display_name,
                        role=str(prediction["predicted_role"]),
                        confidence=confidence,
                    )
                )
            logger.bind(channel=_LOGGER_CHANNEL).debug(
                "send_stats: {} rows",
                len(rows),
            )
            # Empty predictions still send a frame so the UI clears stale rows.
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
        """Read commands from stdin until ``/quit`` or EOF."""
        while not self._quit_event.is_set():
            try:
                line = await asyncio.to_thread(self._stdin.readline)
            except Exception as error:  # noqa: BLE001 — keep input failures visible
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "stdin read failed: {}",
                    str(error),
                )
                return
            if not line:
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
        elif command == "/attach":
            await self._handle_attach(argument)
        else:
            print(f"Unknown command: {line}. Available: /status, /quit, /attach [id]")

    async def _handle_attach(self, arg: str | None) -> None:
        """Select a chat interactively or attach directly to a numeric id."""
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
                print(f"Не удалось получить список чатов: {error}")
                return

            if not dialogs:
                print("Не найдено доступных групповых чатов.")
                return
            for index, dialog in enumerate(dialogs, start=1):
                title = str(dialog.get("title", ""))
                kind = str(dialog.get("kind", ""))
                print(f"{index}. {title} ({kind})")
            print(_ATTACH_PROMPT, end="", flush=True)
            try:
                selection = await asyncio.to_thread(self._stdin.readline)
            except Exception as error:  # noqa: BLE001 — console read boundary
                logger.bind(channel=_LOGGER_CHANNEL).error(
                    "stdin read failed during attach: {}",
                    str(error),
                )
                print(f"Не удалось прочитать выбор чата: {error}")
                return
            if not selection or not selection.strip():
                print("Выбор чата отменён.")
                return
            try:
                dialog_index = int(selection.strip())
            except ValueError:
                print("Некорректный номер чата.")
                return
            if not 1 <= dialog_index <= len(dialogs):
                print("Номер чата вне списка.")
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
                print("Не удалось определить id выбранного чата.")
                return
        else:
            try:
                chat_id = int(arg)
            except ValueError:
                print("chat_id должен быть числом.")
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
            print(f"Не удалось подключиться к чату {chat_id}: {error}")
            return

        if self._scanner is not None:
            await self._scanner.shutdown()
            self._scanner = None

        # Bootstrap uses game_id=0 as a placeholder while no chat is selected.
        # The first attach replaces that placeholder; later attaches intentionally
        # keep the current game/state in v1 rather than splitting it per chat.
        if self._result.game_state.game_id == 0:
            try:
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
                print(f"Не удалось создать запись партии: {error}")
                return
            self._result.game_state = GameState(
                game_id=game_id,
                registry=self._result.registry,
            )
            logger.bind(channel=_LOGGER_CHANNEL).info(
                "created game {} for chat_id={}", game_id, chat_id
            )

        await self._start_scanner(chat_id)
        print(f"Подключён чат {chat_id}.")

    async def _start_scanner(self, chat_id: int) -> None:
        """Construct and start a scanner for the selected chat."""
        scanner = self._scanner_factory(chat_id)
        await scanner.start()
        self._scanner = scanner
        self._chat_id = chat_id

    async def _handle_status(self) -> None:
        """Print the current game snapshot and the five strongest predictions."""
        game_state = self._result.game_state
        players = game_state.players()
        alive = game_state.alive_players()
        total = game_state.total_players
        total_or_known = total if total is not None else len(players)
        print(
            f"Game #{game_state.game_id}, phase={game_state.current_phase().value}, "
            f"current_day={game_state.current_day()}, "
            f"alive={len(alive)}/{total_or_known}"
        )

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
                f"({float(prediction['confidence']):.0%})"
            )
        if not top_predictions:
            print("Предсказаний пока нет.")

        buffer_size = self._scanner.buffer_size if self._scanner is not None else 0
        print(f"messages in buffer: {buffer_size}")

    async def _handle_quit(self) -> None:
        """Signal the main loop to stop."""
        self._quit_event.set()
