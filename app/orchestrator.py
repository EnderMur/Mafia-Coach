"""Headless command loop connecting the scanner to application resources."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import TextIO

from loguru import logger

from app.bootstrap import BootstrapResult, shutdown_result
from domain.events import AnyEvent
from domain.game_state import GameState
from features.scanner.scanner import MessageScanner

_LOGGER_CHANNEL: str = "orchestrator"
_ATTACH_PROMPT: str = "Введите номер чата для подключения (или пустую строку для отмены): "


class Orchestrator:
    """Main async loop that owns the scanner and application lifecycle.

    Step 10a deliberately stays headless: the primary console accepts commands,
    but the two display windows are outside this module and belong to step 10b.
    """

    def __init__(
        self,
        *,
        bootstrap_result: BootstrapResult,
        scanner_factory: Callable[[int], MessageScanner] | None = None,
        stdin: TextIO | None = None,
    ) -> None:
        """Retain initialized resources and injectable console/scanner seams.

        Args:
            bootstrap_result: Dependencies created by :func:`app.bootstrap.bootstrap`.
            scanner_factory: Optional factory used to create a scanner per chat.
            stdin: Optional input stream, primarily for tests.
        """
        self._result = bootstrap_result
        self._scanner_factory = scanner_factory or self._default_scanner
        self._stdin = stdin if stdin is not None else sys.stdin
        self._scanner: MessageScanner | None = None
        self._chat_id = bootstrap_result.config.telegram.chat_id
        self._quit_event = asyncio.Event()
        self._command_task: asyncio.Task[None] | None = None
        self._shutdown_complete = False

    def _default_scanner(self, chat_id: int) -> MessageScanner:
        """Build the production scanner with event persistence wired in."""
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
            on_event=self._persist_event,
        )

    async def run(self) -> None:
        """Start the configured/selected chat and wait for quit or stdin EOF.

        A scanner start failure escapes to the entry point, which always runs
        :meth:`shutdown`. EOF is treated as a graceful end, just like ``/quit``.
        """
        if self._quit_event.is_set():
            return

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

    async def shutdown(self) -> None:
        """Cancel command input, stop the scanner, and close owned resources."""
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
            await shutdown_result(self._result)

    async def _persist_event(self, event: AnyEvent) -> None:
        """Persist one applied scanner event without blocking the event loop."""
        try:
            await asyncio.to_thread(self._result.repos.events.insert_event, event)
        except Exception as error:  # noqa: BLE001 — defensive callback boundary
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "persist event failed: {}: {}",
                type(error).__name__,
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
        print(
            f"Game #{game_state.game_id}, phase={game_state.current_phase().value}, "
            f"current_day={game_state.current_day()}, alive={len(alive)}/{len(players)}"
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
