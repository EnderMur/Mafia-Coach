"""Async owner for the stats and reasoning output-window subprocesses."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Final

from loguru import logger

from features.ui.protocol import IPCEnvelope, ReasoningMessage, StatsMessage, encode

_LOGGER_CHANNEL: Final[str] = "ui"
_WINDOWS_CREATE_NEW_CONSOLE: Final[int] = 0x00000010
_WARNING_INTERVAL_SEC: Final[float] = 10.0
_SHUTDOWN_TIMEOUT_SEC: Final[float] = 3.0


def _auto_create_console() -> bool:
    """Return whether this platform supports a separate Windows console."""
    return sys.platform == "win32"


class WindowHost:
    """Own enabled output windows and send them JSON-lines over stdin."""

    def __init__(
        self,
        *,
        stats_enabled: bool = True,
        reasoning_enabled: bool = True,
        create_new_console: bool | None = None,
        stats_module: str = "features.ui.stats_window",
        reasoning_module: str = "features.ui.reasoning_window",
        stats_command: tuple[str, ...] | None = None,
        reasoning_command: tuple[str, ...] | None = None,
    ) -> None:
        self._stats_enabled = stats_enabled
        self._reasoning_enabled = reasoning_enabled
        self._create_new_console = (
            _auto_create_console() if create_new_console is None else create_new_console
        )
        self._modules = {
            "stats": stats_module,
            "reasoning": reasoning_module,
        }
        self._commands = {
            "stats": stats_command,
            "reasoning": reasoning_command,
        }
        self._processes: dict[str, asyncio.subprocess.Process | None] = {
            "stats": None,
            "reasoning": None,
        }
        self._stderr_tasks: dict[str, asyncio.Task[None]] = {}
        self._last_warning_at: float | None = None
        self._shutdown = False

    async def start(self) -> None:
        """Launch each enabled display process once; failed launches stay headless."""
        if self._shutdown:
            return
        enabled = {
            "stats": self._stats_enabled,
            "reasoning": self._reasoning_enabled,
        }
        for name, is_enabled in enabled.items():
            if not is_enabled or self._is_running(name):
                continue
            command = self._commands[name] or (
                sys.executable,
                "-m",
                self._modules[name],
            )
            try:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=(_WINDOWS_CREATE_NEW_CONSOLE if self._create_new_console else 0),
                )
            except (OSError, ValueError) as error:
                self._warn(f"cannot start {name} window: {error}")
                continue
            self._processes[name] = proc
            self._stderr_tasks[name] = asyncio.create_task(
                self._drain_stderr(proc, f"{name}_window")
            )

    async def _drain_stderr(
        self,
        proc: asyncio.subprocess.Process,
        name: str,
    ) -> None:
        """Read child stderr and log every line.

        If the child crashes, its traceback appears in the logs instead of
        being lost with an unread pipe.
        """
        if proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.bind(channel=_LOGGER_CHANNEL).warning("{} stderr: {}", name, text)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001 — keep the drainer alive
            logger.bind(channel=_LOGGER_CHANNEL).error(
                "{} stderr drain failed: {}",
                name,
                str(e),
            )

    async def send_stats(self, msg: StatsMessage) -> None:
        """Send a stats frame, silently ignoring disabled or dead windows."""
        proc = self._processes["stats"]
        if proc is None or proc.returncode is not None:
            logger.bind(channel=_LOGGER_CHANNEL).warning(
                "stats_window is dead (returncode={}), skip send",
                proc.returncode if proc is not None else None,
            )
            return
        await self._send("stats", IPCEnvelope(type="stats", payload=msg.model_dump()))

    async def send_reasoning(self, msg: ReasoningMessage) -> None:
        """Send one reasoning update, silently ignoring disabled or dead windows."""
        proc = self._processes["reasoning"]
        if proc is None or proc.returncode is not None:
            logger.bind(channel=_LOGGER_CHANNEL).warning(
                "reasoning_window is dead (returncode={}), skip send",
                proc.returncode if proc is not None else None,
            )
            return
        await self._send(
            "reasoning",
            IPCEnvelope(type="reasoning", payload=msg.model_dump()),
        )

    async def _send(self, name: str, envelope: IPCEnvelope) -> None:
        proc = self._processes[name]
        if proc is None or proc.returncode is not None or proc.stdin is None:
            return
        try:
            proc.stdin.write((encode(envelope) + "\n").encode("utf-8"))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as error:
            self._warn(f"{name} window pipe is unavailable: {error}")

    async def shutdown(self, *, timeout: float = _SHUTDOWN_TIMEOUT_SEC) -> None:
        """Gracefully stop children, then terminate any process exceeding timeout."""
        if self._shutdown:
            return
        self._shutdown = True
        for name, task in self._stderr_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._stderr_tasks.clear()
        for name, proc in self._processes.items():
            if proc is None:
                continue
            if proc.returncode is None and proc.stdin is not None:
                try:
                    proc.stdin.write((encode(IPCEnvelope(type="shutdown")) + "\n").encode("utf-8"))
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            if proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=max(0.0, timeout))
                except TimeoutError:
                    proc.terminate()
                    await proc.wait()
            self._processes[name] = None

    def _is_running(self, name: str) -> bool:
        proc = self._processes[name]
        return proc is not None and proc.returncode is None

    def _warn(self, message: str) -> None:
        """Log at most one UI warning in each ten-second interval."""
        now = time.perf_counter()
        if self._last_warning_at is not None:
            elapsed = max(0.0, now - self._last_warning_at)
            if elapsed < _WARNING_INTERVAL_SEC:
                return
        self._last_warning_at = now
        logger.bind(channel=_LOGGER_CHANNEL).warning("{}", message)

    @property
    def stats_running(self) -> bool:
        """Whether the stats child is currently alive."""
        return self._is_running("stats")

    @property
    def reasoning_running(self) -> bool:
        """Whether the reasoning child is currently alive."""
        return self._is_running("reasoning")

    @property
    def stats_pid(self) -> int | None:
        """PID of the stats child, when one has been launched and is alive."""
        proc = self._processes["stats"]
        return proc.pid if proc is not None and proc.returncode is None else None

    @property
    def reasoning_pid(self) -> int | None:
        """PID of the reasoning child, when one has been launched and is alive."""
        proc = self._processes["reasoning"]
        return proc.pid if proc is not None and proc.returncode is None else None
