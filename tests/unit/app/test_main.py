"""Tests for the package entry point and fatal startup handling."""

from __future__ import annotations

from collections.abc import Coroutine
from typing import Any

import pytest

from app import __main__ as app_main


async def test_run_always_shuts_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """The orchestrator is shut down even when its run loop raises."""
    calls: list[str] = []

    async def fake_bootstrap() -> object:
        calls.append("bootstrap")
        return object()

    class FakeOrchestrator:
        def __init__(self, *, bootstrap_result: object) -> None:
            calls.append("construct")

        async def run(self) -> None:
            calls.append("run")
            raise RuntimeError("scanner failed")

        async def shutdown(self) -> None:
            calls.append("shutdown")

    monkeypatch.setattr(app_main, "bootstrap", fake_bootstrap)
    monkeypatch.setattr(app_main, "Orchestrator", FakeOrchestrator)

    with pytest.raises(RuntimeError, match="scanner failed"):
        await app_main._run()

    assert calls == ["bootstrap", "construct", "run", "shutdown"]


def test_main_normal_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal asyncio.run completion returns normally from main()."""

    def fake_asyncio_run(coroutine: Coroutine[Any, Any, None]) -> None:
        coroutine.close()

    monkeypatch.setattr("app.__main__.asyncio.run", fake_asyncio_run)

    app_main.main()


def test_main_keyboard_interrupt_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl+C does not produce a fatal error or non-zero exit."""

    def interrupted(coroutine: Coroutine[Any, Any, None]) -> None:
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr("app.__main__.asyncio.run", interrupted)

    app_main.main()


def test_main_reports_fatal_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bootstrap/runtime errors are shown on stderr and exit with status one."""

    def fail(coroutine: Coroutine[Any, Any, None]) -> None:
        coroutine.close()
        raise RuntimeError("broken startup")

    monkeypatch.setattr("app.__main__.asyncio.run", fail)

    with pytest.raises(SystemExit) as error:
        app_main.main()

    assert error.value.code == 1
    assert "FATAL: RuntimeError: broken startup" in capsys.readouterr().err
