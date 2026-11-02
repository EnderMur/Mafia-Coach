"""Test fixtures for UI window modules."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _disable_conout_reopen_on_win32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disable CONOUT$ reopening in tests.

    On Windows, main() opens CONOUT$ and replaces sys.stdout/stderr.
    This breaks pytest's capsys and monkeypatched spies. For unit
    tests we test rendering logic, not the Windows console handle,
    so we replace _reopen_windows_console with a no-op.
    """
    if sys.platform == "win32":
        from features.ui import reasoning_window, stats_window

        monkeypatch.setattr(stats_window, "_reopen_windows_console", lambda: None)
        monkeypatch.setattr(reasoning_window, "_reopen_windows_console", lambda: None)
