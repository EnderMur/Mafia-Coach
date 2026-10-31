"""Filesystem helpers for Telethon session files without network I/O."""

from __future__ import annotations

from pathlib import Path


def ensure_session_dir(session_file: str | Path) -> Path:
    """Create parent directory for session file if missing and return Path(session_file).

    Args:
        session_file: Configured path to the Telethon session file (with or without ``.session``).

    Returns:
        ``Path(session_file)`` after ensuring its parent directory exists.
    """
    path = Path(session_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def session_exists(session_file: str | Path) -> bool:
    """Return True if the Telethon session file exists on disk.

    Telethon appends ``.session`` when missing, so both ``<session_file>.session``
    and ``<session_file>`` are checked.

    Args:
        session_file: Configured path to the Telethon session file.

    Returns:
        True when either ``<session_file>.session`` or ``<session_file>`` is an existing file.
    """
    candidate = Path(f"{session_file}.session")
    if candidate.is_file():
        return True
    return Path(session_file).is_file()
