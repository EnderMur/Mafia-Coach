"""Unit tests for Telethon session file helpers."""

from __future__ import annotations

from pathlib import Path

from infra.telegram.session import ensure_session_dir, session_exists


def test_ensure_session_dir_creates_missing_parents(tmp_path: Path) -> None:
    """ensure_session_dir creates nested parent directories when missing."""
    target = tmp_path / "data" / "sessions" / "foo"
    returned = ensure_session_dir(target)
    assert returned == target
    assert target.parent.is_dir()


def test_ensure_session_dir_returns_path_when_parent_exists(tmp_path: Path) -> None:
    """ensure_session_dir returns a Path when the parent directory already exists."""
    target_str = str(tmp_path / "existing_session")
    returned = ensure_session_dir(target_str)
    assert isinstance(returned, Path)
    assert returned == Path(target_str)


def test_session_exists_true_for_appended_session_suffix(tmp_path: Path) -> None:
    """session_exists returns True when <path>.session exists on disk."""
    base = tmp_path / "mafia_coach"
    (tmp_path / "mafia_coach.session").write_bytes(b"")
    assert session_exists(base) is True


def test_session_exists_false_when_missing(tmp_path: Path) -> None:
    """session_exists returns False when neither <path>.session nor <path> exists."""
    base = tmp_path / "missing_session"
    assert session_exists(base) is False


def test_session_exists_true_when_exact_path_exists(tmp_path: Path) -> None:
    """session_exists returns True when the exact file exists without extra suffix."""
    exact = tmp_path / "raw_session_file"
    exact.write_bytes(b"")
    assert session_exists(exact) is True
