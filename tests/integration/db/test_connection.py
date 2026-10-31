"""Tests for connection setup and transaction helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from infra.db.connection import close, connect, transaction


def test_connect_memory_returns_row_connection() -> None:
    """In-memory connections use sqlite rows and autocommit mode."""
    connection = connect(":memory:")
    try:
        assert isinstance(connection, sqlite3.Connection)
        assert connection.row_factory is sqlite3.Row
        assert connection.isolation_level is None
    finally:
        close(connection)


def test_connect_enables_foreign_keys() -> None:
    """Foreign key enforcement is enabled on new connections."""
    connection = connect(":memory:")
    try:
        result = connection.execute("PRAGMA foreign_keys").fetchone()
        assert result is not None
        assert result[0] == 1
    finally:
        close(connection)


def test_connect_creates_parent_directory(tmp_path: Path) -> None:
    """A file-backed connection creates missing parent directories."""
    database_path = tmp_path / "sub" / "test.db"
    connection = connect(database_path)
    try:
        assert database_path.parent.is_dir()
        assert database_path.is_file()
    finally:
        close(connection)


def test_transaction_commits_on_success() -> None:
    """Successful explicit transactions persist their writes."""
    connection = connect(":memory:")
    try:
        connection.execute("CREATE TABLE sample (value INTEGER)")
        with transaction(connection) as active_connection:
            active_connection.execute("INSERT INTO sample VALUES (1)")
        assert connection.execute("SELECT value FROM sample").fetchone()[0] == 1
        assert connection.in_transaction is False
    finally:
        close(connection)


def test_transaction_rolls_back_on_exception() -> None:
    """Exceptions roll back all writes in the active transaction."""
    connection = connect(":memory:")
    try:
        connection.execute("CREATE TABLE sample (value INTEGER)")
        with pytest.raises(RuntimeError, match="abort"):
            with transaction(connection):
                connection.execute("INSERT INTO sample VALUES (1)")
                raise RuntimeError("abort")
        assert connection.execute("SELECT COUNT(*) FROM sample").fetchone()[0] == 0
        assert connection.in_transaction is False
    finally:
        close(connection)
