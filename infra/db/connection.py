"""SQLite connection helpers with explicit transaction support."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def connect(path: str | Path) -> sqlite3.Connection:
    """Open SQLite connection.

    check_same_thread = False позволяет использовать соединение
    из разных потоков (нужно для asyncio.to_thread в async
    контексте). Все вызовы должны быть сериализованы через
    asyncio (один in-flight запрос на loop).
    """
    database_path = str(path)
    if database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        database_path,
        detect_types=0,
        isolation_level=None,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run an explicit immediate transaction, rolling back on errors."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def close(conn: sqlite3.Connection) -> None:
    """Close a SQLite connection."""
    conn.close()
