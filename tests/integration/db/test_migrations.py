"""Tests for the numbered SQLite migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from infra.db.connection import close, connect
from infra.db.migrations.runner import apply_migrations


def test_first_migration_applies_once() -> None:
    """The initial schema applies once and is skipped thereafter."""
    connection = connect(":memory:")
    try:
        assert apply_migrations(connection) == 1
        assert apply_migrations(connection) == 0
    finally:
        close(connection)


def test_migration_runner_creates_all_schema_tables(conn: sqlite3.Connection) -> None:
    """The v1 migration creates all six documented tables."""
    names = {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "players",
        "games",
        "game_players",
        "events",
        "predictions",
        "player_style",
    }.issubset(names)


def test_schema_version_records_version_one(conn: sqlite3.Connection) -> None:
    """The initial schema is recorded in the migration ledger."""
    rows = conn.execute("SELECT version FROM schema_version").fetchall()
    assert [row[0] for row in rows] == [1]


def test_migration_creates_expected_indexes(conn: sqlite3.Connection) -> None:
    """The event lookup index exists after applying the migration."""
    names = {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    assert "idx_events_game_id" in names
    assert {
        "idx_events_ts",
        "idx_events_kind",
        "idx_predictions_game_id",
        "idx_predictions_player_id",
        "idx_game_players_game_id",
    }.issubset(names)


def test_failed_migration_rolls_back_all_its_statements(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """A migration failure rolls back DDL and its version record."""
    migration = tmp_path / "002_broken.sql"
    migration.write_text(
        "CREATE TABLE transient_table (id INTEGER PRIMARY KEY);\n" "NOT VALID SQL;\n",
        encoding="utf-8",
    )
    with pytest.raises(sqlite3.OperationalError):
        apply_migrations(conn, migrations_dir=tmp_path)
    table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        ("transient_table",),
    ).fetchone()
    version = conn.execute("SELECT version FROM schema_version WHERE version=2").fetchone()
    assert table is None
    assert version is None
    assert conn.in_transaction is False


def test_incomplete_migration_statement_rolls_back(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """A migration ending with an incomplete statement is rejected."""
    migration = tmp_path / "002_incomplete.sql"
    migration.write_text("CREATE TABLE unfinished_table (id INTEGER PRIMARY KEY)", encoding="utf-8")
    with pytest.raises(sqlite3.OperationalError, match="incomplete SQL"):
        apply_migrations(conn, migrations_dir=tmp_path)
    assert conn.in_transaction is False
    assert (
        conn.execute("SELECT name FROM sqlite_master WHERE name='unfinished_table'").fetchone()
        is None
    )


def test_invalid_migration_filename_raises(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """SQL files without a numeric version prefix are rejected."""
    (tmp_path / "broken.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid migration filename"):
        apply_migrations(conn, migrations_dir=tmp_path)
