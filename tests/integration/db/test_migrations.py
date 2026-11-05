"""Tests for the numbered SQLite migration runner."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from infra.db.connection import close, connect
from infra.db.migrations.runner import apply_migrations


def test_first_migration_applies_once() -> None:
    """The initial schema applies once and is skipped thereafter."""
    connection = connect(":memory:")
    try:
        assert apply_migrations(connection) == 6
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
    assert [row[0] for row in rows] == [1, 2, 3, 4, 5, 6]


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
    migration = tmp_path / "007_broken.sql"
    migration.write_text(
        "CREATE TABLE transient_table (id INTEGER PRIMARY KEY);\nNOT VALID SQL;\n",
        encoding="utf-8",
    )
    with pytest.raises(sqlite3.OperationalError):
        apply_migrations(conn, migrations_dir=tmp_path)
    table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        ("transient_table",),
    ).fetchone()
    version = conn.execute("SELECT version FROM schema_version WHERE version=7").fetchone()
    assert table is None
    assert version is None
    assert conn.in_transaction is False


def test_incomplete_migration_statement_rolls_back(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """A migration ending with an incomplete statement is rejected."""
    migration = tmp_path / "007_incomplete.sql"
    migration.write_text("CREATE TABLE unfinished_table (id INTEGER PRIMARY KEY)", encoding="utf-8")
    with pytest.raises(sqlite3.OperationalError, match=re.escape("incomplete SQL")):
        apply_migrations(conn, migrations_dir=tmp_path)
    assert conn.in_transaction is False
    assert (
        conn.execute("SELECT name FROM sqlite_master WHERE name='unfinished_table'").fetchone()
        is None
    )


def test_invalid_migration_filename_raises(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """SQL files without a numeric version prefix are rejected."""
    (tmp_path / "broken.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("invalid migration filename")):
        apply_migrations(conn, migrations_dir=tmp_path)


def test_migration_002_creates_history() -> None:
    connection = connect(":memory:")
    try:
        assert apply_migrations(connection) == 6
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"predictions", "predictions_history"} <= tables
        assert connection.execute("SELECT max(version) FROM schema_version").fetchone()[0] == 6
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()


def test_migration_002_dedupes_predictions() -> None:
    migrations = Path("infra/db/migrations")
    connection = connect(":memory:")
    try:
        connection.executescript((migrations / "001_init.sql").read_text(encoding="utf-8"))
        connection.execute("INSERT INTO games(id) VALUES (1)")
        connection.execute("INSERT INTO players(id) VALUES (2)")
        for role in ("first", "second", "latest"):
            connection.execute(
                "INSERT INTO predictions(game_id, player_id, ts, predicted_role, confidence) VALUES (1, 2, CURRENT_TIMESTAMP, ?, 0.5)",
                (role,),
            )
        connection.executescript(
            (migrations / "002_predictions_split.sql").read_text(encoding="utf-8")
        )
        assert [row[0] for row in connection.execute("SELECT predicted_role FROM predictions")] == [
            "latest"
        ]
        assert connection.execute("SELECT COUNT(*) FROM predictions_history").fetchone()[0] == 3
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()


def test_migration_002_dedupes_via_runner(tmp_path: Path) -> None:
    """The transactional runner also upgrades populated databases with FK checks on."""
    migrations = Path("infra/db/migrations")
    first = next(migrations.glob("001*.sql"))
    (tmp_path / first.name).write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
    conn = connect(":memory:")
    try:
        assert apply_migrations(conn, tmp_path) == 1
        conn.execute("INSERT INTO games(id) VALUES (1)")
        conn.execute("INSERT INTO players(id) VALUES (2)")
        for role in ("first", "latest"):
            conn.execute(
                "INSERT INTO predictions(game_id, player_id, ts, predicted_role, confidence) "
                "VALUES (1, 2, CURRENT_TIMESTAMP, ?, 0.5)",
                (role,),
            )
        second = migrations / "002_predictions_split.sql"
        (tmp_path / second.name).write_text(second.read_text(encoding="utf-8"), encoding="utf-8")
        assert apply_migrations(conn, tmp_path) == 1
        assert conn.execute("SELECT predicted_role FROM predictions").fetchone()[0] == "latest"
        assert conn.execute("SELECT count(*) FROM predictions_history").fetchone()[0] == 2
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_migration_002_filters_invalid_legacy_rows(tmp_path: Path) -> None:
    """Keep latest valid duplicate, synthesize missing times, and discard invalid rows."""
    source = Path("infra/db/migrations")
    first = source / "001_init.sql"
    (tmp_path / first.name).write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
    conn = connect(":memory:")
    try:
        assert apply_migrations(conn, tmp_path) == 1
        conn.execute("INSERT INTO games(id) VALUES (1)")
        conn.execute("INSERT INTO players(id) VALUES (2)")
        rows = [
            (1, 1, 2, "2020-01-01 12:00:00", "first", 0.4),
            (2, 1, 2, None, "latest valid", 0.8),
            (3, 1, 2, "2020-01-01 12:00:00", None, 0.6),
            (4, 1, 2, "2020-01-01 12:00:00", "invalid", None),
            (5, None, 2, "2020-01-01 12:00:00", "invalid", 0.5),
            (6, 1, None, "2020-01-01 12:00:00", "invalid", 0.5),
        ]
        conn.executemany(
            "INSERT INTO predictions(id, game_id, player_id, ts, predicted_role, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        second = source / "002_predictions_split.sql"
        (tmp_path / second.name).write_text(second.read_text(encoding="utf-8"), encoding="utf-8")
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert apply_migrations(conn, tmp_path) == 1
        assert apply_migrations(conn, tmp_path) == 0
        current = conn.execute("SELECT id, predicted_role, ts FROM predictions").fetchall()
        assert len(current) == 1
        assert current[0][0:2] == (2, "latest valid")
        assert current[0][2] is not None
        history = conn.execute("SELECT id, ts FROM predictions_history ORDER BY id").fetchall()
        assert [row[0] for row in history] == [1, 2]
        assert history[1][1] is not None
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_migration_003_adds_is_winner_column() -> None:
    """The ground-truth winner flag is added by migration 003 as a nullable INTEGER."""
    connection = connect(":memory:")
    try:
        assert apply_migrations(connection) == 6
        assert apply_migrations(connection) == 0
        columns = {row[1]: row[2] for row in connection.execute("PRAGMA table_info(game_players)")}
        assert "is_winner" in columns
        assert columns["is_winner"] == "INTEGER"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        close(connection)


def test_migration_004(conn: sqlite3.Connection) -> None:
    """Participation uniqueness is enforced by an idempotent schema migration."""
    indexes = {row[1]: row[2] for row in conn.execute("PRAGMA index_list(game_players)")}
    assert indexes["idx_game_players_pair"] == 1
    sql = Path("infra/db/migrations/004_game_players_unique.sql").read_text()
    conn.executescript(sql)
    assert apply_migrations(conn) == 0
