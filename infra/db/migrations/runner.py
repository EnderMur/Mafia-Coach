"""Apply ordered, versioned SQLite schema migrations."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from infra.db.connection import transaction

_MIGRATION_PATTERN = re.compile(r"^(\d+)_*.+\.sql$")


def apply_migrations(conn: sqlite3.Connection, migrations_dir: Path | None = None) -> int:
    """Apply unapplied SQL migrations in lexical order.

    Args:
        conn: Open SQLite connection to migrate.
        migrations_dir: Directory containing numbered SQL migration files.

    Returns:
        The number of migrations newly applied.
    """
    directory = migrations_dir if migrations_dir is not None else Path(__file__).parent
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "version INTEGER PRIMARY KEY, "
        "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    applied_versions = {int(row[0]) for row in conn.execute("SELECT version FROM schema_version")}
    applied_count = 0

    for migration_path in sorted(directory.glob("*.sql"), key=lambda path: path.name):
        match = _MIGRATION_PATTERN.fullmatch(migration_path.name)
        if match is None:
            raise ValueError(f"invalid migration filename: {migration_path.name}")
        version = int(match.group(1))
        if version in applied_versions:
            continue

        migration_sql = migration_path.read_text(encoding="utf-8")
        with transaction(conn):
            _execute_statements(conn, migration_sql)
            conn.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))
        applied_versions.add(version)
        applied_count += 1

    return applied_count


def _execute_statements(conn: sqlite3.Connection, script: str) -> None:
    """Execute complete SQL statements individually in the active transaction."""
    statement_buffer = ""
    for line in script.splitlines(keepends=True):
        statement_buffer += line
        if sqlite3.complete_statement(statement_buffer):
            statement = statement_buffer.strip()
            if statement:
                conn.execute(statement)
            statement_buffer = ""

    if statement_buffer.strip():
        raise sqlite3.OperationalError("incomplete SQL statement in migration")
