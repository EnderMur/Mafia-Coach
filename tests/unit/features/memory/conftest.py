"""Shared fixtures for features/memory tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from infra.db.connection import close, connect
from infra.db.migrations.runner import apply_migrations


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """Provide an in-memory database with the v1 schema applied."""
    connection = connect(":memory:")
    apply_migrations(connection)
    try:
        yield connection
    finally:
        close(connection)
