"""Tests for checking fixture names against local chat dumps."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

# scripts is a CLI namespace, not an installed package.
audit_pii = importlib.import_module("scripts.audit_pii")


@pytest.fixture
def audit_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate dumps and fixtures from the working tree and other tests."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "dumps").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    return tmp_path


def _write_dump(root: Path, filename: str, data: dict[str, object]) -> None:
    (root / "data" / "dumps" / filename).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def _write_fixture(root: Path, filename: str, content: str) -> None:
    (root / "tests" / "fixtures" / filename).write_text(content, encoding="utf-8")


def test_audit_pii_skips_anonymized_dump(audit_tree: Path) -> None:
    """Names from a marked dump are skipped without suppressing other dumps."""
    _write_dump(
        audit_tree,
        "chat_anon.json",
        {"anonymized": True, "messages": [{"sender_name": "AnonymizedName"}]},
    )
    _write_dump(
        audit_tree,
        "chat_raw.json",
        {"anonymized": False, "messages": [{"sender_name": "RealName"}]},
    )
    _write_fixture(audit_tree, "anonymous.json", '"AnonymizedName"')
    _write_fixture(audit_tree, "raw.json", '"RealName"')

    assert audit_pii._check_fixtures() == [
        f"{Path('tests/fixtures/raw.json')}: contains 'RealName'"
    ]


def test_audit_pii_skips_user_placeholder(audit_tree: Path) -> None:
    """Only exact user_<digits> names are placeholders; real names still match."""
    _write_dump(
        audit_tree,
        "chat_mixed.json",
        {
            "messages": [
                {"sender_name": "user_1"},
                {"sender_name": "user_123"},
                {"sender_name": "user_1_real"},
            ]
        },
    )
    _write_fixture(audit_tree, "placeholder.json", '"user_1"')
    _write_fixture(audit_tree, "real.json", '"user_1_real"')

    assert audit_pii._check_fixtures() == [
        f"{Path('tests/fixtures/real.json')}: contains 'user_1_real'"
    ]


@pytest.mark.parametrize("anonymized", [False, 1, "true", None])
def test_audit_pii_only_skips_boolean_true(audit_tree: Path, anonymized: object) -> None:
    """Truthy or missing metadata must not silently suppress a real name."""
    _write_dump(
        audit_tree,
        "chat_raw.json",
        {"anonymized": anonymized, "messages": [{"sender_name": "RealName"}]},
    )
    _write_fixture(audit_tree, "raw.json", '"RealName"')

    assert audit_pii._check_fixtures() == [
        f"{Path('tests/fixtures/raw.json')}: contains 'RealName'"
    ]
