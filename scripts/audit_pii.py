"""Audit git history for PII before public release.

Read-only. Checks:

- Forbidden files in git history (any branch): ``*.session``, ``*.db``,
  ``configs/settings.yaml``, ``data/``, ``logs/``.
- Forbidden files in current index (``git ls-files``).
- ``tests/fixtures/`` — grep for real names from ``data/dumps/`` (if dumps
  are available locally).

Empty ``.gitkeep`` directory placeholders under ``data/`` and ``logs/`` are
allowed: they are tracked on purpose and carry no data.

Exit 1 if anything found, 0 if clean. Run from the repository root before
switching the repo Private → Public (README §13): ``python scripts/audit_pii.py``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Final

FORBIDDEN_PATTERNS: Final[tuple[str, ...]] = (
    "*.session",
    "*.session-journal",
    "*.db",
    "*.db-journal",
    "*.sqlite",
    "*.sqlite3",
    "configs/settings.yaml",
    "data/",
    "logs/",
)

#: Empty directory placeholders are tracked on purpose and contain no PII.
_ALLOWED_BASENAMES: Final[frozenset[str]] = frozenset({".gitkeep"})


def _git(*args: str) -> str:
    """Run a read-only git command and return its stdout ("" on failure)."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def _is_forbidden(path: str) -> bool:
    """Match one path against FORBIDDEN_PATTERNS (suffix/prefix/exact)."""
    if Path(path).name in _ALLOWED_BASENAMES:
        return False
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.endswith("/"):
            # Directory prefix: data/, logs/.
            if path.startswith(pattern):
                return True
        elif pattern.startswith("*"):
            # Suffix: *.session, *.db, ...
            if path.endswith(pattern[1:]):
                return True
        elif path == pattern or path.startswith(pattern + "/"):
            return True
    return False


def _check_history() -> list[str]:
    """Files ever committed (any branch)."""
    found: list[str] = []
    for pattern in FORBIDDEN_PATTERNS:
        out = _git(
            "log",
            "--all",
            "--full-history",
            "--name-only",
            "--pretty=format:",
            "--",
            pattern,
        )
        files = {line.strip() for line in out.splitlines() if line.strip()}
        found.extend(f for f in files if _is_forbidden(f))
    return sorted(set(found))


def _check_index() -> list[str]:
    """Files currently tracked."""
    tracked = _git("ls-files").splitlines()
    return sorted({t for t in tracked if _is_forbidden(t)})


def _check_fixtures() -> list[str]:
    """Grep fixtures for likely real names from data/dumps (chat*.json)."""
    dumps = sorted(Path("data/dumps").glob("chat*.json"))
    if not dumps:
        return []
    names: set[str] = set()
    for dump in dumps:
        try:
            data: dict[str, Any] = json.loads(dump.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for m in data.get("messages", []):
            n = m.get("sender_name")
            if n and isinstance(n, str) and len(n) >= 4:
                names.add(n)
    if not names:
        return []
    findings: list[str] = []
    fixtures = sorted(Path("tests/fixtures").rglob("*.json"))
    for fx in fixtures:
        try:
            content = fx.read_text(encoding="utf-8")
        except OSError:
            continue
        for n in sorted(names):
            if n in content:
                findings.append(f"{fx}: contains {n!r}")
                break
    return findings


def main() -> int:
    """Print findings and return 1 if any PII was found, 0 when clean."""
    issues: list[str] = []
    hist = _check_history()
    if hist:
        issues.append("HISTORY (git log --all): " + ", ".join(hist))
    idx = _check_index()
    if idx:
        issues.append("INDEX (git ls-files): " + ", ".join(idx))
    fix = _check_fixtures()
    if fix:
        issues.append("FIXTURES: " + "; ".join(fix))
    if issues:
        print("PII FOUND:")
        for line in issues:
            print(f"  {line}")
        return 1
    print("Clean: no PII in git history, index, or fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
