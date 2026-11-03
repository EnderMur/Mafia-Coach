"""Export SQLite game data as raw JSONL or a simplified ChatML draft for v2.

The ChatML export does not reconstruct game context or train a model. Its user
messages and response schema are placeholders for a separate offline pipeline.

``--anonymize`` pseudonymizes structured player IDs and names, not free text or
other identifying information. Review every export manually before sharing it.
Raw IDs become ``player_N`` strings; ChatML keeps numeric IDs as pseudonyms.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

# All structured player identity pairs in the event payloads used by the app.
_IDENTITY_FIELDS = (
    ("author_id", "author_name"),
    ("victim_id", "victim_name"),
    ("voter_id", "voter_name"),
    ("target_id", "target_name"),
    ("player_id", "player_name"),
)


def _player_id(value: Any) -> int | None:
    """Reject malformed IDs rather than accidentally exporting them unchanged."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise TypeError("cannot anonymize a non-integer player ID")


def _player_name(value: Any) -> str | None:
    """Validate structured names without exposing invalid values in errors."""
    if value is None or isinstance(value, str):
        return value
    raise TypeError("cannot anonymize a non-string player name")


def _event_payload(value: str) -> dict[str, Any]:
    """Decode an anonymizable event payload, refusing unstructured data."""
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise TypeError("cannot anonymize an event with a non-object payload")
    return payload


def _identity_id(
    payload: dict[str, Any], id_field: str, actor_id: int | None, target_id: int | None
) -> int | None:
    """Prefer the payload ID, falling back to the corresponding event column."""
    fallback = target_id if id_field == "target_id" else actor_id
    return _player_id(payload.get(id_field, fallback))


class _PlayerAliases:
    """Assign per-export aliases, linking unambiguous name-only event references."""

    def __init__(self) -> None:
        self._ids: dict[int, str] = {}
        self._name_ids: dict[tuple[int | None, str], set[int]] = {}
        self._unresolved: dict[tuple[int | None, str], str] = {}
        self._next = 1

    def _new_alias(self) -> str:
        alias = f"player_{self._next}"
        self._next += 1
        return alias

    def player(self, player_id: int | None) -> str | None:
        """Return the same alias for every occurrence of an ID; preserve NULL."""
        if player_id is None:
            return None
        if player_id not in self._ids:
            self._ids[player_id] = self._new_alias()
        return self._ids[player_id]

    def collect_names(
        self,
        game_id: int | None,
        payload: dict[str, Any],
        actor_id: int | None,
        target_id: int | None,
    ) -> None:
        """Index name/ID pairs without assigning aliases ahead of output order."""
        for id_field, name_field in _IDENTITY_FIELDS:
            if id_field not in payload and name_field not in payload:
                continue
            player_id = _identity_id(payload, id_field, actor_id, target_id)
            name = _player_name(payload.get(name_field))
            if player_id is not None and name:
                self._name_ids.setdefault((game_id, name), set()).add(player_id)

    def name(
        self, game_id: int | None, name: str | None, player_id: int | None = None
    ) -> str | None:
        """Alias a name, avoiding guesses when multiple players share it."""
        if not name:
            return name
        if player_id is None:
            candidates = self._name_ids.get((game_id, name), set())
            if len(candidates) == 1:
                player_id = next(iter(candidates))
        if player_id is not None:
            return self.player(player_id)
        key = (game_id, name)
        if key not in self._unresolved:
            self._unresolved[key] = self._new_alias()
        return self._unresolved[key]


def _collect_aliases(conn: sqlite3.Connection) -> _PlayerAliases:
    """Index names from all events before exporting name-only references."""
    aliases = _PlayerAliases()
    for row in conn.execute("SELECT game_id, actor_id, target_id, payload_json FROM events"):
        aliases.collect_names(
            row[0], _event_payload(row[3]), _player_id(row[1]), _player_id(row[2])
        )
    return aliases


def _anonymize_payload(
    payload: dict[str, Any],
    *,
    game_id: int | None,
    actor_id: int | None,
    target_id: int | None,
    aliases: _PlayerAliases,
) -> None:
    """Replace structured player identities, including name-only bot reports."""
    for id_field, name_field in _IDENTITY_FIELDS:
        if id_field not in payload and name_field not in payload:
            continue
        player_id = _identity_id(payload, id_field, actor_id, target_id)
        if id_field in payload:
            payload[id_field] = aliases.player(player_id)
        if name_field in payload:
            payload[name_field] = aliases.name(
                game_id, _player_name(payload[name_field]), player_id
            )

    if "alive_player_names" in payload:
        names = payload["alive_player_names"]
        if not isinstance(names, list):
            raise TypeError("cannot anonymize an invalid alive player list")
        payload["alive_player_names"] = [
            aliases.name(game_id, _player_name(name)) for name in names
        ]
    for field in ("winners", "losers"):
        if field not in payload:
            continue
        entries = payload[field]
        if not isinstance(entries, list):
            raise TypeError("cannot anonymize invalid end-game results")
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[1], str):
                raise ValueError("cannot anonymize an invalid end-game player")
            entry[0] = aliases.name(game_id, _player_name(entry[0]))


def _connect(db_path: str) -> sqlite3.Connection:
    """Open the database and apply any missing schema migrations."""
    from infra.db.connection import connect
    from infra.db.migrations.runner import apply_migrations

    conn = connect(db_path)
    apply_migrations(conn)
    return conn


def _export_raw(conn: sqlite3.Connection, out: Path, anonymize: bool = False) -> int:
    """Export raw JSONL using one snapshot for the anonymization pre-scan and output."""
    if not anonymize or conn.in_transaction:
        return _write_raw(conn, out, anonymize)
    conn.execute("BEGIN")
    try:
        return _write_raw(conn, out, anonymize)
    finally:
        # This transaction is read-only and was started here, not by the caller.
        conn.rollback()


def _write_raw(conn: sqlite3.Connection, out: Path, anonymize: bool) -> int:
    """Write events, current predictions, and game players as flat JSONL."""
    aliases = _collect_aliases(conn) if anonymize else None
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in conn.execute(
            "SELECT id, game_id, ts, kind, actor_id, target_id, "
            "phase, day_number, payload_json FROM events ORDER BY id"
        ):
            actor_id, target_id = row[4], row[5]
            if aliases is None:
                payload = json.loads(row[8])
            else:
                original_actor, original_target = _player_id(actor_id), _player_id(target_id)
                actor_id, target_id = (
                    aliases.player(original_actor),
                    aliases.player(original_target),
                )
                payload = _event_payload(row[8])
                _anonymize_payload(
                    payload,
                    game_id=row[1],
                    actor_id=original_actor,
                    target_id=original_target,
                    aliases=aliases,
                )
            f.write(
                json.dumps(
                    {
                        "type": "event",
                        "id": row[0],
                        "game_id": row[1],
                        "ts": row[2],
                        "kind": row[3],
                        "actor_id": actor_id,
                        "target_id": target_id,
                        "phase": row[6],
                        "day_number": row[7],
                        "payload": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
        for row in conn.execute(
            "SELECT id, game_id, ts, player_id, predicted_role, "
            "confidence, reasoning, correct FROM predictions ORDER BY id"
        ):
            f.write(
                json.dumps(
                    {
                        "type": "prediction",
                        "id": row[0],
                        "game_id": row[1],
                        "ts": row[2],
                        "player_id": aliases.player(_player_id(row[3]))
                        if aliases is not None
                        else row[3],
                        "predicted_role": row[4],
                        "confidence": row[5],
                        "reasoning": row[6],
                        "correct": row[7],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
        for row in conn.execute(
            "SELECT id, game_id, player_id, final_role, survived, "
            "is_winner FROM game_players ORDER BY id"
        ):
            f.write(
                json.dumps(
                    {
                        "type": "game_player",
                        "id": row[0],
                        "game_id": row[1],
                        "player_id": aliases.player(_player_id(row[2]))
                        if aliases is not None
                        else row[2],
                        "final_role": row[3],
                        "survived": row[4],
                        "is_winner": row[5],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def _export_chatml(conn: sqlite3.Connection, out: Path, anonymize: bool = False) -> int:
    """Export prediction history as simplified system/user/assistant dialogues.

    Each line contains a ``messages`` list, ``game_id``, and ``ts``. This v2
    draft uses a placeholder schema and a synthetic user message rather than
    reconstructed context; a full context builder belongs to a later step.
    """
    import jinja2

    from domain.roles import load_roles

    root = Path(__file__).resolve().parent.parent
    reg = load_roles(root / "configs" / "roles.yaml")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(root / "infra" / "llm" / "prompts")),
        autoescape=False,
    )
    system_prompt = env.get_template("system.jinja").render(
        role_names_ru=sorted(r.name_ru for r in reg.all_roles()),
        response_schema_json="{}",  # v1 schema placeholder
    )

    aliases = _PlayerAliases() if anonymize else None
    # Ties must be deterministic when they determine pseudonym assignment.
    order = "ORDER BY game_id, ts, id" if anonymize else "ORDER BY game_id, ts"
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in conn.execute(
            "SELECT id, game_id, ts, player_id, predicted_role, "
            "confidence, reasoning FROM predictions_history " + order
        ):
            if aliases is None:
                player_id = row[3]
                player_name = f"player_{row[3]}"
                user_player = f"player {row[3]}"
            else:
                alias = aliases.player(_player_id(row[3]))
                if alias is None:
                    raise ValueError("cannot anonymize a missing prediction player ID")
                # AnalysisResponse uses an integer player_id, unlike raw JSONL.
                player_id = int(alias.removeprefix("player_"))
                player_name = user_player = alias
            assistant = json.dumps(
                {
                    "analysis": [
                        {
                            "player_id": player_id,
                            "player_name": player_name,
                            "suspected_role": row[4],
                            "confidence": row[5],
                            "reasons": [],
                        }
                    ],
                    "game_read": "",
                    "reasoning": row[6] or "",
                },
                ensure_ascii=False,
            )
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": f"Game {row[1]}, prediction for "
                                f"{user_player} at {row[2]}.",
                            },
                            {"role": "assistant", "content": assistant},
                        ],
                        "game_id": row[1],
                        "ts": row[2],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    """Export the selected dataset format and report the number of records."""
    parser = argparse.ArgumentParser(
        description="Export SQLite data to JSONL (raw or ChatML).",
    )
    parser.add_argument("--db", required=True, help="path to mafia.db")
    parser.add_argument("--out", required=True, help="output .jsonl path")
    parser.add_argument(
        "--format",
        choices=["raw", "chatml"],
        default="raw",
        help="output format: raw or chatml (simplified draft; default: raw)",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="pseudonymize structured player names/IDs; free text may still contain PII",
    )
    args = parser.parse_args(argv)

    if args.anonymize:
        print(
            "WARNING: --anonymize replaces structured player IDs/names only; message text, "
            "reasoning and other fields may contain PII. Review the entire export before sharing.",
            file=sys.stderr,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(args.db)
    try:
        if args.format == "raw":
            n = _export_raw(conn, out, anonymize=args.anonymize)
        else:
            n = _export_chatml(conn, out, anonymize=args.anonymize)
    finally:
        conn.close()
    print(
        f"Exported {n} records to {out} (format={args.format})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
