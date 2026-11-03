"""Export SQLite game data as raw JSONL or a simplified ChatML draft for v2.

The ChatML export does not reconstruct game context or train a model. Its user
messages and response schema are placeholders for a separate offline pipeline.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _connect(db_path: str) -> sqlite3.Connection:
    """Open the database and apply any missing schema migrations."""
    from infra.db.connection import connect
    from infra.db.migrations.runner import apply_migrations

    conn = connect(db_path)
    apply_migrations(conn)
    return conn


def _export_raw(conn: sqlite3.Connection, out: Path) -> int:
    """Export events, current predictions, and game players as flat JSONL."""
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in conn.execute(
            "SELECT id, game_id, ts, kind, actor_id, target_id, "
            "phase, day_number, payload_json FROM events ORDER BY id"
        ):
            f.write(
                json.dumps(
                    {
                        "type": "event",
                        "id": row[0],
                        "game_id": row[1],
                        "ts": row[2],
                        "kind": row[3],
                        "actor_id": row[4],
                        "target_id": row[5],
                        "phase": row[6],
                        "day_number": row[7],
                        "payload": json.loads(row[8]),
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
                        "player_id": row[3],
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
                        "player_id": row[2],
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


def _export_chatml(conn: sqlite3.Connection, out: Path) -> int:
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

    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in conn.execute(
            "SELECT id, game_id, ts, player_id, predicted_role, "
            "confidence, reasoning FROM predictions_history "
            "ORDER BY game_id, ts"
        ):
            assistant = json.dumps(
                {
                    "analysis": [
                        {
                            "player_id": row[3],
                            "player_name": f"player_{row[3]}",
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
                                f"player {row[3]} at {row[2]}.",
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
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(args.db)
    try:
        if args.format == "raw":
            n = _export_raw(conn, out)
        else:
            n = _export_chatml(conn, out)
    finally:
        conn.close()
    print(
        f"Exported {n} records to {out} (format={args.format})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
