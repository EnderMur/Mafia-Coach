"""Build an SFT (OpenAI messages / ChatML) dataset from SQLite game data.

For every prediction row (from ``predictions`` by default, or
``predictions_history`` with ``--source history``) the script:

1. resolves ground truth (``game_players.final_role``) and correctness;
2. reconstructs the game state at the prediction moment (events with
   ``ts <= prediction.ts`` applied tolerantly);
3. renders the same system/user prompt the live analyzer produces;
4. writes one JSONL line with ``[system, user, assistant]`` messages, where
   the assistant answer is the correct JSON response (the model's own answer
   when it was right, or the ground-truth role when it was wrong).

Only examples with a resolvable ground truth are kept. ``--anonymize``
replaces structured player ids/names with ``player_N`` aliases; free text is
left untouched — review the export before sharing it.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jinja2
from loguru import logger

from domain.club_patterns import load_club_patterns
from domain.exceptions import ConfigError
from domain.roles import RoleRegistry, load_roles
from features.analyzer.context_builder import PredictionSnapshot
from features.finetune.anonymizer import PlayerAliasMap
from features.finetune.dataset_builder import build_example
from features.finetune.game_reconstructor import reconstruct_game_state
from features.finetune.models import PredictionRow
from infra.db.connection import connect
from infra.db.migrations.runner import apply_migrations
from infra.db.repositories.events import EventsRepository

LOG_CHANNEL: str = "finetune.prepare"

_DEFAULT_ROLES = "configs/roles.yaml"
_DEFAULT_CLUB_PATTERNS = "configs/club_patterns.yaml"
_DEFAULT_MAX_MESSAGES = 200


def _connect(db_path: str) -> sqlite3.Connection:
    """Open the database and apply any missing schema migrations."""
    conn = connect(db_path)
    apply_migrations(conn)
    return conn


def _load_club_patterns(path_str: str) -> list[dict[str, object]]:
    """Load club patterns from YAML, falling back to the example file."""
    primary = Path(path_str)
    if primary.is_file():
        patterns = load_club_patterns(primary)
    else:
        example = primary.with_name(primary.stem + ".example" + primary.suffix)
        if not example.is_file():
            logger.bind(channel=LOG_CHANNEL).warning(
                "club_patterns: neither {} nor {} found — using empty list",
                primary,
                example,
            )
            return []
        logger.bind(channel=LOG_CHANNEL).warning(
            "club_patterns: using example file ({}); create {} for custom patterns",
            example,
            primary,
        )
        patterns = load_club_patterns(example)
    return [
        {
            "id": p.id,
            "trigger": p.trigger,
            "conclusion_role": p.conclusion_role,
            "confidence_tier": p.confidence_tier,
            "requires_confirmation": p.requires_confirmation,
            "description": p.description,
        }
        for p in patterns.patterns
    ]


def _load_aliases(conn: sqlite3.Connection) -> PlayerAliasMap:
    """Collect player identities from every event and prediction row."""
    aliases = PlayerAliasMap()
    for row in conn.execute("SELECT game_id, actor_id, target_id, payload_json FROM events"):
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError) as error:
            logger.bind(channel=LOG_CHANNEL).warning(
                "aliases: skipping event {} with invalid payload: {}",
                row.get("id", "?"),
                error,
            )
            continue
        aliases.collect_raw_event(
            game_id=int(row["game_id"]),
            payload=payload,
            actor_id=row["actor_id"],
            target_id=row["target_id"],
        )
    for row in conn.execute(
        "SELECT player_id FROM predictions UNION SELECT player_id FROM predictions_history"
    ):
        aliases.collect_player_id(int(row["player_id"]))
    for row in conn.execute("SELECT player_id FROM game_players"):
        aliases.collect_player_id(int(row["player_id"]))
    return aliases


def _prediction_rows(
    conn: sqlite3.Connection, source: str
) -> list[sqlite3.Row]:
    """Return prediction rows in chronological order."""
    table = "predictions" if source == "current" else "predictions_history"
    cursor = conn.execute(
        f"SELECT * FROM {table} ORDER BY game_id, ts, id"
    )
    return list(cursor.fetchall())


def _final_roles_map(conn: sqlite3.Connection) -> dict[tuple[int, int], str]:
    """Map ``(game_id, player_id)`` to the stored final role."""
    rows = conn.execute("SELECT game_id, player_id, final_role FROM game_players")
    return {
        (int(row["game_id"]), int(row["player_id"])): row["final_role"]
        for row in rows
        if row["final_role"] is not None
    }


def _canonical_role(
    role_name: str, registry: RoleRegistry
) -> str | None:
    """Resolve a stored role name to a canonical ``name_ru``."""
    role = registry.resolve_alias(role_name)
    return role.name_ru if role is not None else None


def _compute_correct(
    prediction: PredictionRow,
    final_role: str,
    row_correct: bool | None,
) -> bool:
    """Determine whether the prediction matched the ground truth."""
    if row_correct is not None:
        return row_correct
    return prediction.predicted_role == final_role


def _previous_snapshots(
    conn: sqlite3.Connection,
    *,
    game_id: int,
    prediction_ts: datetime,
) -> list[PredictionSnapshot]:
    """Load earlier predictions of the game as anchor snapshots."""
    rows = conn.execute(
        "SELECT ts, player_id, predicted_role, confidence "
        "FROM predictions_history WHERE game_id = ? ORDER BY ts ASC, id ASC",
        (game_id,),
    ).fetchall()
    latest: dict[int, PredictionSnapshot] = {}
    for row in rows:
        ts = _parse_ts(row["ts"])
        if ts >= prediction_ts:
            break
        player_id = int(row["player_id"])
        role = row["predicted_role"]
        if role is None:
            continue
        latest[player_id] = PredictionSnapshot(
            player_id=player_id,
            suspected_role=str(role),
            confidence=float(row["confidence"]),
        )
    return list(latest.values())


def _parse_ts(raw: Any) -> datetime:
    """Parse a SQLite timestamp into a timezone-aware datetime."""
    ts = datetime.fromisoformat(str(raw))
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def _write_parquet(records: list[dict[str, Any]], parquet_path: Path) -> None:
    """Write records to Parquet (optional, requires pyarrow)."""
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as error:
        logger.bind(channel=LOG_CHANNEL).warning(
            "pyarrow не установлен — parquet не записан: {}", error
        )
        return
    table = pa.Table.from_pylist(records)
    pq.write_table(table, parquet_path)
    logger.bind(channel=LOG_CHANNEL).info("parquet written: {}", parquet_path)


def _render_summary(
    records: int, skipped: Counter[str], out: Path
) -> str:
    """Build the CLI summary line."""
    skipped_text = ", ".join(f"{key}={value}" for key, value in sorted(skipped.items()))
    summary = f"Prepared {records} examples → {out}"
    if skipped_text:
        summary += f" (skipped: {skipped_text})"
    return summary


def prepare_dataset(
    *,
    conn: sqlite3.Connection,
    out: Path,
    source: str,
    only_correct: bool = False,
    min_confidence: float = 0.0,
    anonymize: bool = False,
    parquet: bool = False,
    registry: RoleRegistry,
    club_patterns: list[dict[str, object]],
    jinja_env: jinja2.Environment,
    max_messages: int,
) -> tuple[int, Counter[str]]:
    """Run the dataset preparation and return the record count and skip stats."""
    out.parent.mkdir(parents=True, exist_ok=True)
    final_roles = _final_roles_map(conn)
    aliases = _load_aliases(conn) if anonymize else None

    events_repo = EventsRepository(conn)
    events_cache: dict[int, list[Any]] = {}

    records = 0
    skipped: Counter[str] = Counter()
    history_signatures: dict[tuple[int, int], tuple[Any, ...]] = {}

    with out.open("w", encoding="utf-8") as handle:
        for row in _prediction_rows(conn, source):
            prediction = PredictionRow.from_sqlite_row(row)
            if prediction.predicted_role is None:
                skipped["no_role_predicted"] += 1
                continue
            final_role_raw = final_roles.get((prediction.game_id, prediction.player_id))
            if final_role_raw is None:
                skipped["no_ground_truth"] += 1
                continue
            final_role = _canonical_role(final_role_raw, registry)
            if final_role is None:
                skipped["unknown_ground_truth_role"] += 1
                continue
            correct = _compute_correct(prediction, final_role, prediction.correct)
            if only_correct and not correct:
                skipped["incorrect"] += 1
                continue
            if prediction.confidence < min_confidence:
                skipped["low_confidence"] += 1
                continue
            if source == "history":
                signature = (
                    prediction.predicted_role,
                    prediction.confidence,
                    prediction.reason,
                )
                key = (prediction.game_id, prediction.player_id)
                if history_signatures.get(key) == signature:
                    skipped["duplicate_history"] += 1
                    continue
                history_signatures[key] = signature

            if prediction.game_id not in events_cache:
                events_cache[prediction.game_id] = events_repo.list_events(
                    prediction.game_id
                )
            game_events = events_cache[prediction.game_id]
            relevant_events = [
                event for event in game_events if event.ts <= prediction.ts
            ]
            game_state = reconstruct_game_state(
                events=relevant_events,
                game_id=prediction.game_id,
                registry=registry,
            )
            previous = _previous_snapshots(
                conn, game_id=prediction.game_id, prediction_ts=prediction.ts
            )
            example = build_example(
                game_state=game_state,
                prediction=prediction,
                previous_snapshots=previous,
                registry=registry,
                jinja_env=jinja_env,
                club_patterns=club_patterns,
                final_role=final_role,
                correct=correct,
                source=source,
                max_messages=max_messages,
                aliases=aliases,
            )
            if example is None:
                skipped["player_missing"] += 1
                continue
            handle.write(
                json.dumps(example.model_dump(mode="json"), ensure_ascii=False) + "\n"
            )
            records += 1

    if parquet:
        parquet_path = out.with_suffix(".parquet")
        records_raw = [
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _write_parquet(records_raw, parquet_path)

    return records, skipped


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Build an SFT dataset (OpenAI messages) from SQLite game data.",
    )
    parser.add_argument("--db", required=True, help="path to mafia.db")
    parser.add_argument("--out", required=True, help="output .jsonl path")
    parser.add_argument(
        "--source",
        choices=["current", "history"],
        default="current",
        help="predictions table: current (labeled) or history (timeline)",
    )
    parser.add_argument(
        "--only-correct",
        action="store_true",
        help="keep only predictions marked correct",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="drop predictions below this confidence (default: 0.0)",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="pseudonymize structured player ids/names; free text may still contain PII",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="also write a .parquet copy (requires pyarrow)",
    )
    parser.add_argument(
        "--roles",
        default=_DEFAULT_ROLES,
        help=f"roles YAML (default: {_DEFAULT_ROLES})",
    )
    parser.add_argument(
        "--club-patterns",
        default=_DEFAULT_CLUB_PATTERNS,
        help=f"club patterns YAML (default: {_DEFAULT_CLUB_PATTERNS})",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=_DEFAULT_MAX_MESSAGES,
        help="message window size for context reconstruction",
    )
    args = parser.parse_args(argv)

    if args.anonymize:
        logger.bind(channel=LOG_CHANNEL).warning(
            "--anonymize replaces structured player ids/names only; message text, "
            "reasoning and other fields may contain PII. Review the export before sharing."
        )

    out = Path(args.out)
    try:
        registry = load_roles(args.roles)
    except ConfigError as error:
        logger.bind(channel=LOG_CHANNEL).error("roles: {}", error)
        return 1
    club_patterns = _load_club_patterns(args.club_patterns)
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("infra/llm/prompts"),
        autoescape=False,
    )
    conn = _connect(args.db)
    try:
        records, skipped = prepare_dataset(
            conn=conn,
            out=out,
            source=args.source,
            only_correct=args.only_correct,
            min_confidence=args.min_confidence,
            anonymize=args.anonymize,
            parquet=args.parquet,
            registry=registry,
            club_patterns=club_patterns,
            jinja_env=jinja_env,
            max_messages=args.max_messages,
        )
    finally:
        conn.close()
    print(_render_summary(records, skipped, out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
