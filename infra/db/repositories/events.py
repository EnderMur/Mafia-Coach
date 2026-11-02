"""Repository for serialized domain events."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from domain.events import (
    AliveListEvent,
    AliveListPayload,
    AnyEvent,
    DeathEvent,
    MessageEvent,
    PhaseChangeEvent,
    RoleRevealEvent,
    VoteEvent,
)
from domain.exceptions import ConfigError


class EventsRepository:
    """Persist domain events and reconstruct them from the event log."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the repository with a SQLite connection."""
        self._conn = conn

    def insert_event(self, event: AnyEvent) -> int:
        """Serialize and insert a domain event, returning its row id."""
        actor_id: int | None = None
        target_id: int | None = None
        phase: str | None = None
        day_number: int | None = None

        if isinstance(event, MessageEvent):
            actor_id = event.payload.author_id
        elif isinstance(event, DeathEvent):
            actor_id = event.payload.victim_id
        elif isinstance(event, VoteEvent):
            actor_id = event.payload.voter_id
            target_id = event.payload.target_id
        elif isinstance(event, RoleRevealEvent):
            actor_id = event.payload.player_id
        elif isinstance(event, PhaseChangeEvent):
            phase = event.payload.to_phase.value
            day_number = event.payload.day_number
        elif isinstance(event, AliveListEvent):
            # alive_list carries no actor/target/phase/day metadata —
            # actor_id, target_id, phase, day_number stay NULL.
            pass

        cursor = self._conn.execute(
            "INSERT INTO events(game_id, ts, kind, actor_id, target_id, phase, "
            "day_number, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.game_id,
                event.ts.isoformat(),
                event.kind,
                actor_id,
                target_id,
                phase,
                day_number,
                event.payload.model_dump_json(),
            ),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def list_events(self, game_id: int) -> list[AnyEvent]:
        """Return a game's events in insertion order."""
        rows = self._conn.execute(
            "SELECT * FROM events WHERE game_id = ? ORDER BY id ASC", (game_id,)
        ).fetchall()
        events: list[AnyEvent] = []
        for row in rows:
            kind = str(row["kind"])
            event_data: dict[str, Any] = {
                "ts": row["ts"],
                "game_id": row["game_id"],
                "payload": json.loads(row["payload_json"]),
            }
            event: AnyEvent
            if kind == "message":
                event = MessageEvent.model_validate(event_data)
            elif kind == "death":
                event = DeathEvent.model_validate(event_data)
            elif kind == "vote":
                event = VoteEvent.model_validate(event_data)
            elif kind == "role_reveal":
                event = RoleRevealEvent.model_validate(event_data)
            elif kind == "phase_change":
                event = PhaseChangeEvent.model_validate(event_data)
            elif kind == "alive_list":
                payload = AliveListPayload.model_validate_json(row["payload_json"])
                event = AliveListEvent.model_validate(
                    {
                        "ts": row["ts"],
                        "game_id": row["game_id"],
                        "payload": payload,
                    }
                )
            else:
                raise ConfigError(f"unknown event kind: {kind}")
            events.append(event)
        return events

    def count_by_kind(self, game_id: int) -> dict[str, int]:
        """Return counts of event rows grouped by kind for a game."""
        rows = self._conn.execute(
            "SELECT kind, COUNT(*) FROM events WHERE game_id = ? GROUP BY kind",
            (game_id,),
        ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}
