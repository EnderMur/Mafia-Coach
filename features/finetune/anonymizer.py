"""Player pseudonymisation for fine-tune dataset exports.

Mirrors the policy of ``scripts/export_dataset.py``: structured player ids and
names are replaced with stable ``player_N`` aliases; free text (message bodies,
model reasoning) is left untouched and the caller is responsible for reviewing
the export before sharing it.

The alias map is collected from the whole database before any example is
rendered, so the same real player always receives the same alias inside one
export. Name-only references (bot alive lists, votes without ids) resolve to
the id alias when the name is unambiguous within the game, and to a fresh
alias otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from domain.events import AnyEvent
from features.analyzer.context_builder import (
    DeathInfo,
    LLMContext,
    MessageInfo,
    PlayerInfo,
    PredictionSnapshot,
    RevealedPlayer,
    VoteInfo,
)

#: Structured identity fields present in event payloads.
_IDENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("author_id", "author_name"),
    ("victim_id", "victim_name"),
    ("voter_id", "voter_name"),
    ("target_id", "target_name"),
    ("player_id", "player_name"),
)

#: Bot fallback names ``user_<id>`` carry the real Telegram id (see
#: ``domain.game_state._placeholder_player_id``); treat them as id
#: references so they do not consume alias slots.
_USER_ID_RE = re.compile(r"^user_(\d+)$")


class PlayerAliasMap:
    """Assign per-export ``player_N`` aliases to real player identities."""

    def __init__(self) -> None:
        self._id_to_alias: dict[int, str] = {}
        self._name_to_ids: dict[tuple[int | None, str], set[int]] = {}
        self._unresolved_names: dict[tuple[int | None, str], str] = {}
        self._next = 1

    def _new_alias(self) -> str:
        alias = f"player_{self._next}"
        self._next += 1
        return alias

    def alias_for_id(self, player_id: int | None) -> str | None:
        """Return the stable alias for a player id; ``None`` passes through."""
        if player_id is None:
            return None
        if player_id not in self._id_to_alias:
            self._id_to_alias[player_id] = self._new_alias()
        return self._id_to_alias[player_id]

    def remap_id(self, player_id: int | None) -> int | None:
        """Map a real id to the stable positive int used in outputs."""
        alias = self.alias_for_id(player_id)
        if alias is None:
            return None
        return int(alias.removeprefix("player_"))

    def alias_for_name(
        self,
        game_id: int | None,
        name: str | None,
        player_id: int | None = None,
    ) -> str | None:
        """Return an alias for a display name.

        When the name is unambiguous within the game (maps to exactly one
        known id) the id's alias is used; otherwise a fresh alias is assigned
        so two different people who share a display name never merge.
        """
        if not name:
            return name
        if player_id is None:
            candidates = self._name_to_ids.get((game_id, name), set())
            if len(candidates) == 1:
                player_id = next(iter(candidates))
        if player_id is not None:
            return self.alias_for_id(player_id)
        key = (game_id, name)
        if key not in self._unresolved_names:
            self._unresolved_names[key] = self._new_alias()
        return self._unresolved_names[key]

    def collect_events(self, events: Iterable[AnyEvent]) -> None:
        """Index every player identity found in event payloads."""
        for event in events:
            self._collect_payload(
                game_id=event.game_id,
                payload=event.payload,
                actor_id=_actor_id(event),
                target_id=_target_id(event),
            )

    def collect_raw_event(
        self,
        *,
        game_id: int | None,
        payload: Any,
        actor_id: int | None,
        target_id: int | None,
    ) -> None:
        """Index a raw event row (``game_id`` + decoded payload + columns)."""
        self._collect_payload(
            game_id=game_id,
            payload=payload,
            actor_id=actor_id,
            target_id=target_id,
        )

    def collect_player_id(self, player_id: int | None) -> None:
        """Reserve an alias for a bare player id (predictions, game players)."""
        self.alias_for_id(player_id)

    def _collect_payload(
        self,
        *,
        game_id: int | None,
        payload: Any,
        actor_id: int | None,
        target_id: int | None,
    ) -> None:
        if not isinstance(payload, Mapping):
            return
        for id_field, name_field in _IDENTITY_FIELDS:
            if id_field not in payload and name_field not in payload:
                continue
            player_id = _payload_player_id(payload, id_field, actor_id, target_id)
            name = payload.get(name_field)
            if player_id is not None and isinstance(name, str) and name:
                self._name_to_ids.setdefault((game_id, name), set()).add(player_id)
        alive_names = payload.get("alive_player_names")
        if isinstance(alive_names, list):
            for name in alive_names:
                if isinstance(name, str) and name:
                    match = _USER_ID_RE.match(name)
                    if match:
                        self.alias_for_id(int(match.group(1)))
                    else:
                        self.alias_for_name(game_id, name)


def anonymize_context(context: LLMContext, aliases: PlayerAliasMap) -> LLMContext:
    """Return a copy of ``context`` with player identities replaced.

    The rendered prompt must be built from this copy — the aliases live in the
    structured fields (players, message authors, votes, deaths, revealed
    roles), while message bodies and model reasoning stay untouched.
    """
    game_id = context.game_id

    def remap(player_id: int) -> int:
        mapped = aliases.remap_id(player_id)
        if mapped is None:  # pragma: no cover — ids in context are never None
            raise ValueError(f"cannot anonymize player id {player_id}")
        return mapped

    def alias_name(name: str, player_id: int | None = None) -> str:
        return aliases.alias_for_name(game_id, name, player_id) or name

    players = [
        PlayerInfo(
            player_id=remap(player.player_id),
            display_name=alias_name(player.display_name, player.player_id),
            alive=player.alive,
            known_role_name_ru=player.known_role_name_ru,
            is_placeholder=player.is_placeholder,
        )
        for player in context.players
    ]
    recent_messages = [
        MessageInfo(
            ts=message.ts,
            author_name=alias_name(message.author_name),
            text=message.text,
        )
        for message in context.recent_messages
    ]
    recent_deaths = [
        DeathInfo(
            ts=death.ts,
            victim_name=alias_name(death.victim_name),
            cause=death.cause,
        )
        for death in context.recent_deaths
    ]
    recent_votes = [
        VoteInfo(
            ts=vote.ts,
            voter_name=alias_name(vote.voter_name),
            target_name=alias_name(vote.target_name) if vote.target_name else None,
        )
        for vote in context.recent_votes
    ]
    revealed_roles = [
        RevealedPlayer(
            player_id=remap(revealed.player_id),
            player_name=alias_name(revealed.player_name, revealed.player_id),
            role_name_ru=revealed.role_name_ru,
            is_winner=revealed.is_winner,
        )
        for revealed in context.revealed_roles
    ]
    previous_predictions = [
        PredictionSnapshot(
            player_id=remap(snapshot.player_id),
            suspected_role=snapshot.suspected_role,
            confidence=snapshot.confidence,
        )
        for snapshot in context.previous_predictions
    ]
    silent_ids = frozenset(remap(pid) for pid in context.silent_player_ids)
    return LLMContext(
        game_id=context.game_id,
        phase=context.phase,
        players=players,
        recent_messages=recent_messages,
        recent_deaths=recent_deaths,
        recent_votes=recent_votes,
        previous_predictions=previous_predictions,
        silent_player_ids=silent_ids,
        truncated_messages=context.truncated_messages,
        total_messages_seen=context.total_messages_seen,
        role_count=context.role_count,
        total_players=context.total_players,
        role_slots=context.role_slots,
        role_kinds=context.role_kinds,
        composition_matches_total=context.composition_matches_total,
        unknown_role_names=context.unknown_role_names,
        alive_list_names=[alias_name(name) for name in context.alive_list_names],
        placeholder_player_names=[
            alias_name(name) for name in context.placeholder_player_names
        ],
        revealed_roles=revealed_roles,
    )


def anonymize_payload(payload: dict[str, Any], aliases: PlayerAliasMap) -> dict[str, Any]:
    """Remap player ids and names inside an assistant JSON payload."""
    analysis = []
    for item in payload.get("analysis", []):
        if not isinstance(item, dict):
            analysis.append(item)
            continue
        mapped_id = aliases.remap_id(item.get("player_id"))
        name = aliases.alias_for_name(None, item.get("player_name"), item.get("player_id"))
        analysis.append(
            {
                **item,
                "player_id": mapped_id if mapped_id is not None else item.get("player_id"),
                "player_name": name if name else item.get("player_name"),
            }
        )
    return {**payload, "analysis": analysis}


def _actor_id(event: AnyEvent) -> int | None:
    """Return the actor id of an event, if the event kind carries one."""
    payload = event.payload
    for field in ("author_id", "victim_id", "voter_id", "player_id"):
        value = getattr(payload, field, None)
        if isinstance(value, int):
            return value
    return None


def _target_id(event: AnyEvent) -> int | None:
    """Return the target id of an event, if the event kind carries one."""
    payload = event.payload
    value = getattr(payload, "target_id", None)
    return value if isinstance(value, int) else None


def _payload_player_id(
    payload: Mapping[str, Any],
    id_field: str,
    actor_id: int | None,
    target_id: int | None,
) -> int | None:
    """Prefer the payload id, falling back to the event column."""
    value = payload.get(id_field)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    fallback = target_id if id_field == "target_id" else actor_id
    return fallback
