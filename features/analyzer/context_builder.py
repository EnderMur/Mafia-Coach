"""Build the LLM context for one analysis cycle and render its prompt messages.

The module is intentionally pure: no I/O, no async, no clients, no imports from
``infra/`` or ``app/``. Everything it needs arrives from :class:`GameState`,
:class:`RoleRegistry`, and an already configured Jinja environment.

Two responsibilities live here:

- :func:`build_context` — collapse the mutable :class:`GameState` into an
  immutable :class:`LLMContext` (players, last messages, votes, deaths, phase);
- :func:`to_llm_messages` — render ``system.jinja`` + ``analyze.jinja`` into the
  ``messages`` list consumed by the LLM client.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

import jinja2
from pydantic import BaseModel, ConfigDict, Field

from domain.events import AliveListEvent, DeathEvent, MessageEvent, VoteEvent
from domain.game_state import GameState
from domain.names import name_match_keys, names_match
from domain.phases import Phase
from domain.roles import RoleRegistry

DEFAULT_MAX_MESSAGES: Final[int] = 200
"""How many incoming messages are kept in the LLM context window."""

_ROLE_NAME_TRIM: Final[str] = " .,;:!?«»\"'()[]{}"

SYSTEM_TEMPLATE_NAME: Final[str] = "system.jinja"
ANALYZE_TEMPLATE_NAME: Final[str] = "analyze.jinja"

_UNRENDERED_MARKERS: Final[tuple[str, ...]] = ("{{", "{%")
"""Jinja markers that must not survive ``Template.render``."""


class PlayerInfo(BaseModel):
    """One player in the LLM context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: int
    display_name: str = Field(min_length=1)
    alive: bool
    known_role_name_ru: str | None = None
    #: ``True`` for deterministic placeholder ids (player never wrote in chat).
    is_placeholder: bool = False


class MessageInfo(BaseModel):
    """One message in the recent window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: datetime
    author_name: str = Field(min_length=1)
    text: str


class VoteInfo(BaseModel):
    """A recent vote, including abstentions and partially known targets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: datetime
    voter_name: str = Field(min_length=1)
    target_name: str | None


class DeathInfo(BaseModel):
    """Recent death in the context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: datetime
    victim_name: str = Field(min_length=1)
    cause: str


class PhaseInfo(BaseModel):
    """Current phase state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    current_phase: Phase
    current_day: int = Field(ge=0)


class RevealedPlayer(BaseModel):
    """Игрок с подтверждённой (раскрытой ботом) ролью.

    ``ground truth`` для модели: такую роль нельзя пересматривать, поэтому она
    вынесена в отдельный блок промпта, а не только в строку ``players``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: int
    player_name: str = Field(min_length=1)
    role_name_ru: str = Field(min_length=1)
    is_winner: bool | None = None


class PredictionSnapshot(BaseModel):
    """Previous prediction (для контекста модели)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: int
    suspected_role: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class LLMContext(BaseModel):
    """Everything the LLM needs to see for one analysis cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    game_id: int
    phase: PhaseInfo
    players: list[PlayerInfo]
    recent_messages: list[MessageInfo]
    recent_deaths: list[DeathInfo]
    recent_votes: list[VoteInfo] = Field(default_factory=list)
    previous_predictions: list[PredictionSnapshot] = Field(default_factory=list)
    #: Alive player ids with no messages and no votes in the game so far.
    #: Only they may receive ``suspected_role = null`` from the model.
    silent_player_ids: frozenset[int] = Field(default_factory=frozenset)
    truncated_messages: int = Field(ge=0)
    total_messages_seen: int = Field(ge=0)
    #: Bot-reported role composition (canonical name_ru → count), or None.
    role_count: dict[str, int] | None = None
    #: Bot-reported total alive headcount, or None.
    total_players: int | None = None
    #: Sum of ``role_count`` values: exact number of role slots to distribute.
    role_slots: int = 0
    #: Number of distinct roles in the bot's composition.
    role_kinds: int = 0
    #: ``role_slots == total_players``, or ``None`` when either value is absent.
    composition_matches_total: bool | None = None
    #: Role names from the bot that do not resolve to the registry.
    unknown_role_names: list[str] = Field(default_factory=list)
    #: Alive nicknames exactly as the last bot snapshot listed them.
    alive_list_names: list[str] = Field(default_factory=list)
    #: Bot-listed nicknames backed by synthetic (negative) player ids.
    placeholder_player_names: list[str] = Field(default_factory=list)
    #: Roles confirmed by the bot (mid-game reveal or final report).
    revealed_roles: list[RevealedPlayer] = Field(default_factory=list)


def _canonical_role_count(
    role_count: dict[str, int] | None,
    registry: RoleRegistry,
) -> tuple[dict[str, int] | None, list[str]]:
    """Map bot-spelled role names onto canonical ``name_ru`` values.

    Older events in SQLite may still contain the raw bot spelling (e.g.
    ``"Мирный Житель"``). Canonicalising here — as well as in the parser —
    guarantees the prompt never shows a role name that the response validator
    would reject. Duplicate keys after canonicalisation are summed.

    Args:
        role_count: composition as stored in ``GameState``.
        registry: configured role catalogue.

    Returns:
        ``(canonical_counts, unknown_names)``. ``canonical_counts`` is ``None``
        when the input was ``None`` or contained no usable names.
    """
    if role_count is None:
        return None, []
    canonical: dict[str, int] = {}
    unknown: list[str] = []
    for raw_name, count in role_count.items():
        candidate = " ".join(raw_name.split()).strip(_ROLE_NAME_TRIM)
        if not candidate:
            continue
        role = registry.resolve_alias(candidate)
        if role is None:
            key = candidate
            if key not in unknown:
                unknown.append(key)
        else:
            key = role.name_ru
        canonical[key] = canonical.get(key, 0) + count
    return canonical or None, unknown


def build_context(
    *,
    game_state: GameState,
    registry: RoleRegistry,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    previous_predictions: list[PredictionSnapshot] | None = None,
) -> LLMContext:
    """Build an :class:`LLMContext` from the current :class:`GameState`.

    - ``recent_messages`` — the last ``max_messages`` :class:`MessageEvent`
      objects, chronological (oldest → newest);
    - ``recent_deaths`` — all :class:`DeathEvent` objects (usually fewer than 15
      per game);
    - ``recent_votes`` — the last 20 :class:`VoteEvent` objects in append order;
    - ``players`` — alive and dead players (real ids and whitelisted
      placeholders), with ``known_role_name_ru`` filled in when the role was
      publicly revealed and resolved by the registry;
    - ``previous_predictions`` — carried over from the previous analysis cycle;
    - ``truncated_messages`` — ``max(0, total_messages_seen - max_messages)``.

    Only :class:`MessageEvent` reaches ``recent_messages``. Phase changes, votes,
    deaths, and role reveals are *not* duplicated there: they are reflected in
    ``phase``, ``recent_votes``, ``recent_deaths``, and players' ``known_role``.

    The message text is used as stored by :class:`GameState`, which keeps the
    already normalized parser output; no extra cleanup is applied here.

    Args:
        game_state: aggregate to read the snapshot from.
        registry: role registry (reserved for role-aware enrichment).
        max_messages: size of the message window; ``0`` keeps no messages.
        previous_predictions: predictions from the previous analysis cycle.

    Returns:
        An immutable :class:`LLMContext`.

    Raises:
        ValueError: if ``max_messages`` is negative.
    """
    if max_messages < 0:
        raise ValueError("max_messages must be non-negative")

    message_events = [event for event in game_state.events() if isinstance(event, MessageEvent)]
    total_messages = len(message_events)
    window = message_events[-max_messages:] if max_messages > 0 else []
    recent_messages = [
        MessageInfo(
            ts=event.ts,
            author_name=event.payload.author_name,
            text=event.payload.text,
        )
        for event in window
    ]

    recent_deaths = [
        DeathInfo(
            ts=event.ts,
            victim_name=event.payload.victim_name,
            cause=event.payload.cause,
        )
        for event in game_state.events_by_kind("death")
        if isinstance(event, DeathEvent)
    ]

    vote_events = [
        event for event in game_state.events_by_kind("vote") if isinstance(event, VoteEvent)
    ]
    recent_votes = [
        VoteInfo(
            ts=event.ts,
            voter_name=event.payload.voter_name,
            target_name=event.payload.target_name,
        )
        for event in vote_events[-20:]
    ]

    # Players who never wrote in chat and never voted have no behavioural
    # signal at all: the model may only assign them ``suspected_role = null``.
    message_authors = {event.payload.author_id for event in message_events}
    vote_voters = {event.payload.voter_id for event in vote_events}
    silent_player_ids = frozenset(
        player.player_id
        for player in game_state.players()
        if player.alive
        and player.player_id not in message_authors
        and player.player_id not in vote_voters
    )

    # ALL players of the snapshot reach the LLM, including deterministic
    # placeholders for players who never wrote in chat. Otherwise a 16-player
    # alive list can produce only e.g. 5 addressable ids and a full role
    # distribution is impossible.
    players = [
        PlayerInfo(
            player_id=player.player_id,
            display_name=player.display_name,
            alive=player.alive,
            known_role_name_ru=(
                player.known_role.name_ru
                if player.revealed and player.known_role is not None
                else None
            ),
            is_placeholder=player.player_id <= 0,
        )
        for player in game_state.players()
    ]

    # Roles confirmed by the bot are ground truth: they are rendered as their
    # own block before the player list, so the model does not "re-discover"
    # them as "Мирный житель".
    revealed: list[RevealedPlayer] = []
    for p in game_state.players():
        confirmed_role = p.final_role or p.known_role
        if not p.revealed or confirmed_role is None:
            continue
        revealed.append(
            RevealedPlayer(
                player_id=p.player_id,
                player_name=p.display_name,
                role_name_ru=confirmed_role.name_ru,
                is_winner=p.is_winner,
            )
        )

    alive_events = [
        event
        for event in game_state.events_by_kind("alive_list")
        if isinstance(event, AliveListEvent)
    ]
    alive_list_names = list(alive_events[-1].payload.alive_player_names) if alive_events else []
    alive_keys: set[str] = set()
    for name in alive_list_names:
        alive_keys.update(name_match_keys(name))
    placeholder_player_names = [
        info.display_name
        for info in players
        if info.is_placeholder
        and (
            not alive_list_names
            or any(names_match(info.display_name, listed) for listed in alive_list_names)
            or bool(name_match_keys(info.display_name) & alive_keys)
        )
    ]

    canonical_roles, unknown_role_names = _canonical_role_count(game_state.role_count, registry)
    role_slots = sum(canonical_roles.values()) if canonical_roles else 0
    role_kinds = len(canonical_roles) if canonical_roles else 0
    total_players = game_state.total_players
    composition_matches_total: bool | None = None
    if canonical_roles is not None and total_players is not None:
        composition_matches_total = role_slots == total_players

    return LLMContext(
        game_id=_game_id(game_state),
        phase=PhaseInfo(
            current_phase=game_state.current_phase(),
            current_day=game_state.current_day(),
        ),
        players=players,
        recent_messages=recent_messages,
        recent_deaths=recent_deaths,
        recent_votes=recent_votes,
        previous_predictions=list(previous_predictions or []),
        silent_player_ids=silent_player_ids,
        truncated_messages=max(0, total_messages - max_messages),
        total_messages_seen=total_messages,
        role_count=canonical_roles,
        total_players=total_players,
        role_slots=role_slots,
        role_kinds=role_kinds,
        composition_matches_total=composition_matches_total,
        unknown_role_names=unknown_role_names,
        alive_list_names=alive_list_names,
        placeholder_player_names=placeholder_player_names,
        revealed_roles=revealed,
    )


def to_llm_messages(
    context: LLMContext,
    *,
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    response_schema_json: str | None = None,
    club_notes_summary: str = "",
    club_patterns: list[dict[str, object]] | None = None,
) -> list[dict[str, str]]:
    """Render ``system.jinja`` and ``analyze.jinja`` for one analysis cycle.

    ``system.jinja`` receives the sorted Russian role names of the registry and
    the JSON Schema of the expected response. ``analyze.jinja`` receives the
    context fields, with ``recent_messages`` ordered chronologically and
    ``known_role`` expressed as ``name_ru``.

    Args:
        context: snapshot produced by :func:`build_context`.
        registry: role registry used for the role catalogue and aliases.
        jinja_env: environment loaded from ``infra/llm/prompts`` with
            ``autoescape=False``.
        response_schema_json: JSON Schema of the response. Without an explicit
            value the ``response_schema_json`` global of ``jinja_env`` is used,
            and :data:`DEFAULT_RESPONSE_SCHEMA_JSON` as the last resort.
        club_notes_summary: optional textual summary of heuristic hints.
        club_patterns: optional structured club patterns (list of dicts with
            keys id, trigger, conclusion_role, confidence_tier,
            requires_confirmation, description).

    Returns:
        ``[{"role": "system", ...}, {"role": "user", ...}]``.

    Raises:
        ValueError: if a rendered message still contains Jinja markers.
    """
    role_names_ru = sorted(role.name_ru for role in registry.all_roles())
    schema_json = _response_schema_json(jinja_env, response_schema_json)

    system_content = jinja_env.get_template(SYSTEM_TEMPLATE_NAME).render(
        role_names_ru=role_names_ru,
        response_schema_json=schema_json,
        previous_predictions=[
            _prediction_payload(prediction) for prediction in context.previous_predictions
        ],
    )
    analyze_content = jinja_env.get_template(ANALYZE_TEMPLATE_NAME).render(
        game_id=context.game_id,
        current_phase=context.phase.current_phase.value,
        current_day=context.phase.current_day,
        players=[_player_payload(player) for player in context.players],
        recent_messages=[
            _message_payload(message)
            for message in sorted(context.recent_messages, key=lambda item: item.ts)
        ],
        previous_predictions=[
            _prediction_payload(prediction) for prediction in context.previous_predictions
        ],
        role_count=context.role_count,
        total_players=context.total_players,
        role_slots=context.role_slots,
        role_kinds=context.role_kinds,
        composition_matches_total=context.composition_matches_total,
        unknown_role_names=context.unknown_role_names,
        alive_list_names=context.alive_list_names,
        placeholder_player_names=context.placeholder_player_names,
        revealed_roles=[_revealed_payload(player) for player in context.revealed_roles],
        club_notes_summary=club_notes_summary,
        recent_votes=context.recent_votes,
        role_notes=_render_role_notes(registry, context.role_count),
        club_patterns=club_patterns,
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": analyze_content},
    ]
    _ensure_fully_rendered(messages)
    return messages


def _render_role_notes(registry: RoleRegistry, role_count: dict[str, int] | None = None) -> str:
    """Render nonempty role notes for the club heuristics prompt.

    When the bot already reported the exact composition, notes for roles that
    are not in the game only dilute a small local model's attention. Keep the
    full catalogue when composition is unknown; otherwise render only present
    roles.
    """
    lines = []
    wanted = set(role_count) if role_count else None
    for role in registry.all_roles():
        if wanted is not None and role.name_ru not in wanted:
            continue
        notes = (role.notes or "").strip()
        if notes:
            lines.append(f"- {role.name_ru}: {notes}")
    return "\n".join(lines)


def _game_id(game_state: GameState) -> int:
    """Return the game identifier through the public game-state API."""
    return game_state.game_id


def _response_schema_json(jinja_env: jinja2.Environment, override: str | None) -> str:
    """Resolve the JSON Schema handed to the system prompt."""
    if override is not None:
        return override
    injected = jinja_env.globals.get("response_schema_json")
    if isinstance(injected, str) and injected:
        return injected
    return DEFAULT_RESPONSE_SCHEMA_JSON


def _player_payload(player: PlayerInfo) -> dict[str, object]:
    """Flatten a player for the ``analyze.jinja`` template."""
    return {
        "player_id": player.player_id,
        "display_name": player.display_name,
        "alive": player.alive,
        "known_role": player.known_role_name_ru,
        "is_placeholder": player.is_placeholder,
    }


def _revealed_payload(player: RevealedPlayer) -> dict[str, object]:
    """Flatten a confirmed role for the ``analyze.jinja`` template."""
    return {
        "player_id": player.player_id,
        "player_name": player.player_name,
        "role_name_ru": player.role_name_ru,
        "is_winner": player.is_winner,
    }


def _message_payload(message: MessageInfo) -> dict[str, object]:
    """Flatten a message for the ``analyze.jinja`` template."""
    return {
        "ts": message.ts.isoformat(timespec="seconds"),
        "author_name": message.author_name,
        "text": message.text,
    }


def _prediction_payload(prediction: PredictionSnapshot) -> dict[str, object]:
    """Flatten a previous prediction for the ``analyze.jinja`` template."""
    return {
        "player_id": prediction.player_id,
        "suspected_role": prediction.suspected_role,
        "confidence": prediction.confidence,
    }


def _ensure_fully_rendered(messages: list[dict[str, str]]) -> None:
    """Reject prompts that still contain unprocessed Jinja markers.

    A leftover marker means the template or its data was not rendered, and the
    model would receive raw template syntax instead of the game context.
    """
    for message in messages:
        content = message["content"]
        for marker in _UNRENDERED_MARKERS:
            if marker in content:
                raise ValueError(
                    f"prompt rendering left the Jinja marker {marker!r} "
                    f"in the {message['role']} message"
                )


# JSON Schema of AnalysisResponse (v1), embedded for the system prompt.
# Generated from AnalysisResponse.model_json_schema() in infra/llm/schemas.py:
# features/analyzer must not import infra, so a caller may pass the live schema
# to to_llm_messages instead. The value is pinned to the contract model by
# tests/unit/features/analyzer/test_context_builder.py, which fails as soon as
# the two drift apart.
# JSON Schema of AnalysisResponse (v1), embedded for the system prompt.
# Generated from AnalysisResponse.model_json_schema() in infra/llm/schemas.py:
# features/analyzer must not import infra, so a caller may pass the live schema
# to to_llm_messages instead. The value is pinned to the contract model by
# tests/unit/features/analyzer/test_context_builder.py, which fails as soon as
# the two drift apart.
DEFAULT_RESPONSE_SCHEMA_JSON: Final[str] = """
{
  "$defs": {
    "AlternativePrediction": {
      "additionalProperties": false,
      "description": "An alternative role hypothesis with a confidence score.",
      "properties": {
        "role": {
          "minLength": 1,
          "title": "Role",
          "type": "string"
        },
        "confidence": {
          "maximum": 1.0,
          "minimum": 0.0,
          "title": "Confidence",
          "type": "number"
        }
      },
      "required": [
        "role",
        "confidence"
      ],
      "title": "AlternativePrediction",
      "type": "object"
    },
    "PlayerPrediction": {
      "additionalProperties": false,
      "description": "Role hypotheses and reasons for one player.",
      "properties": {
        "player_id": {
          "title": "Player Id",
          "type": "integer"
        },
        "player_name": {
          "minLength": 1,
          "title": "Player Name",
          "type": "string"
        },
        "suspected_role": {
          "anyOf": [
            {
              "minLength": 1,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Suspected Role"
        },
        "confidence": {
          "maximum": 1.0,
          "minimum": 0.0,
          "title": "Confidence",
          "type": "number"
        },
        "reason": {
          "minLength": 1,
          "title": "Reason",
          "type": "string"
        },
        "alternative": {
          "anyOf": [
            {
              "$ref": "#/$defs/AlternativePrediction"
            },
            {
              "type": "null"
            }
          ],
          "default": null
        },
        "reasons": {
          "items": {
            "type": "string"
          },
          "maxItems": 5,
          "minItems": 1,
          "title": "Reasons",
          "type": "array"
        },
        "pattern_ids": {
          "items": {
            "type": "string"
          },
          "title": "Pattern Ids",
          "type": "array"
        },
        "pattern_descriptions": {
          "items": {
            "type": "string"
          },
          "title": "Pattern Descriptions",
          "type": "array"
        }
      },
      "required": [
        "player_id",
        "player_name",
        "confidence",
        "reason",
        "reasons"
      ],
      "title": "PlayerPrediction",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "description": "The structured final response returned by the language model.",
  "properties": {
    "analysis": {
      "items": {
        "$ref": "#/$defs/PlayerPrediction"
      },
      "title": "Analysis",
      "type": "array"
    },
    "game_read": {
      "minLength": 1,
      "title": "Game Read",
      "type": "string"
    },
    "reasoning": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Reasoning"
    }
  },
  "required": [
    "analysis",
    "game_read"
  ],
  "title": "AnalysisResponse",
  "type": "object"
}
"""
