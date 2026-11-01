"""Build the LLM context for one analysis cycle and render its prompt messages.

The module is intentionally pure: no I/O, no async, no clients, no imports from
``infra/`` or ``app/``. Everything it needs arrives from :class:`GameState`,
:class:`RoleRegistry`, and an already configured Jinja environment.

Two responsibilities live here:

- :func:`build_context` — collapse the mutable :class:`GameState` into an
  immutable :class:`LLMContext` (players, last messages, deaths, phase);
- :func:`to_llm_messages` — render ``system.jinja`` + ``analyze.jinja`` into the
  ``messages`` list consumed by the LLM client.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

import jinja2
from pydantic import BaseModel, ConfigDict, Field

from domain.events import DeathEvent, MessageEvent
from domain.game_state import GameState
from domain.phases import Phase
from domain.roles import RoleRegistry

DEFAULT_MAX_MESSAGES: Final[int] = 200
"""How many incoming messages are kept in the LLM context window."""

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


class MessageInfo(BaseModel):
    """One message in the recent window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: datetime
    author_name: str = Field(min_length=1)
    text: str


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
    previous_predictions: list[PredictionSnapshot] = []
    truncated_messages: int = Field(ge=0)
    total_messages_seen: int = Field(ge=0)


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
    - ``players`` — alive and dead players, with ``known_role_name_ru`` filled in
      when the role was publicly revealed and resolved by the registry;
    - ``previous_predictions`` — carried over from the previous analysis cycle;
    - ``truncated_messages`` — ``max(0, total_messages_seen - max_messages)``.

    Only :class:`MessageEvent` reaches ``recent_messages``. Phase changes, votes,
    deaths, and role reveals are *not* duplicated there: they are already
    reflected in ``phase``, ``recent_deaths``, and the players' ``known_role``.

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

    players = [
        PlayerInfo(
            player_id=player.player_id,
            display_name=player.display_name,
            alive=player.alive,
            known_role_name_ru=(
                player.known_role.name_ru if player.known_role is not None else None
            ),
        )
        for player in game_state.players()
    ]

    return LLMContext(
        game_id=_game_id(game_state),
        phase=PhaseInfo(
            current_phase=game_state.current_phase(),
            current_day=game_state.current_day(),
        ),
        players=players,
        recent_messages=recent_messages,
        recent_deaths=recent_deaths,
        previous_predictions=list(previous_predictions or []),
        truncated_messages=max(0, total_messages - max_messages),
        total_messages_seen=total_messages,
    )


def to_llm_messages(
    context: LLMContext,
    *,
    registry: RoleRegistry,
    jinja_env: jinja2.Environment,
    response_schema_json: str | None = None,
    club_notes_summary: str = "",
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
        club_notes_summary=club_notes_summary,
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": analyze_content},
    ]
    _ensure_fully_rendered(messages)
    return messages


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
          "minLength": 1,
          "title": "Suspected Role",
          "type": "string"
        },
        "confidence": {
          "maximum": 1.0,
          "minimum": 0.0,
          "title": "Confidence",
          "type": "number"
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
        }
      },
      "required": [
        "player_id",
        "player_name",
        "suspected_role",
        "confidence",
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
      "minLength": 1,
      "title": "Reasoning",
      "type": "string"
    }
  },
  "required": [
    "analysis",
    "game_read",
    "reasoning"
  ],
  "title": "AnalysisResponse",
  "type": "object"
}
"""
