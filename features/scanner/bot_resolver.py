"""Convert parsed bot messages into domain events.

Pure module: no I/O, no ``async``, no Telegram/``app``/``infra`` imports.
Everything the resolver needs (name → id mapping, current phase) is passed
in by the caller — the scanner (step 9b) — so this module stays a
deterministic function of its arguments and is trivially unit-testable.

Mapping (``BotMessageKind`` → events)
-------------------------------------
``PHASE_CHANGE``   → ``[PhaseChangeEvent]``   (only for a real transition)
``DEATH``          → ``[DeathEvent]``         (victim must resolve to an id)
``ROLE_REVEAL``    → ``[RoleRevealEvent]``    (role id and name must resolve)
``VOTE``           → ``[VoteEvent, ...]``     (one event per vote pair)
``LAST_WORDS``     → ``[MessageEvent]``       (author is the dying player)
``ALIVE_LIST``     → ``[AliveListEvent]``     (bot snapshot, names unresolved)
everything else    → ``[]``                   (informational only)

Unresolvable data is never fatal: the event is skipped with a WARNING on
the ``scanner`` log channel, because a single unknown nickname must not
stop the whole pipeline.

Player names are normalized with :func:`_strip_markdown` before lookup: real
game bots mention players as Telegram markdown links
(``[Yulios](tg://user?id=1)``) or with ``**bold**``/``*italic*`` markup, and
those decorated strings never match the plain nicknames the scanner collected
from Telegram messages.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Final, Literal, TypeAlias

from loguru import logger

from domain.events import (
    AliveListEvent,
    AliveListPayload,
    AnyEvent,
    DeathEvent,
    DeathPayload,
    MessageEvent,
    MessagePayload,
    PhaseChangeEvent,
    PhaseChangePayload,
    RoleRevealEvent,
    RoleRevealPayload,
    VoteEvent,
    VotePayload,
)
from domain.exceptions import UnknownRoleError
from domain.phases import Phase
from domain.roles import Role, RoleRegistry
from features.parser.bot_messages import BotMessageKind, BotMessageParse

_LOGGER_CHANNEL: Final[str] = "scanner"

#: ``DeathPayload.cause`` is a closed literal, so the mapping below is total:
#: any cause the parser cannot produce degrades to ``unknown`` instead of
#: raising a validation error inside the message loop.
_DeathCause: TypeAlias = Literal["night_kill", "lynch", "unknown"]

#: Fallback cause when the parser did not determine how the player died.
_UNKNOWN_CAUSE: Final[_DeathCause] = "unknown"

#: Raw ``BotMessageParse.death_cause`` values accepted by the domain model.
_DEATH_CAUSES: Final[dict[str | None, _DeathCause]] = {
    None: "unknown",
    "": "unknown",
    "night_kill": "night_kill",
    "lynch": "lynch",
    "unknown": "unknown",
}

#: Placeholder text for a last-words announcement without any words.
_NO_LAST_WORDS: Final[str] = "(no words)"

#: Telegram mention of a player: ``[Yulios](tg://user?id=1)`` → ``Yulios``.
_MARKDOWN_LINK_RE: Final[re.Pattern[str]] = re.compile(r"\[?([^\[\]]+)\]\(tg://user\?id=\d+\)?")

#: MarkdownV2 bold: ``**Вася**`` → ``Вася``.
_MD_BOLD_RE: Final[re.Pattern[str]] = re.compile(r"\*\*([^*]+)\*\*")

#: Legacy italic: ``*Петя*`` → ``Петя`` (without eating bold markers).
_MD_ITALIC_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


def _strip_markdown(text: str) -> str:
    """Extract the plain player name from Telegram markdown decoration.

    Game bots write names as mention links and emphasize them with markup, so
    the raw text almost never equals the nickname the scanner stored in
    ``name_to_id``. Only decorations are removed — the name itself is kept
    verbatim, including inner spaces and case.

    Args:
        text: player name exactly as the bot message spelled it.

    Returns:
        The bare name with surrounding whitespace removed. Text without any
        decoration is returned unchanged.
    """
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    return text.strip()


def resolve_bot_message(
    *,
    parse: BotMessageParse,
    name_to_id: Mapping[str, int],
    registry: RoleRegistry,
    game_id: int,
    ts: datetime,
    current_phase: Phase,
) -> list[AnyEvent]:
    """Convert a parsed bot message into zero or more domain events.

    Args:
        parse: result of :func:`features.parser.bot_messages.parse_bot_message`.
        name_to_id: player name **exactly as the bot writes it in game** →
            ``player_id``.  The scanner (step 9b) builds it from Telegram
            messages; this resolver never touches Telegram itself.
        registry: role registry used to turn ``role_id`` into ``name_ru``.
        game_id: id of the game the events belong to.
        ts: timezone-aware timestamp of the source Telegram message.
        current_phase: current phase of ``GameState``.  It becomes
            ``PhaseChangePayload.from_phase`` because ``GameState.apply``
            validates ``from_phase == current_phase`` and raises
            ``ConfigError`` otherwise.

    Returns:
        ``list[AnyEvent]`` — 0 events (informational or unresolvable
        message), 1 event, or N events (multi-vote message).

    Notes:
        Informational kinds (``REGISTRATION``, ``ROLE_ACTION``, ``SYSTEM``,
        ``GIFT``, ``PROTECTION_USED``, ``VOTE_ABSTAIN``, ``UNKNOWN``) produce
        no events. ``ALIVE_LIST`` produces an ``AliveListEvent`` unless the
        parser found no player names.
    """
    kind = parse.kind
    if kind is BotMessageKind.PHASE_CHANGE:
        return _resolve_phase_change(
            parse=parse, game_id=game_id, ts=ts, current_phase=current_phase
        )
    if kind is BotMessageKind.DEATH:
        return _resolve_death(parse=parse, name_to_id=name_to_id, game_id=game_id, ts=ts)
    if kind is BotMessageKind.ROLE_REVEAL:
        return _resolve_role_reveal(
            parse=parse, name_to_id=name_to_id, registry=registry, game_id=game_id, ts=ts
        )
    if kind is BotMessageKind.VOTE:
        return _resolve_votes(parse=parse, name_to_id=name_to_id, game_id=game_id, ts=ts)
    if kind is BotMessageKind.LAST_WORDS:
        return _resolve_last_words(parse=parse, name_to_id=name_to_id, game_id=game_id, ts=ts)
    if kind is BotMessageKind.ALIVE_LIST:
        if not parse.alive_player_names:
            _warn("alive_list without player names, skipped")
            return []
        payload = AliveListPayload(
            alive_player_names=list(parse.alive_player_names),
            role_count=dict(parse.role_count) if parse.role_count else None,
            total_players=parse.total_players,
        )
        return [AliveListEvent(ts=ts, game_id=game_id, payload=payload)]
    # REGISTRATION, ROLE_ACTION, SYSTEM, GIFT, PROTECTION_USED,
    # VOTE_ABSTAIN, UNKNOWN — informational, no domain event.
    return []


# ---------------------------------------------------------------------------
# Per-kind resolvers (pure helpers)
# ---------------------------------------------------------------------------


def _resolve_phase_change(
    *,
    parse: BotMessageParse,
    game_id: int,
    ts: datetime,
    current_phase: Phase,
) -> list[AnyEvent]:
    """Build a phase transition event, skipping no-op and unknown phases."""
    if parse.to_phase is None:
        _warn("phase_change without to_phase, skipped")
        return []
    if parse.to_phase == current_phase:
        _warn("phase_change to current phase {}, skipped", current_phase)
        return []
    payload = PhaseChangePayload(
        from_phase=current_phase,
        to_phase=parse.to_phase,
        day_number=parse.day_number or 0,
    )
    return [PhaseChangeEvent(ts=ts, game_id=game_id, payload=payload)]


def _resolve_death(
    *,
    parse: BotMessageParse,
    name_to_id: Mapping[str, int],
    game_id: int,
    ts: datetime,
) -> list[AnyEvent]:
    """Build a death event when the victim name resolves to a player id."""
    if not parse.player_name:
        _warn("death without player_name, skipped")
        return []
    victim_name = _strip_markdown(parse.player_name)
    victim_id = name_to_id.get(victim_name)
    if victim_id is None:
        _warn("death: unresolved victim name {!r}, skipped", victim_name)
        return []
    cause = _DEATH_CAUSES.get(parse.death_cause)
    if cause is None:
        _warn("death: unrecognized cause {!r}, recorded as unknown", parse.death_cause)
        cause = _UNKNOWN_CAUSE
    return [
        DeathEvent(
            ts=ts,
            game_id=game_id,
            payload=DeathPayload(
                victim_id=victim_id,
                victim_name=victim_name,
                cause=cause,
            ),
        )
    ]


def _resolve_role_reveal(
    *,
    parse: BotMessageParse,
    name_to_id: Mapping[str, int],
    registry: RoleRegistry,
    game_id: int,
    ts: datetime,
) -> list[AnyEvent]:
    """Build a role reveal event when both the role and the player resolve."""
    if not parse.role_id or not parse.player_name:
        _warn(
            "role_reveal without role_id or player_name (confidence {}), skipped",
            parse.confidence,
        )
        return []
    try:
        role: Role = registry.get(parse.role_id)
    except UnknownRoleError:
        _warn("role_reveal: unknown role id {!r}, skipped", parse.role_id)
        return []
    player_name = _strip_markdown(parse.player_name)
    player_id = name_to_id.get(player_name)
    if player_id is None:
        _warn("role_reveal: unresolved player name {!r}, skipped", player_name)
        return []
    return [
        RoleRevealEvent(
            ts=ts,
            game_id=game_id,
            payload=RoleRevealPayload(
                player_id=player_id,
                player_name=player_name,
                role_name_ru=role.name_ru,
            ),
        )
    ]


def _resolve_votes(
    *,
    parse: BotMessageParse,
    name_to_id: Mapping[str, int],
    game_id: int,
    ts: datetime,
) -> list[AnyEvent]:
    """Build one vote event per pair; unresolved voters are skipped."""
    votes: list[tuple[str, str | None]] = list(parse.votes) if parse.votes is not None else []
    if not votes and parse.voter_name is not None:
        # Fallback for parsers/messages that carry a single vote pair only.
        # ``parse.target_name`` is normalized by the loop below as well.
        votes = [(parse.voter_name, parse.target_name)]

    events: list[AnyEvent] = []
    for raw_voter, raw_target in votes:
        voter = _strip_markdown(raw_voter)
        voter_id = name_to_id.get(voter)
        if voter_id is None:
            _warn("vote: unresolved voter name {!r}, pair skipped", voter)
            continue
        target = _strip_markdown(raw_target) if raw_target else None
        target_id = name_to_id.get(target) if target else None
        if target and target_id is None:
            _warn("vote: unresolved target name {!r}", target)
        events.append(
            VoteEvent(
                ts=ts,
                game_id=game_id,
                payload=VotePayload(
                    voter_id=voter_id,
                    voter_name=voter,
                    target_id=target_id,
                    target_name=target,
                ),
            )
        )
    return events


def _resolve_last_words(
    *,
    parse: BotMessageParse,
    name_to_id: Mapping[str, int],
    game_id: int,
    ts: datetime,
) -> list[AnyEvent]:
    """Turn a last-words announcement into a message from the dying player."""
    if not parse.player_name:
        _warn("last_words without player_name, skipped")
        return []
    player_name = _strip_markdown(parse.player_name)
    player_id = name_to_id.get(player_name)
    if player_id is None:
        _warn("last_words: unresolved player name {!r}, skipped", player_name)
        return []
    text = (parse.last_words or "").strip() or _NO_LAST_WORDS
    return [
        MessageEvent(
            ts=ts,
            game_id=game_id,
            payload=MessagePayload(
                author_id=player_id,
                author_name=player_name,
                text=text,
            ),
        )
    ]


def _warn(message: str, *args: object) -> None:
    """Log a skip reason on the ``scanner`` channel."""
    logger.bind(channel=_LOGGER_CHANNEL).warning(message, *args)
