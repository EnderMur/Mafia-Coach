"""Semantic parser for game-bot messages.

Pure module: no I/O, no ``async``, no imports from ``infra``/``app``/``telethon``.
All functions are synchronous and side-effect free.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict

from domain.phases import Phase
from domain.roles import Role, RoleRegistry


# ---------------------------------------------------------------------------
# Enums and models
# ---------------------------------------------------------------------------


class BotMessageKind(str, Enum):
    """Semantic kind of a parsed game-bot message."""

    REGISTRATION = "registration"
    PHASE_CHANGE = "phase_change"
    DEATH = "death"
    ROLE_REVEAL = "role_reveal"
    VOTE = "vote"
    ALIVE_LIST = "alive_list"
    UNKNOWN = "unknown"


class BotMessageParse(BaseModel):
    """Result of parsing one bot message.

    Names are STRINGS, not ids — resolution to ``int`` ids is the job
    of the scanner (step 9) via ``GameState``.  Here we only map
    text → meaning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: BotMessageKind
    confidence: float = 1.0

    # role_reveal
    role_id: str | None = None
    player_name: str | None = None

    # death
    death_cause: str | None = None

    # phase_change
    to_phase: Phase | None = None
    day_number: int | None = None

    # vote
    voter_name: str | None = None
    target_name: str | None = None

    raw_text: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TODO(step-12): migrate to roles.yaml field `bot_emoji`.
# Currently hardcoded here for simplicity; authoritative source
# is rules/Mafia_rules.txt headers.
_EMOJI_TO_ROLE_ID: Final[dict[str, str]] = {
    "🕵🏼": "commissioner",
    "👮🏻\u200d♂️": "sergeant",
    "👨🏼\u200d⚕️": "doctor",
    "💃": "mistress",
    "🧙🏻": "bum",
    "👨🏼": "civilian",
    "🤞🏻": "lucky",
    "💣": "kamikaze",
    "🤦🏼": "suicide",
    "🎖": "mayor",
    "🤵🏻": "don",
    "🤵🏼": "mafia",
    "👨🏼\u200d💼": "lawyer",
    "🕴️": "killer",
    "👩🏼\u200d💻": "journalist",
    "🔪": "maniac",
    "🧟": "arsonist",
    "🤹🏻": "swindler",
    "🤓": "snitch",
    "🧙": "mage",
    "🐺": "werewolf",
}

# Suffixes for Russian instrumental case (творительный падеж)
# used to normalise role names in "X был Мафией" patterns.
_ROLE_INSTRUMENTAL_SUFFIXES: Final[tuple[str, ...]] = (
    "ией",
    "ей",
    "ой",
    "ом",
    "ем",
    "ым",
    "им",
    "ам",
    "ям",
    "ью",
)


# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

_RE_REGISTRATION: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:регистрация\s+(?:открыта|закрыта|заверш)"
    r"|записал[ои]сь?\s+\d"
    r"|записал(?:ся|ось)\b"
    r"|регистраци[яи])",
)

_RE_NIGHT_START: Final[re.Pattern[str]] = re.compile(
    r"(?:🌙|наступает\s+ночь|город\s+засыпает|ночь\s+наступает|^ночь\s*\d)",
    re.IGNORECASE,
)
_RE_DAY_START: Final[re.Pattern[str]] = re.compile(
    r"(?:🌝|наступает\s+день|город\s+прос|утро|день\s+наступает|^день\s*\d)",
    re.IGNORECASE,
)
_RE_VOTE_PHASE: Final[re.Pattern[str]] = re.compile(
    r"(?:пришло\s+время\s+голосован|время\s+голосован|начинается\s+голосован)",
    re.IGNORECASE,
)
_RE_DAY_NUMBER: Final[re.Pattern[str]] = re.compile(
    r"(?:день|ночь)\s*(?:#?\s*)(\d+)",
    re.IGNORECASE,
)

_RE_ALIVE_LIST: Final[re.Pattern[str]] = re.compile(
    r"^[\s\n]*живы[её]?\s*[:\-]?",
    re.IGNORECASE,
)

_RE_DEATH_NIGHT: Final[re.Pattern[str]] = re.compile(
    r"(?:кто[\-\s]?то\s+(?:был\s+)?убит"
    r"|убит\s+этой\s+ночью"
    r"|этой\s+ночью\s+убили"
    r"|ночью\s+убили"
    r"|был\s+убит\s+ночью)",
    re.IGNORECASE,
)
_RE_DEATH_LYNCH: Final[re.Pattern[str]] = re.compile(
    r"(?:сегодня\s+повешен" r"|повешен\s+голосован" r"|голосован.*повешен" r"|линчеван)",
    re.IGNORECASE,
)
_RE_DEATH_NAME: Final[re.Pattern[str]] = re.compile(
    r"(?:убит|повешен|убили|повешен)\s+(?:🌙\s*)?(\S+(?:\s+\S+)?)",
    re.IGNORECASE,
)

_RE_VOTE_FOR: Final[re.Pattern[str]] = re.compile(
    r"(\S+)\s+проголосовал(?:а)?\s+против\s+(\S+)",
    re.IGNORECASE,
)
_RE_VOTE_RESULTS: Final[re.Pattern[str]] = re.compile(
    r"(?:результат[ыи]\s+голосован|голосован\w*\s+заверш)",
    re.IGNORECASE,
)

# Role-reveal: "X был ROLE" / "X была ROLE"
_RE_BYL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(.+?)\s+был(?:а)?\s+(.+)",
    re.IGNORECASE,
)

# Emoji-stripping helper pattern
_RE_LEADING_EMOJI: Final[re.Pattern[str]] = re.compile(
    r"^["
    r"\U0001f300-\U0001f9ff"
    r"\U0001f600-\U0001f64f"
    r"\U0001f680-\U0001f6ff"
    r"\U00002600-\U000027bf"
    r"\U0001f900-\U0001f9ff"
    r"\U0001fa70-\U0001faff"
    r"\U0001f1e6-\U0001f1ff"
    r"\U0000fe00-\U0000fe0f"
    r"\U0000200d"
    r"\U000020e3"
    r"]+\s*",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_leading_emoji(text: str) -> str:
    """Remove a leading emoji run (with trailing whitespace)."""
    return _RE_LEADING_EMOJI.sub("", text).strip()


def _extract_day_number(text: str) -> int | None:
    """Pull a day/night number from text like 'День 3' / 'Ночь 2'."""
    match = _RE_DAY_NUMBER.search(text)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, OverflowError):
            return None
    return None


def _normalize_role_word(word: str) -> str:
    """Strip a Russian instrumental suffix to get the word stem."""
    lower = word.casefold().strip()
    for suffix in _ROLE_INSTRUMENTAL_SUFFIXES:
        if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
            return lower[: -len(suffix)]
    return lower


def _try_resolve_role(candidate: str, registry: RoleRegistry) -> Role | None:
    """Resolve a role from a possibly inflected Russian phrase.

    Order of attempts:
    1. Exact match via ``registry.resolve_alias``.
    2. Strip instrumental suffixes from each word, rejoin, and retry.
    3. Prefix-match against all registered names/aliases.
    """
    candidate_stripped = candidate.strip()
    if not candidate_stripped:
        return None

    # 1. Exact
    role = registry.resolve_alias(candidate_stripped)
    if role is not None:
        return role

    # 2. Per-word suffix strip then re-join
    words = candidate_stripped.split()
    normalized_words = [_normalize_role_word(w) for w in words]
    rejoined = " ".join(normalized_words)
    role = registry.resolve_alias(rejoined)
    if role is not None:
        return role

    # 3. Prefix-match: check if any role name/alias starts with
    #    the normalised phrase.
    for role_candidate in registry.all_roles():
        for name_form in (role_candidate.name_ru, *role_candidate.bot_aliases):
            name_lower = name_form.casefold()
            if name_lower.startswith(rejoined) and len(rejoined) >= 3:
                return role_candidate

    # 4. Single first-word match (handles cases like "Мирным жителем"
    #    where suffix strip on both words might overshoot).
    if len(words) >= 1:
        first_base = _normalize_role_word(words[0])
        role = registry.resolve_alias(first_base)
        if role is not None:
            return role
        for role_candidate in registry.all_roles():
            for name_form in (role_candidate.name_ru, *role_candidate.bot_aliases):
                name_lower = name_form.casefold()
                if name_lower.startswith(first_base) and len(first_base) >= 3:
                    return role_candidate

    return None


def _try_detect_emoji_role(text: str) -> str | None:
    """If text starts with a known role emoji, return its role_id."""
    for emoji, role_id in _EMOJI_TO_ROLE_ID.items():
        if text.startswith(emoji):
            return role_id
    return None


def _find_role_in_text(text: str, registry: RoleRegistry) -> Role | None:
    """Search for any known role name or alias inside text."""
    text_lower = text.casefold()
    for role in registry.all_roles():
        for name_form in (role.name_ru, *role.bot_aliases):
            needle = name_form.casefold()
            if needle in text_lower and len(needle) >= 3:
                return role
    return None


# ---------------------------------------------------------------------------
# Individual parsers (one per kind)
# ---------------------------------------------------------------------------


def _parse_registration(text: str) -> BotMessageParse | None:
    if _RE_REGISTRATION.search(text):
        return BotMessageParse(
            kind=BotMessageKind.REGISTRATION,
            raw_text=text,
        )
    return None


def _parse_phase_change(text: str) -> BotMessageParse | None:
    day_number = _extract_day_number(text)
    to_phase: Phase | None = None

    # Guard: if the text also contains death indicators, it's a death message
    # (not a phase change).  E.g. "🌙 Кто-то был убит этой ночью." → DEATH.
    if _RE_DEATH_NIGHT.search(text) or _RE_DEATH_LYNCH.search(text):
        return None

    # Vote phase (check before day — "голосование" may appear in day context)
    if _RE_VOTE_PHASE.search(text):
        to_phase = Phase.VOTE
    elif _RE_NIGHT_START.search(text):
        to_phase = Phase.NIGHT
    elif _RE_DAY_START.search(text):
        to_phase = Phase.DAY

    # Fallback: standalone "голосование" — but NOT "Голосование завершено"
    # (that's a VOTE-kind message) or "Голосованием повешен" (that's DEATH).
    if to_phase is None:
        text_stripped = text.strip()
        if re.match(r"(?i)^голосован(ие|ия|ию)?\s*$", text_stripped):
            to_phase = Phase.VOTE

    if to_phase is not None:
        return BotMessageParse(
            kind=BotMessageKind.PHASE_CHANGE,
            to_phase=to_phase,
            day_number=day_number,
            raw_text=text,
        )
    return None


def _parse_alive_list(text: str) -> BotMessageParse | None:
    if _RE_ALIVE_LIST.match(text):
        return BotMessageParse(
            kind=BotMessageKind.ALIVE_LIST,
            raw_text=text,
        )
    return None


def _parse_death(text: str) -> BotMessageParse | None:
    cause: str | None = None
    if _RE_DEATH_LYNCH.search(text):
        cause = "lynch"
    elif _RE_DEATH_NIGHT.search(text):
        cause = "night_kill"

    if cause is None:
        return None

    # Try to extract victim name
    player_name: str | None = None
    name_match = _RE_DEATH_NAME.search(text)
    if name_match:
        candidate = name_match.group(1).strip()
        # Filter out noise: if the captured word is a common stop word, skip
        noise_words = {"этой", "ночью", "был", "кто-то", "голосованием"}
        if candidate.casefold() not in noise_words and len(candidate) >= 2:
            player_name = candidate

    return BotMessageParse(
        kind=BotMessageKind.DEATH,
        death_cause=cause,
        player_name=player_name,
        raw_text=text,
    )


def _parse_vote(text: str) -> BotMessageParse | None:
    # "X проголосовал против Y"
    vote_match = _RE_VOTE_FOR.search(text)
    if vote_match:
        return BotMessageParse(
            kind=BotMessageKind.VOTE,
            voter_name=vote_match.group(1),
            target_name=vote_match.group(2),
            raw_text=text,
        )

    # "Результаты голосования" / "Голосование завершено"
    if _RE_VOTE_RESULTS.search(text):
        return BotMessageParse(
            kind=BotMessageKind.VOTE,
            raw_text=text,
        )

    return None


def _parse_role_reveal(text: str, registry: RoleRegistry) -> BotMessageParse | None:
    # PRIMARY: "X был ROLE" / "X была ROLE"
    byl_match = _RE_BYL_PATTERN.search(text)
    if byl_match:
        player_candidate = byl_match.group(1).strip()
        role_candidate = byl_match.group(2).strip()

        # Clean up role candidate: remove trailing punctuation, limit length
        role_candidate = role_candidate.rstrip(".,;:!?").strip()
        # If there's a comma, only take the first part (before comma)
        if "," in role_candidate:
            role_candidate = role_candidate.split(",")[0].strip()
        # Limit candidate length to avoid processing spam
        if len(role_candidate) > 100:
            role_candidate = role_candidate[:100].rsplit(" ", 1)[0]

        role = _try_resolve_role(role_candidate, registry)
        if role is not None:
            # Clean up player name: strip leading emoji
            clean_player = _strip_leading_emoji(player_candidate)
            return BotMessageParse(
                kind=BotMessageKind.ROLE_REVEAL,
                confidence=1.0,
                role_id=role.id,
                player_name=clean_player if clean_player else None,
                raw_text=text,
            )

    # FALLBACK: starts with known emoji + text contains role name
    emoji_role_id = _try_detect_emoji_role(text)
    if emoji_role_id is not None:
        role = registry.get(emoji_role_id)
        # Check if any known name/alias appears in the text
        role_in_text = _find_role_in_text(text, registry)
        if role_in_text is not None and role_in_text.id == emoji_role_id:
            clean_text = _strip_leading_emoji(text)
            player_name = clean_text.split()[0] if clean_text.split() else None
            return BotMessageParse(
                kind=BotMessageKind.ROLE_REVEAL,
                confidence=0.9,
                role_id=role.id,
                player_name=player_name,
                raw_text=text,
            )
        # Even if we don't find the role name in text, emoji alone is
        # a strong signal — but only if the text has enough content
        text_no_emoji = _strip_leading_emoji(text)
        if text_no_emoji and len(text_no_emoji) >= 2:
            # Check if the text after emoji looks like a name
            parts = text_no_emoji.split()
            if parts:
                first_word = parts[0]
                # Heuristic: if the first word after emoji is not a common
                # Russian stop word and not all digits, treat as role_reveal
                stop_words = {
                    "кто",
                    "что",
                    "это",
                    "тут",
                    "вот",
                    "да",
                    "нет",
                    "не",
                    "и",
                    "а",
                    "но",
                }
                if first_word.casefold() not in stop_words and not first_word.isdigit():
                    return BotMessageParse(
                        kind=BotMessageKind.ROLE_REVEAL,
                        confidence=0.9,
                        role_id=role.id,
                        player_name=first_word,
                        raw_text=text,
                    )

    # Second fallback: text contains "—" followed by a role name
    # e.g. "🕵🏼 user_1 — Комиссар Каттани"
    dash_match = re.search(r"—\s*(.+?)(?:\s*$)", text)
    if dash_match:
        after_dash = dash_match.group(1).strip().rstrip(".,;:!?")
        role = _try_resolve_role(after_dash, registry)
        if role is not None:
            # Extract player name from before the dash
            before_dash = text[: dash_match.start()].strip()
            clean_player = _strip_leading_emoji(before_dash)
            player_name = clean_player.split()[0] if clean_player.split() else None
            return BotMessageParse(
                kind=BotMessageKind.ROLE_REVEAL,
                confidence=0.95,
                role_id=role.id,
                player_name=player_name,
                raw_text=text,
            )

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_bot_message(
    text: str | None,
    *,
    registry: RoleRegistry,
) -> BotMessageParse | None:
    """Parse a single bot message into its semantic kind.

    Parameters
    ----------
    text :
        Message text as-is (may contain emoji).
    registry :
        ``RoleRegistry`` for alias resolution.

    Returns
    -------
    ``BotMessageParse`` with ``kind=UNKNOWN`` if no pattern matched
    (for C3 statistics).  ``None`` for empty/None input.

    Contract
    --------
    - NEVER raises for any string input (including garbage).
    - NEVER raises for emoji-only / 1000-char messages.
    - Empty / ``None`` input → ``None`` (not ``UNKNOWN``).
    """
    # Guard: empty / None
    if text is None:
        return None
    try:
        if not isinstance(text, str):
            return None
        stripped = text.strip()
        if not stripped:
            return None
    except Exception:  # pragma: no cover — defensive
        return None

    # Order of checks matters — emoji/phase patterns can overlap.
    # 1. Registration
    result = _parse_registration(stripped)
    if result is not None:
        return result

    # 2. Phase change (night / day / vote)
    result = _parse_phase_change(stripped)
    if result is not None:
        return result

    # 3. Alive list
    result = _parse_alive_list(stripped)
    if result is not None:
        return result

    # 4. Death
    result = _parse_death(stripped)
    if result is not None:
        return result

    # 5. Vote
    result = _parse_vote(stripped)
    if result is not None:
        return result

    # 6. Role reveal (most complex — last)
    result = _parse_role_reveal(stripped, registry)
    if result is not None:
        return result

    # 7. Unknown — bot message that we couldn't classify
    return BotMessageParse(
        kind=BotMessageKind.UNKNOWN,
        confidence=0.0,
        raw_text=text,
    )
