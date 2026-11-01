"""Extract speech-style tags per player from a game's messages.

The module is pure: no I/O, no async. Text is normalized through
:func:`features.parser.speech.prepare_for_llm` before any pattern is matched, so
emoji and ragged whitespace never affect the result.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from domain.events import MessageEvent
from domain.roles import RoleRegistry
from features.parser.speech import prepare_for_llm

TAG_FREQUENT_APOLOGY: Final[str] = "frequent_apology"
TAG_AGGRESSIVE_ACCUSATION: Final[str] = "aggressive_accusation"
TAG_AVOID_CHECK: Final[str] = "avoid_check"
TAG_PASSIVE_SILENCE: Final[str] = "passive_silence"

MAX_EVIDENCE_ITEMS: Final[int] = 5
MAX_EVIDENCE_CHARS: Final[int] = 100

PASSIVE_MIN_MESSAGES: Final[int] = 2
APOLOGY_THRESHOLD: Final[int] = 2
AGGRESSION_THRESHOLD: Final[int] = 2
AVOID_CHECK_THRESHOLD: Final[int] = 1

APOLOGY_WEIGHT: Final[float] = 0.7
AGGRESSION_WEIGHT: Final[float] = 0.8
AVOID_CHECK_WEIGHT: Final[float] = 0.6
PASSIVE_WEIGHT: Final[float] = 0.4

_APOLOGY_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(извин\w+|прости\w*|виноват\w*|моя\s+вина)\b",
    re.IGNORECASE,
)
_AGGRESSION_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(ты\s+(мафия|маф|убийца|дон)|"
    r"вешать|повесить|убить|застрелить|подозрительн\w+)\b",
    re.IGNORECASE,
)
_AVOID_CHECK_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(не\s+буду\s+(проверять|говорить|отвечать)|"
    r"не\s+хочу\s+проверять|не\s+моя\s+роль)\b",
    re.IGNORECASE,
)


class PlayerStyle(BaseModel):
    """One detected style tag with evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tag: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(min_length=1, max_length=5)


def extract_styles(
    events: list[MessageEvent],
    *,
    registry: RoleRegistry,
) -> dict[int, list[PlayerStyle]]:
    """Extract style tags per player from MessageEvents.

    Returns dict[player_id, list[PlayerStyle]]. Players без detected patterns —
    НЕ включены в dict. Игрок без единого :class:`MessageEvent` тоже не попадает
    в результат: это absent, а не passive.

    Patterns:

    - ``frequent_apology``: >=2 apology messages → weight 0.7
    - ``aggressive_accusation``: >=2 aggression messages → weight 0.8
    - ``avoid_check``: >=1 avoid-check message → weight 0.6
    - ``passive_silence``: <2 total messages В events → weight 0.4

    Args:
        events: All message events of one game.
        registry: Role registry; reserved for role-aware style rules and kept
            in the signature so callers do not change when those land.

    Returns:
        Mapping player_id → detected styles. Evidence держит до 5
        нормализованных сообщений, каждое обрезано до 100 символов.
    """
    del registry  # reserved: role-aware style rules are not part of step 8c

    messages_by_author: dict[int, list[str]] = {}
    total_by_author: dict[int, int] = {}
    for event in events:
        author_id = event.payload.author_id
        total_by_author[author_id] = total_by_author.get(author_id, 0) + 1
        text = prepare_for_llm(event.payload.text)
        if text:
            messages_by_author.setdefault(author_id, []).append(text)

    styles: dict[int, list[PlayerStyle]] = {}
    for author_id, total in total_by_author.items():
        texts = messages_by_author.get(author_id, [])
        detected: list[PlayerStyle] = []

        for tag, weight, pattern, threshold in (
            (TAG_FREQUENT_APOLOGY, APOLOGY_WEIGHT, _APOLOGY_RE, APOLOGY_THRESHOLD),
            (TAG_AGGRESSIVE_ACCUSATION, AGGRESSION_WEIGHT, _AGGRESSION_RE, AGGRESSION_THRESHOLD),
            (TAG_AVOID_CHECK, AVOID_CHECK_WEIGHT, _AVOID_CHECK_RE, AVOID_CHECK_THRESHOLD),
        ):
            evidence = _collect_evidence(texts, pattern)
            if len(evidence) >= threshold:
                detected.append(PlayerStyle(tag=tag, weight=weight, evidence=evidence))

        if total < PASSIVE_MIN_MESSAGES:
            detected.append(
                PlayerStyle(
                    tag=TAG_PASSIVE_SILENCE,
                    weight=PASSIVE_WEIGHT,
                    evidence=_truncate_evidence(texts) or ["<нет пригодного текста>"],
                )
            )

        if detected:
            styles[author_id] = detected

    return styles


def _collect_evidence(texts: list[str], pattern: re.Pattern[str]) -> list[str]:
    """Return up to five normalized messages that match ``pattern``."""
    matched = [text for text in texts if pattern.search(text)]
    return _truncate_evidence(matched)


def _truncate_evidence(texts: list[str]) -> list[str]:
    """Cut each message to the evidence length limit and keep at most five."""
    return [text[:MAX_EVIDENCE_CHARS] for text in texts[:MAX_EVIDENCE_ITEMS]]
