"""Normalisation of player names printed by the game bot.

The bot renders nicknames with club markers, emoji, mathematical alphabets and
combining decorations, while the scanner stores Telegram's ``first_name`` (or a
``username``). Exact string equality is therefore too strict for ALIVE_LIST,
death, vote and reveal messages: ``"| Дмитрий"`` and ``"Дмитрий"`` are the same
player.

This module is pure domain code (stdlib only): it produces *comparison keys*,
never a replacement display name. The original spelling is always kept for the
UI, the event log and ``GameState``.
"""

from __future__ import annotations

import unicodedata
from typing import Final

#: Unicode categories that never carry a stable, human-readable name character.
#: ``Mn``/``Me``/``Mc`` are combining marks, ``Cf`` includes ZWJ and variation
#: selectors, ``Cc`` are control characters.
_IGNORED_CATEGORIES: Final[frozenset[str]] = frozenset({"Mn", "Me", "Mc", "Cf", "Cc"})

#: Characters explicitly stripped from both ends of a comparison key.
_TRIM_CHARS: Final[str] = " \t\r\n|·•.,;:!?~_-—–'«»\"()[]{}"

#: Letter categories that count as a "real" name character. ``Lm`` (modifier
#: letters such as ``ᵗ``/``ᵏ`` in club tags) is deliberately excluded.
_CORE_LETTER_CATEGORIES: Final[frozenset[str]] = frozenset({"Lu", "Ll", "Lt", "Lo"})


def _is_core_char(char: str) -> bool:
    """Return whether ``char`` is a base letter or digit, not a decoration."""
    category = unicodedata.category(char)
    return category.startswith("N") or category in _CORE_LETTER_CATEGORIES


def strip_decorations(name: str) -> str:
    """Trim leading/trailing emoji, symbols, marks and punctuation.

    Interior decorations are kept: ``"Дэшка🏹"`` → ``"Дэшка"``, but a nickname
    consisting only of emoji (``"🇦🇪"``) yields ``""``. Callers must fall back
    to :func:`normalize_name` in that case so such names stay matchable.

    Args:
        name: raw display name as written by Telegram or the game bot.

    Returns:
        The central readable part of the name, or an empty string.
    """
    text = name.strip()
    start = 0
    end = len(text)
    while start < end and not _is_core_char(text[start]):
        start += 1
    while end > start and not _is_core_char(text[end - 1]):
        end -= 1
    return text[start:end]


def normalize_name(name: str) -> str:
    """Build a case-insensitive, decoration-tolerant comparison key.

    Applies NFKC (mathematical bold/italic alphabets → plain letters), drops
    combining marks and format characters, casefolds and collapses whitespace.
    NFKC is applied *after* :func:`strip_decorations` scoring so that symbols
    like ``ℹ`` are not mistaken for a real name character.

    Args:
        name: raw display name.

    Returns:
        Normalised key, possibly empty for a name made only of decorations.
    """
    text = unicodedata.normalize("NFKC", name)
    text = "".join(
        char
        for char in text
        if unicodedata.category(char) not in _IGNORED_CATEGORIES and char not in "\u200d\ufeff"
    )
    text = text.casefold()
    text = " ".join(text.split())
    return text.strip(_TRIM_CHARS)


def name_match_keys(name: str) -> frozenset[str]:
    """Return every plausible comparison key for one raw display name.

    Variants cover the three ways club bots spell nicknames:

    - verbatim: ``"🕷️ᵗᵏ| Shinichiro🌝🖤"`` → full normalised key;
    - club marker segments: the same string also yields ``"shinichiro"``;
    - decorative edges: ``"Дэшка🏹"`` also yields ``"дэшка"``.

    A name made only of decorations (``"🇦🇪"``) still has its full key, so it
    remains matchable when Telegram reports the same emoji-only nickname.

    Args:
        name: raw display name.

    Returns:
        Frozen set of non-empty comparison keys.
    """
    keys: set[str] = set()
    full = normalize_name(name)
    if full:
        keys.add(full)

    cleaned = strip_decorations(name)
    if cleaned:
        key = normalize_name(cleaned)
        if key:
            keys.add(key)

    if "|" in name:
        for segment in name.split("|"):
            segment_clean = strip_decorations(segment)
            if not segment_clean:
                continue
            key = normalize_name(segment_clean)
            if not key:
                continue
            # A one-letter segment is kept only when it is the readable start
            # of the whole nickname ("н 🐮" → "н"), not a stray club-tag letter
            # like "ℹ" in "🇷🇺ℹ | user_1 | 💋".
            if len(key) >= 2 or full.startswith(key) or len(full) <= 2:
                keys.add(key)
    return frozenset(keys)


def names_match(left: str, right: str) -> bool:
    """Return whether two raw spellings refer to the same display name.

    Args:
        left: first spelling (e.g. Telegram ``first_name``).
        right: second spelling (e.g. bot's ALIVE_LIST entry).

    Returns:
        ``True`` when their comparison keys intersect.
    """
    return bool(name_match_keys(left) & name_match_keys(right))
