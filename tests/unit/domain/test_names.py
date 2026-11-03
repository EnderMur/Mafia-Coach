"""Tests for club-nickname normalisation used by ALIVE_LIST matching."""

from __future__ import annotations

from domain.names import name_match_keys, names_match, normalize_name, strip_decorations


def test_normalize_name_handles_fancy_unicode() -> None:
    """NFKC + casefold + combining-mark removal produce stable keys."""
    assert names_match("𝖁𝖊𝖑𝖑𝖆𝖓𝖊 ᥫ᭡", "Vellane ᥫ")
    assert normalize_name("Дэшка🏹") == "дэшка🏹"
    assert normalize_name("| Дмитрий") == "дмитрий"


def test_strip_decorations_keeps_interior_name() -> None:
    """Leading/trailing emoji are dropped, an emoji-only nickname is not lost."""
    assert strip_decorations("Дэшка🏹") == "Дэшка"
    assert strip_decorations(" Shinichiro🌝🖤") == "Shinichiro"
    assert strip_decorations("🇦🇪") == ""
    assert normalize_name("🇦🇪")  # emoji-only names still have a full key


def test_names_match_club_markers() -> None:
    """Decorated bot spellings resolve to the plain Telegram first_name."""
    assert names_match("| Дмитрий", "Дмитрий")
    assert names_match("🕷️ᵗᵏ| Shinichiro🌝🖤", "Shinichiro")
    assert names_match("🇷🇺ℹ | user_1 | 💋", "user_1")
    assert names_match("Дэшка🏹", "Дэшка")
    assert names_match("𝖁𝖊𝖑𝖑𝖆𝖓𝖊 ᥫ᭡", "Vellane ᥫ")


def test_names_match_does_not_merge_different_players() -> None:
    """Short but distinct nicknames stay distinct."""
    assert not names_match("мяу", "moon")
    assert not names_match("Е А", "Е")
    assert "shinichiro" in name_match_keys("🕷️ᵗᵏ| Shinichiro🌝🖤")
