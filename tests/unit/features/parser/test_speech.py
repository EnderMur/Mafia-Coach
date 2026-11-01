"""Tests for speech normalization helpers used to build LLM context."""

from __future__ import annotations

import pytest

from features.parser.speech import (
    DEFAULT_MAX_CHARS,
    normalize_whitespace,
    prepare_for_llm,
    remove_emoji,
    truncate,
)

# --- remove_emoji ---------------------------------------------------------


def test_remove_emoji_keeps_words() -> None:
    """Эмодзи удаляются, а слова вокруг остаются."""
    cleaned = remove_emoji("привет 🕵️‍♂️ мир")
    assert "🕵" not in cleaned
    assert "привет" in cleaned
    assert "мир" in cleaned
    assert normalize_whitespace(cleaned) == "привет мир"


def test_remove_emoji_no_emoji_unchanged() -> None:
    """Текст без эмодзи не меняется."""
    text = "Я мафия, голосуйте в Петю!"
    assert remove_emoji(text) == text


def test_remove_emoji_only_emoji() -> None:
    """Строка из одних эмодзи становится пустой."""
    assert remove_emoji("🎉🎉🎉") == ""


def test_remove_emoji_preserves_cyrillic_and_punctuation() -> None:
    """Кириллица, цифры и пунктуация сохраняются."""
    assert remove_emoji("Ночь 1: ...?!,") == "Ночь 1: ...?!,"


def test_remove_emoji_strips_flags_and_keycaps() -> None:
    """Флаги и keycap-модификаторы удаляются, сама цифра остаётся."""
    assert remove_emoji("🇷🇺 1️⃣ тест") == " 1 тест"


# --- normalize_whitespace -------------------------------------------------


def test_normalize_whitespace_collapses_spaces() -> None:
    """Пробелы схлопываются, края обрезаются."""
    assert normalize_whitespace("  a   b  ") == "a b"


def test_normalize_whitespace_collapses_blank_lines() -> None:
    """3+ перевода строки схлопываются в 2."""
    assert normalize_whitespace("a\n\n\n\nb") == "a\n\nb"


def test_normalize_whitespace_keeps_single_newline() -> None:
    """Одиночный перевод строки сохраняется."""
    assert normalize_whitespace("a\nb") == "a\nb"


def test_normalize_whitespace_removes_control_chars() -> None:
    """Control-символы (кроме \\n и \\t) удаляются."""
    assert normalize_whitespace("a\x00b\x07c\x7f") == "abc"


def test_normalize_whitespace_tabs_become_space() -> None:
    """Табы превращаются в пробелы."""
    assert normalize_whitespace("a\t\tb") == "a b"


def test_normalize_whitespace_empty_stays_empty() -> None:
    """Пустая строка остаётся пустой."""
    assert normalize_whitespace("") == ""


# --- truncate -------------------------------------------------------------


def test_truncate_shorter_text_unchanged() -> None:
    """Короткий текст возвращается без изменений и без ellipsis."""
    assert truncate("abcdef", max_chars=10) == "abcdef"


def test_truncate_exact_length_unchanged() -> None:
    """Текст ровно по лимиту не обрезается."""
    assert truncate("abcde", max_chars=5) == "abcde"


def test_truncate_adds_ellipsis() -> None:
    """Длинный текст обрезается до max_chars-1 + один символ …."""
    result = truncate("abcdefghij", max_chars=5)
    assert result == "abcd…"
    assert len(result) == 5


def test_truncate_default_limit() -> None:
    """По умолчанию лимит — 2000 символов."""
    assert DEFAULT_MAX_CHARS == 2000
    assert len(truncate("я" * 5000)) == 2000


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_truncate_rejects_non_positive_limit(bad: int) -> None:
    """max_chars < 1 — ValueError."""
    with pytest.raises(ValueError, match="max_chars"):
        truncate("abc", max_chars=bad)


@pytest.mark.parametrize("bad", [True, "5", 5.0, None])
def test_truncate_rejects_non_int_limit(bad: object) -> None:
    """max_chars не int (в том числе bool) — TypeError."""
    with pytest.raises(TypeError, match="max_chars"):
        truncate("abc", max_chars=bad)  # type: ignore[arg-type]


# --- prepare_for_llm ------------------------------------------------------


def test_prepare_for_llm_none() -> None:
    """None → пустая строка."""
    assert prepare_for_llm(None) == ""


def test_prepare_for_llm_empty_string() -> None:
    """Пустая строка → пустая строка."""
    assert prepare_for_llm("") == ""


def test_prepare_for_llm_whitespace_only() -> None:
    """Строка из пробелов → пустая строка."""
    assert prepare_for_llm("   ") == ""


def test_prepare_for_llm_emoji_only() -> None:
    """Только эмодзи → пустая строка, а не пробел."""
    assert prepare_for_llm("🕵️‍♂️") == ""


def test_prepare_for_llm_keeps_emoji_when_disabled() -> None:
    """strip_emoji=False сохраняет эмодзи."""
    result = prepare_for_llm("привет 🎉", strip_emoji=False)
    assert result == "привет 🎉"


def test_prepare_for_llm_preserves_url() -> None:
    """URL не ломается нормализацией."""
    url = "https://example.com/x?y=1"
    assert prepare_for_llm(f"смотри {url}") == f"смотри {url}"


def test_prepare_for_llm_truncates_long_cyrillic() -> None:
    """5000 кириллических символов обрезаются до 2000, unicode-безопасно."""
    result = prepare_for_llm("я" * 5000)
    assert len(result) == 2000
    assert result.endswith("…")
    assert result.startswith("я")


def test_prepare_for_llm_custom_max_chars() -> None:
    """Кастомный max_chars соблюдается."""
    result = prepare_for_llm("abcdefghij", max_chars=4)
    assert result == "abc…"


def test_prepare_for_llm_normalizes_mixed_noise() -> None:
    """Полный pipeline: эмодзи, лишние пробелы, пустые строки."""
    raw = "  Мафия  🕵️‍♂️ \n\n\n\n  молчит\tсегодня  "
    # Пробельные серии схлопываются, пустые строки — до одной, края обрезаны.
    assert prepare_for_llm(raw) == "Мафия \n\n молчит сегодня"
