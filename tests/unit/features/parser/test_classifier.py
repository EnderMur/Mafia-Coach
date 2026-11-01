"""Tests for the pure message classifier."""

from __future__ import annotations

import pytest

from features.parser.classifier import (
    DEFAULT_MIN_SPEECH_LENGTH,
    MessageKind,
    classify_message,
)


BOT_ID = 4242


def test_bot_text_is_bot_message() -> None:
    """Сообщение от игрового бота классифицируется как BOT_MESSAGE."""
    assert (
        classify_message(text="Город засыпает", from_id=BOT_ID, bot_id=BOT_ID)
        is MessageKind.BOT_MESSAGE
    )


def test_bot_short_text_is_bot_message() -> None:
    """Короткий текст бота ("Ночь") — BOT_MESSAGE: правило бота выше min_length."""
    assert classify_message(text="Ночь", from_id=BOT_ID, bot_id=BOT_ID) is MessageKind.BOT_MESSAGE


def test_bot_none_text_is_irrelevant() -> None:
    """Пустой текст проверяется раньше отправителя: None у бота — IRRELEVANT."""
    assert classify_message(text=None, from_id=BOT_ID, bot_id=BOT_ID) is MessageKind.IRRELEVANT


def test_player_long_text_is_player_speech() -> None:
    """Осмысленная речь игрока — PLAYER_SPEECH."""
    assert (
        classify_message(text="Привет, я мирный", from_id=100, bot_id=BOT_ID)
        is MessageKind.PLAYER_SPEECH
    )


def test_player_short_text_is_irrelevant() -> None:
    """Два символа при min_speech_length=3 — шум."""
    assert classify_message(text="ок", from_id=100, bot_id=BOT_ID) is MessageKind.IRRELEVANT


def test_player_short_text_allowed_when_min_zero() -> None:
    """При min_speech_length=0 короткая речь проходит."""
    assert (
        classify_message(text="ок", from_id=100, bot_id=BOT_ID, min_speech_length=0)
        is MessageKind.PLAYER_SPEECH
    )


def test_empty_text_is_irrelevant() -> None:
    """Пустая строка — IRRELEVANT."""
    assert classify_message(text="", from_id=100, bot_id=BOT_ID) is MessageKind.IRRELEVANT


def test_whitespace_only_text_is_irrelevant() -> None:
    """Строка из пробелов/табов/переводов строки — IRRELEVANT."""
    assert classify_message(text="   \t\n", from_id=100, bot_id=BOT_ID) is MessageKind.IRRELEVANT


def test_none_text_from_player_is_irrelevant() -> None:
    """None от игрока — IRRELEVANT."""
    assert classify_message(text=None, from_id=100, bot_id=BOT_ID) is MessageKind.IRRELEVANT


def test_none_from_id_is_irrelevant() -> None:
    """Анонимный источник (from_id=None) — IRRELEVANT даже при тексте."""
    assert classify_message(text="hello", from_id=None, bot_id=BOT_ID) is MessageKind.IRRELEVANT


def test_zero_ids_match_as_bot() -> None:
    """from_id == bot_id == 0 — совпадение бота работает и с нулём."""
    assert classify_message(text="День", from_id=0, bot_id=0) is MessageKind.BOT_MESSAGE


def test_bool_bot_id_rejected() -> None:
    """bool — подкласс int, но bot_id=True должен быть отклонён."""
    with pytest.raises(ValueError, match="bot_id"):
        classify_message(text="привет", from_id=1, bot_id=True)


@pytest.mark.parametrize("invalid_bot_id", ["42", 4.2, None])
def test_non_int_bot_id_rejected(invalid_bot_id: object) -> None:
    """Не-int bot_id отклоняется с ValueError."""
    with pytest.raises(ValueError, match="bot_id"):
        classify_message(text="привет", from_id=1, bot_id=invalid_bot_id)  # type: ignore[arg-type]


def test_negative_min_speech_length_rejected() -> None:
    """Отрицательный min_speech_length — ValueError."""
    with pytest.raises(ValueError, match="min_speech_length"):
        classify_message(text="привет", from_id=1, bot_id=BOT_ID, min_speech_length=-1)


def test_default_min_speech_length() -> None:
    """Значение по умолчанию — 3 символа."""
    assert DEFAULT_MIN_SPEECH_LENGTH == 3


def test_speech_around_min_length_boundary() -> None:
    """Ровно min_speech_length символов — уже речь; на один меньше — шумом."""
    assert (
        classify_message(text="да!", from_id=100, bot_id=BOT_ID, min_speech_length=3)
        is MessageKind.PLAYER_SPEECH
    )
    assert (
        classify_message(text="да", from_id=100, bot_id=BOT_ID, min_speech_length=3)
        is MessageKind.IRRELEVANT
    )


def test_length_uses_stripped_text() -> None:
    """Длина считается по strip(): окружение пробелами не спасает короткую речь."""
    assert (
        classify_message(text="  ок  ", from_id=100, bot_id=BOT_ID, min_speech_length=3)
        is MessageKind.IRRELEVANT
    )


def test_bot_id_required_keyword_only() -> None:
    """Все аргументы — keyword-only (контракт публичного API)."""
    with pytest.raises(TypeError):
        classify_message("привет", 100, BOT_ID)  # type: ignore[call-arg]
