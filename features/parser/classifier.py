"""Binary-ish classification of incoming Telegram messages.

Модуль чистый: никакого I/O, никаких импортов из ``infra``/``app``/
``domain``, никаких ``async``-функций. Классификатор — детерминированная
функция от входных данных.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class MessageKind(str, Enum):
    """Classification of an incoming Telegram message."""

    BOT_MESSAGE = "bot_message"  # от игрового бота (game events)
    PLAYER_SPEECH = "player_speech"  # речь игрока (контекст для LLM)
    IRRELEVANT = "irrelevant"  # системное, пустое, служебное


DEFAULT_MIN_SPEECH_LENGTH: Final[int] = 3


def _validate_int(name: str, value: object) -> None:
    """Ensure ``value`` is a real ``int`` and not a ``bool``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")


def classify_message(
    *,
    text: str | None,
    from_id: int | None,
    bot_id: int,
    min_speech_length: int = DEFAULT_MIN_SPEECH_LENGTH,
) -> MessageKind:
    """Classify a raw Telegram message into one of three kinds.

    Rules (in order):

    1. ``text`` is ``None`` or ``text.strip() == ""`` → ``IRRELEVANT``
       (системное сообщение, стикер без подписи, служебное).
    2. ``from_id == bot_id`` → ``BOT_MESSAGE``
       (сообщения бота могут быть короткими, например «Ночь»).
    3. ``from_id is None`` → ``IRRELEVANT``
       (анонимный канал, неизвестный источник).
    4. ``len(text.strip()) < min_speech_length`` → ``IRRELEVANT``
       (слишком короткая речь игрока, шум).
    5. иначе → ``PLAYER_SPEECH``.

    Args:
        text: message text (Telethon ``message.message``), may be ``None``.
        from_id: sender user id, may be ``None`` for anonymous messages.
        bot_id: игровой бот id (из settings).
        min_speech_length: минимальная длина «осмысленной» речи игрока.
            Default = 3.

    Returns:
        ``MessageKind`` for the message.

    Raises:
        ValueError: если ``bot_id`` не ``int`` (в том числе ``bool``) или
            ``min_speech_length`` отрицательный.
    """
    _validate_int("bot_id", bot_id)
    _validate_int("min_speech_length", min_speech_length)
    if min_speech_length < 0:
        raise ValueError(f"min_speech_length must be >= 0, got {min_speech_length}")
    # Rule 1: пустой текст — системное/служебное сообщение.
    if text is None or text.strip() == "":
        return MessageKind.IRRELEVANT

    # Rule 2: сообщение от игрового бота (проверяется до длины текста).
    if from_id == bot_id:
        return MessageKind.BOT_MESSAGE

    # Rule 3: неизвестный отправитель.
    if from_id is None:
        return MessageKind.IRRELEVANT

    # Rule 4: слишком короткая речь игрока.
    if len(text.strip()) < min_speech_length:
        return MessageKind.IRRELEVANT

    # Rule 5: осмысленная речь игрока.
    return MessageKind.PLAYER_SPEECH
