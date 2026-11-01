"""Нормализация речи игроков для передачи в контекст LLM.

Модуль чистый: без I/O, без ``async``, только stdlib (``re``, ``typing``).
Все функции — синхронные, потому что работают за микросекунды и не
блокируют event loop.
"""

from __future__ import annotations

import re
from typing import Final

DEFAULT_MAX_CHARS: Final[int] = 2000

_EMOJI_PATTERN: Final[re.Pattern[str]] = re.compile(
    "["
    "\U0001f300-\U0001f9ff"  # symbols & pictographs
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport
    "\U00002600-\U000027bf"  # misc symbols, dingbats
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa70-\U0001faff"  # extended-A
    "\U0001f1e6-\U0001f1ff"  # flags
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"  # zero-width joiner
    "\U000020e3"  # combining keycap
    "]+",
    flags=re.UNICODE,
)
_WHITESPACE_RUN: Final[re.Pattern[str]] = re.compile(r"[ \t]+")
_BLANK_LINES_RUN: Final[re.Pattern[str]] = re.compile(r"\n{3,}")
_CONTROL_CHARS: Final[re.Pattern[str]] = re.compile(
    "[\U00000000-\U00000008\U0000000b\U0000000c\U0000000e-\U0000001f\U0000007f]"
)
ELLIPSIS: Final[str] = "\u2026"


def remove_emoji(text: str) -> str:
    """Удалить эмодзи (и их модификаторы).

    Кириллица, латиница и пунктуация сохраняются.

    Args:
        text: исходный текст сообщения.

    Returns:
        Текст без эмодзи.
    """
    return _EMOJI_PATTERN.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Привести пробелы и переводы строк к аккуратному виду.

    - control-символы удаляются (кроме ``\\n`` и ``\\t``);
    - пробелы и табы схлопываются в один пробел;
    - три и более переводов строки схлопываются в два;
    - ведущие и завершающие пробелы обрезаются.

    Args:
        text: исходный текст.

    Returns:
        Нормализованный текст.
    """
    cleaned = _CONTROL_CHARS.sub("", text)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned)
    cleaned = _BLANK_LINES_RUN.sub("\n\n", cleaned)
    return cleaned.strip()


def truncate(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Жёстко обрезать текст до ``max_chars`` символов.

    Если текст длиннее, он обрезается до ``max_chars - 1`` символа и в
    конец добавляется ``U+2026`` (…).

    Args:
        text: исходный текст.
        max_chars: максимальная длина результата в символах.

    Returns:
        Текст длиной не более ``max_chars``.

    Raises:
        ValueError: если ``max_chars < 1``.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError(f"max_chars must be int, got {type(max_chars).__name__}")
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + ELLIPSIS


def prepare_for_llm(
    text: str | None,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    strip_emoji: bool = True,
) -> str:
    """Очистить текст сообщения для контекста LLM.

    Pipeline строгий: emoji → whitespace → truncate → empty-check.

    - ``None`` или ``""`` → ``""``;
    - ``remove_emoji`` (если ``strip_emoji=True``);
    - ``normalize_whitespace``;
    - ``truncate(max_chars)``;
    - если после всех шагов пусто → ``""``.

    Args:
        text: исходный текст, может быть ``None``.
        max_chars: максимальная длина результата.
        strip_emoji: удалять ли эмодзи.

    Returns:
        Чистая строка (возможно пустая).

    Raises:
        ValueError: если ``max_chars < 1``.
    """
    if text is None or text == "":
        return ""
    prepared = remove_emoji(text) if strip_emoji else text
    prepared = normalize_whitespace(prepared)
    prepared = truncate(prepared, max_chars)
    if prepared.strip() == "":
        return ""
    return prepared
