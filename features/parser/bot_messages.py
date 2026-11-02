"""Semantic parser for game-bot messages.

Pure module: no I/O, no ``async``, no imports from ``infra``/``app``/``telethon``.
All functions are synchronous and side-effect free.

Classification priorities (step 7b-fix-1)
----------------------------------------
Bot templates overlap heavily, so the order of the branches in
:func:`parse_bot_message` is load-bearing:

1. empty / ``None``     — returns ``None`` (not ``UNKNOWN``).
2. ``END_GAME``         — first: final report may contain a "Живые игроки"-like
   section (``"Другие пользователи:"``) and must win over ``ALIVE_LIST``.
3. ``ALIVE_LIST``       — the list body contains role emoji.
4. ``REGISTRATION``
5. ``DEATH``            — BEFORE phase change (``"🌙 Кто-то был убит"``).
6. ``PHASE_CHANGE``     — vote start / vote end only.
7. ``VOTE``             — multi-line, collects ``votes``.
8. ``PROTECTION_USED``  — ``"использовал защиту от линчевания"`` is NOT death.
9. ``VOTE_ABSTAIN``
10. ``ROLE_REVEAL``     — *grammatical* player reveals only.
11. ``ROLE_ACTION``     — role "action notifications" (``"Дон выбрал жертву"``).
12. ``PHASE_CHANGE``    — night / day transitions (short messages only).
13. ``SYSTEM``
14. ``UNKNOWN``         — fallback, kept for C3 statistics.
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
    #: Only grammatical player reveals: ``"X был[а] Роль"`` / ``"X — Роль"``.
    ROLE_REVEAL = "role_reveal"
    #: Role action notification ("Роль сделала X") — NOT a player reveal.
    ROLE_ACTION = "role_action"
    VOTE = "vote"
    LAST_WORDS = "last_words"
    GIFT = "gift"
    ALIVE_LIST = "alive_list"
    #: "Кто-то потратил защиту", "Счастливчику повезло".
    PROTECTION_USED = "protection_used"
    #: "X решил никого не вешать".
    VOTE_ABSTAIN = "vote_abstain"
    #: "Город ветров начинает игру", "Автоматический старт игры".
    SYSTEM = "system"
    #: Final report: ``"Игра окончена!"`` + winners / losers / duration.
    END_GAME = "end_game"
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

    # role_reveal / role_action
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
    #: ``[(voter_name, target_name), ...]`` for multi-line vote messages
    #: (``kind=VOTE`` only, otherwise ``None``).
    votes: list[tuple[str, str | None]] | None = None

    # last_words
    last_words: str | None = None

    # alive_list
    #: ``display_name`` strings from the "Живые игроки:" section, as written
    #: by the bot. Resolution to ids happens in ``GameState.apply`` (A2-1).
    alive_player_names: list[str] | None = None
    #: Bot-reported role composition: ``name_ru → count`` (registry matching
    #: is the analyzer's job, not the parser's).
    role_count: dict[str, int] | None = None
    #: Bot-reported "Всего: N" headcount of alive players.
    total_players: int | None = None

    # end_game
    #: ``[(player_name, role_name_ru), ...]`` — победители (достигли цели,
    #: не обязательно выжили). ``role_name_ru`` — как пишет бот (``name_ru``),
    #: резолв в ``role_id`` — дело ``GameState``.
    winners: list[tuple[str, str]] | None = None
    #: ``[(player_name, role_name_ru), ...]`` — остальные участники.
    losers: list[tuple[str, str]] | None = None
    #: ``"Игра длилась: N минут"``.
    duration_minutes: int | None = None

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

#: Verbs used by the bot in *action notifications* ("Роль <verb> …").
#: A message containing one of them is a ROLE_ACTION, never a reveal.
_ACTION_VERB_STEMS: Final[tuple[str, ...]] = (
    "выбрал",
    "выберет",
    "выбирает",
    "выбрала",
    "ушёл",
    "ушел",
    "ушла",
    "ушли",
    "вышел",
    "вышла",
    "вышли",
    "отправил",
    "начал",
    "спрятал",
    "зарядил",
    "пошёл",
    "пошел",
    "пошла",
    "пошли",
    "сделал",
    "превратил",
    "решил",
    "ищет",
    "искать",
    "проверил",
    "проверяет",
    "лечит",
    "вылечил",
    "заблокировал",
    "блокирует",
    "посетил",
    "навестил",
    "убил",
    "поджёг",
    "поджег",
    "ждёт",
    "ждет",
    "ждём",
    "ждем",
    "смотрит",
    "спит",
    "проснулся",
    "проснулась",
    "защитил",
    "защищает",
    "спас",
    "спасла",
    "поменял",
    "взломал",
    "узнал",
    "следил",
    "проверял",
    "лечил",
    "блокировал",
    "выстрелил",
    "застрелил",
    "ходил",
    "сходил",
    "пришёл",
    "пришел",
    "принес",
    "принёс",
    "пытался убить",
    "простить",
    "покушение",
)

#: Words that look like a role but are actually death/permission verbs.
_NON_ROLE_WORDS: Final[frozenset[str]] = frozenset(
    {
        "убит",
        "убита",
        "убиты",
        "убить",
        "повешен",
        "повешена",
        "повешены",
        "повесить",
        "задушен",
        "задушена",
        "отравлен",
        "отравлена",
        "застрелен",
        "застрелена",
        "ранен",
        "ранена",
        "вылечен",
        "вылечена",
        "заблокирован",
        "заблокирована",
        "спасён",
        "спасен",
        "спасена",
        "раскрыт",
        "раскрыта",
        "изгнан",
        "изгнана",
        "сожжён",
        "сожжен",
        "съеден",
    }
)

#: Tokens that can never be a player name.
_NOISE_NAME_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "кто-то",
        "кто",
        "то",
        "этой",
        "этот",
        "эта",
        "ночью",
        "сегодня",
        "был",
        "была",
        "было",
        "были",
        "голосованием",
        "голосование",
        "игрок",
        "игрока",
        "игроки",
        "жители",
        "живых",
        "защиту",
        "жертву",
        "жертва",
        "никого",
        "кого",
        "цель",
        "он",
        "она",
        "оно",
        "уже",
        "это",
    }
)

#: Token that terminates name extraction ("Вася и Маша" → "Вася").
_NAME_STOP_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "и",
        "а",
        "но",
        "или",
        "он",
        "она",
        "оно",
        "они",
        "был",
        "была",
        "было",
        "были",
        "его",
        "её",
        "ее",
        "их",
        "это",
        "который",
        "которая",
        "что",
        "чтобы",
        "с",
        "со",
        "в",
        "во",
        "на",
        "за",
        "под",
        "по",
        "от",
        "до",
        "у",
        "к",
        "не",
        "ни",
        "уже",
        "также",
        "тоже",
    }
)

#: Phase-change detection is disabled for long messages (spec 7b-fix-1).
_PHASE_MAX_LEN: Final[int] = 500

# Max number of tokens kept when extracting a player name.
_MAX_NAME_TOKENS: Final[int] = 3


# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

_RE_LEADING_EMOJI: Final[re.Pattern[str]] = re.compile(
    r"^(?:["
    r"\U0001f300-\U0001faff"
    r"\U00002600-\U000027bf"
    r"\U00002b00-\U00002bff"
    r"\U0001f1e6-\U0001f1ff"
    r"\U00002139\U00002122\U00002049\U0000203c"
    r"\U0000fe00-\U0000fe0f"
    r"\U0000200d"
    r"\U000020e3"
    r"]+\s*)+",
)

# --- registration ---------------------------------------------------------

_RE_REGISTRATION: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:ведётся\s+набор"
    r"|ведется\s+набор"
    r"|набор\s+в\s+игру"
    r"|набор\s+(?:открыт|закрыт)"
    r"|регистрация\s+(?:открыта|закрыта|заверш\w*)"
    r"|записал(?:ся|ось)"
    r"|регистраци[яию])",
)

# --- alive list -----------------------------------------------------------

_RE_ALIVE_LIST: Final[re.Pattern[str]] = re.compile(
    r"^\s*Живые(?:\s+игроки)?\s*(?:\([^)]*\))?\s*:",
    re.IGNORECASE,
)

#: One ``"N. display_name"`` line of the "Живые игроки:" block. Names may
#: contain Cyrillic letters, emoji and spaces — everything after ``N. `` is
#: kept verbatim.
_ALIVE_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*\d+\.\s+(.+?)\s*$",
    re.MULTILINE,
)

# --- death ----------------------------------------------------------------

_RE_DEATH_NIGHT: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:кто[\-\s]?то\s+(?:был[а]?\s+)?убит"
    r"|кто[\-\s]?то\s+погиб"
    r"|сегодня\s+был[а]?\s+жестоко\s+убит"
    r"|жестоко\s+убит"
    r"|этой\s+ночью\s+убили"
    r"|ночью\s+убили"
    r"|убит\s+этой\s+ночью"
    r"|этой\s+ночью\s+(?:был[а]?\s+)?убит"
    r"|\S+\s+был[а]?\s+убит"
    r"|(?:^|[.!?\n])\s*убит[а]?\s+\S)",
)

_RE_DEATH_LYNCH: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:сегодня\s+повешен"
    r"|голосованием\s+повешен"
    r"|повешен\w*\s+голосовани"
    r"|не\s+выдержал\w*.{0,50}?повесил"
    r"|(?:^|[.!?\n])\s*повешен[а]?\s+\S"
    r"|линч[её]ван(?!и))",
)

_RE_NAME_BEFORE_SUICIDE: Final[re.Pattern[str]] = re.compile(
    r"(?i)^\s*(?P<name>[^\n]+?)\s+не\s+выдержал",
)
_RE_NAME_BEFORE_KILLED: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?P<name>[^\s]+)\s+(?:был[а]?|были)\s+убит",
)
_RE_NAME_AFTER_HANG: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:сегодня\s+повешен[а]?|голосованием\s+повешен[а]?|повешен[а]?)"
    r"[\s:\-—]*(?P<name>[^\n]+)",
)
_RE_NAME_AFTER_KILL: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:жестоко\s+убит[а]?|убили|убит[а]?)\s+(?P<name>[^\n]+)",
)

# --- phase change ---------------------------------------------------------

_RE_NIGHT_START: Final[re.Pattern[str]] = re.compile(
    r"(?:🌙|наступает\s+ночь|город\s+засыпает|ночь\s+наступает|^ночь\s*:?\s*\d)",
    re.IGNORECASE,
)
_RE_DAY_START: Final[re.Pattern[str]] = re.compile(
    r"(?:🌝|наступает\s+день|город\s+просыпается|город\s+прос|утро\s*:?\s*\d" r"|^день\s*:?\s*\d)",
    re.IGNORECASE,
)
_RE_VOTE_PHASE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:пришло\s+время\s+определить"
    r"|пришло\s+время\s+голосован"
    r"|голосование\s+окончено"
    r"|начинается\s+голосован"
    r"|голосование\s+продлится"
    r"|время\s+голосования)",
)
_RE_VOTE_PHASE_STANDALONE: Final[re.Pattern[str]] = re.compile(
    r"(?i)^\s*голосование\s*:?\s*$",
)
_RE_DAY_NUMBER: Final[re.Pattern[str]] = re.compile(
    r"(?:День|Утро|Ночь)\s*:?\s*(?:#\s*)?(\d+)",
    re.IGNORECASE,
)
_RE_GAME_END: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:игра\s+окончена|конец\s+игры|игра\s+завершена"
    r"|победа\s+(?:мафии|мирных|города|маньяка))",
)

# --- end game (final report) ----------------------------------------------

#: The final report *starts* with this line; only there is it authoritative
#: (a mid-game message mentioning "игра окончена" stays a phase change).
_END_GAME_RE: Final[re.Pattern[str]] = re.compile(r"^\s*Игра\s+окончена!", re.MULTILINE)

#: One report section: ``"Победители:"`` or ``"Другие пользователи:"`` with its
#: indented ``"N. name - role"`` body, terminated by the next section header or
#: by the duration line.
_SECTION_RE: Final[re.Pattern[str]] = re.compile(
    r"(Победители|Другие\s+пользователи):\s*\n(.*?)(?=\n\s*(?:Другие\s+пользователи|Игра\s+длилась)|\Z)",
    re.DOTALL,
)

#: One ``"N. имя - 🎭 Роль"`` entry inside a report section.
_ENTRY_RE: Final[re.Pattern[str]] = re.compile(r"^\s*\d+\.\s+(.+?)\s+-\s+(.+?)\s*$", re.MULTILINE)

#: ``"Игра длилась: 20 минут"``.
_DURATION_RE: Final[re.Pattern[str]] = re.compile(r"Игра\s+длилась:\s*(\d+)\s*минут")

# --- votes ----------------------------------------------------------------

_VOTE_VERB: Final[str] = r"(?:про)?голос(?:овал[аи]?|ует)(?:\(а\))?"
_VOTE_RE: Final[re.Pattern[str]] = re.compile(
    r"([^\n]+?)\s+голосовал(?:\(а\)|а)?\s+за:\s*(.*?)(?=\n|$)",
    re.UNICODE | re.IGNORECASE,
)
_RE_LAST_WORDS: Final[re.Pattern[str]] = re.compile(
    r"(?is)^\s*Кто-то\s+из\s+жителей\s+слышал,\s*как\s+"
    r"(?P<name>.+?)\s+кричал\s+перед\s+смертью:[ \t]*(?:\n(?P<words>.*))?$"
)
_RE_GIFT: Final[re.Pattern[str]] = re.compile(
    r"(?is)^\s*(?P<giver>.+?)\s+подарил\s+💎(\d+)\s+к\s+(?P<target>.+?)!?\s*$"
)
_RE_VOTE_FOR: Final[re.Pattern[str]] = re.compile(
    rf"(?i)(?P<voter>.{{1,60}}?)\s+{_VOTE_VERB}\s+за\s*:?\s*(?P<target>.+)",
)
_RE_VOTE_AGAINST: Final[re.Pattern[str]] = re.compile(
    rf"(?i)(?P<voter>.{{1,60}}?)\s+{_VOTE_VERB}\s+против\s+(?P<target>.+)",
)
_RE_VOTE_RESULTS: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:результат[ыи]\s+голосован"
    r"|голосован\w*\s+заверш"
    r"|голосован\w*\s+окончен"
    r"|никто\s+не\s+(?:был[а]?\s+)?повешен"
    r"|голосов?\s+не\s+набрано)",
)
_RE_VOTE_ABSTAIN: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:(?P<name>.{1,60}?)\s+решил[аи]?(?:\(а\))?\s+никого\s+не\s+вешать"
    r"|никого\s+не\s+вешать)",
)

# --- protection -----------------------------------------------------------

_RE_PROTECTION: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:кто[\-\s]?то\s+потратил\s+защит"
    r"|счастливчику.{0,50}?повезло"
    r"|использовал[аи]?(?:\(а\))?\s+защиту"
    r"|потратил[аи]?\s+защит)",
)

# --- role reveal / role action -------------------------------------------

_RE_REVEAL_BYL: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?P<name>[^\n]{1,80}?)\s+(?:был|была|было)\s+(?P<role>[^\n]+)$",
)
_RE_REVEAL_DASH: Final[re.Pattern[str]] = re.compile(
    r"^(?P<name>[^\n]{1,80}?)\s*(?:[—–]|\s-\s)\s*(?P<role>[^\n]+)$",
)
_RE_ACTION_VERB: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:" + "|".join(re.escape(stem) for stem in _ACTION_VERB_STEMS) + r")\w*",
)
_RE_ACTION_TARGET: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:выбрал[аи]?(?:\(а\))?|выберет|выбирает|посетил[аи]?|навестил[аи]?"
    r"|проверил[аи]?|убил[аи]?|вылечил[аи]?|заблокировал[аи]?)\s+(?P<target>[^\s,\.!?]+)",
)

# --- system ---------------------------------------------------------------

_RE_SYSTEM: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:город\s+ветров|автоматическ\w*\s+старт(?:\s+игры)?"
    r"|странно.{0,30}этой\s+ночью\s+все\s+остались\s+в\s+живых"
    r"|недостаточно\s+игроков\s+для\s+начала\s+игры)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_leading_emoji(text: str) -> str:
    """Remove a leading emoji run (with trailing whitespace)."""
    previous = ""
    result = text
    while previous != result:
        previous = result
        result = _RE_LEADING_EMOJI.sub("", result).lstrip()
    return result


def _strip_marker_prefix(text: str) -> str:
    """Drop chat-marker prefixes like ``"🇷🇺ℹ | name | 💋"``.

    Returns the first ``|``-separated segment that contains a word
    character; falls back to the original text when there is no ``|``.
    """
    if "|" not in text:
        return text
    for segment in (part.strip() for part in text.split("|")):
        cleaned = _strip_leading_emoji(segment).strip()
        if cleaned and re.search(r"\w", cleaned):
            return cleaned
    return text


def _first_non_empty_line(text: str) -> str:
    """Return the first non-blank line of ``text`` (or ``""``)."""
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


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


def _clean_name(raw: str) -> str | None:
    """Normalise a captured player name; return ``None`` if it is noise."""
    candidate = _strip_marker_prefix(raw.strip())
    candidate = _strip_leading_emoji(candidate).strip()
    if not candidate:
        return None

    selected: list[str] = []
    for token in candidate.split():
        cleaned = token.strip(".,;:!?—–-«»\"'()[]{}")
        if not cleaned:
            continue
        if cleaned.casefold() in _NAME_STOP_TOKENS:
            break
        selected.append(cleaned)
        if len(selected) >= _MAX_NAME_TOKENS:
            break

    if not selected:
        return None
    if selected[0].casefold() in _NOISE_NAME_TOKENS:
        return None

    name = " ".join(selected)
    if len(name) < 2 or not re.search(r"\w", name):
        return None
    return name


def _looks_like_player_token(token: str) -> bool:
    """Heuristic: can this token be a player name (capitalised or user-like)?"""
    if not token:
        return False
    if token.casefold() in _NOISE_NAME_TOKENS:
        return False
    return bool(token[0].isupper() or any(char.isdigit() or char in "_@-" for char in token))


def _try_role_by_name(candidate: str, registry: RoleRegistry) -> Role | None:
    """Resolve a role from a possibly inflected Russian phrase.

    Order of attempts:

    1. Exact match (``registry.resolve_alias`` / Russian name).
    2. Strip instrumental suffixes from every word, re-join, retry.
    3. Prefix match against registered names/aliases.
    4. First-word prefix match (``"Комиссаром Каттани"`` fallbacks).
    """
    cleaned = _strip_leading_emoji(candidate.strip())
    cleaned = cleaned.strip(" .,;:!?—–-«»\"'()")
    if not cleaned:
        return None
    # Cut trailing clauses: "Доном, а user_2 был Мафией" → "Доном".
    for separator in (",", ";", "!", "?", " ("):
        if separator in cleaned:
            cleaned = cleaned.split(separator)[0].strip()
    if not cleaned:
        return None
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rsplit(" ", 1)[0].strip()
        if not cleaned:
            return None

    # 1. Exact
    role = registry.resolve_alias(cleaned)
    if role is not None:
        return role
    role = registry.get_by_name_ru(cleaned)
    if role is not None:
        return role

    words = cleaned.split()
    if not words:
        return None

    # 2. Per-word suffix strip then re-join
    normalized = " ".join(_normalize_role_word(word) for word in words)
    role = registry.resolve_alias(normalized)
    if role is not None:
        return role

    # 3. Prefix-match on the normalised phrase
    if len(normalized) >= 3:
        for role_candidate in registry.all_roles():
            for name_form in (role_candidate.name_ru, *role_candidate.bot_aliases):
                if name_form.casefold().startswith(normalized):
                    return role_candidate

    # 4. First-word prefix match
    first_base = _normalize_role_word(words[0])
    role = registry.resolve_alias(first_base)
    if role is not None:
        return role
    if len(first_base) >= 3:
        for role_candidate in registry.all_roles():
            for name_form in (role_candidate.name_ru, *role_candidate.bot_aliases):
                if name_form.casefold().startswith(first_base):
                    return role_candidate

    return None


def _looks_like_role_word(candidate: str) -> bool:
    """Grammar-only heuristic for an unresolved role word."""
    cleaned = _strip_leading_emoji(candidate.strip()).strip(" .,;:!?—–-«»\"'()")
    if len(cleaned) < 3:
        return False
    words = cleaned.split()
    if not words or len(words) > 4:
        return False
    first = words[0].strip(".,;:!?—–-")
    if not first or not first[0].isupper():
        return False
    if first.casefold() in _NON_ROLE_WORDS:
        return False
    return not _RE_ACTION_VERB.match(first)


def _try_detect_emoji_role(text: str) -> str | None:
    """If text starts with a known role emoji, return its role_id."""
    for emoji, role_id in _EMOJI_TO_ROLE_ID.items():
        if text.startswith(emoji):
            return role_id
    return None


def _find_role_in_text(text: str, registry: RoleRegistry) -> Role | None:
    """Search for any known role name or alias inside text."""
    text_lower = text.casefold()
    best: Role | None = None
    best_len = 0
    for role in registry.all_roles():
        for name_form in (role.name_ru, *role.bot_aliases):
            needle = name_form.casefold()
            if len(needle) >= 3 and needle in text_lower and len(needle) > best_len:
                best = role
                best_len = len(needle)
    return best


# ---------------------------------------------------------------------------
# Individual parsers (one per kind)
# ---------------------------------------------------------------------------


def _parse_end_game(text: str) -> BotMessageParse | None:
    """Parse the bot's final report.

    Ground truth for the fine-tune dataset: ``is_winner``, ``final_role`` and
    ``survived`` all come from this message, so it must never be confused with
    an ``ALIVE_LIST`` (both wrap numbered ``"N. name"`` blocks).

    Role names are kept exactly as the bot wrote them (``name_ru``) with the
    leading emoji removed; resolving them to ``role_id`` belongs to
    :meth:`domain.game_state.GameState.apply`. A report without any parseable
    entry still yields ``kind=END_GAME`` but no data — the resolver then drops
    it with a warning instead of silently inventing players.

    Args:
        text: message text as received (already stripped of outer whitespace).

    Returns:
        ``BotMessageParse`` for a final report, or ``None`` when this is not one.
    """
    if _END_GAME_RE.match(text) is None:
        return None

    winners: list[tuple[str, str]] = []
    losers: list[tuple[str, str]] = []
    for section_match in _SECTION_RE.finditer(text):
        section_name = section_match.group(1)
        block = section_match.group(2)
        entries: list[tuple[str, str]] = []
        for entry_match in _ENTRY_RE.finditer(block):
            name = _strip_leading_emoji(entry_match.group(1).strip()).strip()
            role_clean = _strip_leading_emoji(entry_match.group(2).strip()).strip()
            if name and role_clean:
                entries.append((name, role_clean))
        if "Победители" in section_name:
            winners.extend(entries)
        else:
            losers.extend(entries)

    duration: int | None = None
    duration_match = _DURATION_RE.search(text)
    if duration_match is not None:
        try:
            duration = int(duration_match.group(1))
        except (ValueError, OverflowError):  # pragma: no cover - regex is \d+
            duration = None

    return BotMessageParse(
        kind=BotMessageKind.END_GAME,
        confidence=1.0,
        winners=winners or None,
        losers=losers or None,
        duration_minutes=duration,
        raw_text=text,
    )


def _parse_registration(text: str) -> BotMessageParse | None:
    if _RE_REGISTRATION.search(text):
        return BotMessageParse(kind=BotMessageKind.REGISTRATION, raw_text=text)
    return None


def _parse_alive_list(text: str) -> BotMessageParse | None:
    """Parse the bot's ``Живые игроки … Из них … Всего: N`` snapshot.

    Names stay as ``display_name`` strings (no id resolution here — A2-1 is
    applied later by :meth:`domain.game_state.GameState.apply`). ``role_count``
    is keyed by the Russian role names exactly as the bot wrote them; mapping
    them onto ``RoleRegistry`` entries belongs to the analyzer (step 10c-fix-9).
    """
    first_line = _first_non_empty_line(text)
    if not first_line or not _RE_ALIVE_LIST.match(first_line):
        return None

    # Section "Живые игроки:" — "N. name" lines up to "Из них:"/"Всего:".
    alive_names: list[str] = []
    head_match = re.search(
        r"Живые\s+игроки:?\s*\n(.*?)(?=\n\s*(?:Из\s+них:|Всего:)|$)",
        text,
        re.DOTALL,
    )
    if head_match:
        for match in _ALIVE_LINE_RE.finditer(head_match.group(1)):
            name = match.group(1).strip()
            if name and name not in alive_names:
                alive_names.append(name)

    # Section "Из них:" — role composition. A trailing " - N" is the count;
    # without it the role occurs once. Duplicate names are accumulated.
    role_count: dict[str, int] = {}
    roles_match = re.search(
        r"Из\s+них:\s*(.+?)(?=\n\s*Всего:|\Z)",
        text,
        re.DOTALL,
    )
    if roles_match:
        for chunk in roles_match.group(1).split(","):
            # Collapse line breaks introduced by the bot's soft wrapping.
            chunk = " ".join(chunk.split())
            if not chunk:
                continue
            # Strip the leading emoji (incl. ZWJ sequences and skin tones).
            chunk_clean = _strip_leading_emoji(chunk)
            parts = re.split(r"\s*-\s*", chunk_clean, maxsplit=1)
            name = parts[0].strip()
            count = 1
            if len(parts) == 2:
                try:
                    count = int(parts[1].strip())
                except ValueError:
                    count = 1
            if name:
                role_count[name] = role_count.get(name, 0) + count

    # Section "Всего: N" — bot-reported headcount of alive players.
    total: int | None = None
    total_match = re.search(r"Всего:\s*(\d+)", text)
    if total_match:
        total = int(total_match.group(1))

    return BotMessageParse(
        kind=BotMessageKind.ALIVE_LIST,
        confidence=1.0,
        alive_player_names=alive_names,
        role_count=role_count or None,
        total_players=total,
        raw_text=text,
    )


def _extract_death_name(text: str, cause: str) -> str | None:
    if cause == "lynch":
        patterns = (_RE_NAME_BEFORE_SUICIDE, _RE_NAME_AFTER_HANG)
    else:
        patterns = (_RE_NAME_BEFORE_KILLED, _RE_NAME_AFTER_KILL)

    for pattern in patterns:
        for match in pattern.finditer(text):
            name = _clean_name(match.group("name"))
            if name is not None:
                return name
    return None


def _parse_death(text: str) -> BotMessageParse | None:
    # Guard: "X использовал(а) защиту от линчевания!" is a protection
    # notification, NOT a death (checked before any death pattern).
    if _RE_PROTECTION.search(text):
        return None

    cause: str | None = None
    if _RE_DEATH_LYNCH.search(text):
        cause = "lynch"
    elif _RE_DEATH_NIGHT.search(text):
        cause = "night_kill"

    if cause is None:
        return None

    return BotMessageParse(
        kind=BotMessageKind.DEATH,
        death_cause=cause,
        player_name=_extract_death_name(text, cause),
        raw_text=text,
    )


def _clean_username(raw: str) -> str | None:
    value = raw.strip()
    if "|" in value:
        parts = [part.strip() for part in value.split("|")]
        if len(parts) >= 2:
            value = parts[len(parts) // 2]
    decorations = " 💎💋🦱🕵️🕵🏻🕵🏼🤵🏻🤵🏼"
    value = value.strip()
    while value and value[0] in decorations:
        value = value[1:].lstrip()
    while value and value[-1] in decorations:
        value = value[:-1].rstrip()
    if (
        not value
        or value == "."
        or not any(ch.isalnum() or 0x1F1E6 <= ord(ch) <= 0x1F1FF for ch in value)
    ):
        return None
    return value


def _parse_last_words(text: str) -> BotMessageParse | None:
    match = _RE_LAST_WORDS.match(text)
    if match is None:
        return None
    words = (match.group("words") or "").strip()
    return BotMessageParse(
        kind=BotMessageKind.LAST_WORDS,
        player_name=match.group("name").strip(),
        last_words=words or None,
        raw_text=text,
    )


def _parse_gift(text: str) -> BotMessageParse | None:
    match = _RE_GIFT.match(text)
    if match is None:
        return None
    return BotMessageParse(
        kind=BotMessageKind.GIFT,
        player_name=_clean_username(match.group("giver")),
        target_name=_clean_username(match.group("target")),
        raw_text=text,
    )


def _clean_vote_target(raw: str) -> str | None:
    target = _strip_marker_prefix(raw.strip())
    # Drop trailing parenthetical noise: "Петя (5 голосов)" → "Петя".
    target = re.sub(r"\s*\([^)]*\)\s*$", "", target).strip()
    return _clean_name(target)


def _parse_vote(text: str) -> BotMessageParse | None:
    votes: list[tuple[str, str | None]] = []
    for match in _VOTE_RE.finditer(text):
        voter = _clean_username(match.group(1))
        target = _clean_username(match.group(2))
        if voter is not None:
            votes.append((voter, target))
    # Preserve support for older vote phrasing ("против", "проголосовал").
    if not votes:
        for line in text.splitlines():
            against_match = _RE_VOTE_AGAINST.search(line)
            if against_match is None:
                continue
            voter = _clean_username(against_match.group("voter"))
            target = _clean_username(against_match.group("target"))
            if voter is not None:
                votes.append((voter, target))

    if votes:
        return BotMessageParse(
            kind=BotMessageKind.VOTE,
            confidence=1.0,
            voter_name=votes[0][0],
            target_name=votes[0][1],
            votes=votes,
            raw_text=text,
        )

    if _RE_VOTE_RESULTS.search(text):
        return BotMessageParse(kind=BotMessageKind.VOTE, confidence=0.9, raw_text=text)

    return None


def _parse_protection_used(text: str) -> BotMessageParse | None:
    if _RE_PROTECTION.search(text):
        return BotMessageParse(kind=BotMessageKind.PROTECTION_USED, raw_text=text)
    return None


def _parse_vote_abstain(text: str) -> BotMessageParse | None:
    match = _RE_VOTE_ABSTAIN.search(text)
    if match is None:
        return None
    raw_name = match.group("name")
    return BotMessageParse(
        kind=BotMessageKind.VOTE_ABSTAIN,
        player_name=_clean_name(raw_name) if raw_name else None,
        raw_text=text,
    )


def _parse_role_reveal(text: str, registry: RoleRegistry) -> BotMessageParse | None:
    """Grammatical player reveals only (``"X был Роль"`` / ``"X — Роль"``).

    Action notifications ("Дон выбрал жертву") are explicitly rejected —
    they are handled by :func:`_parse_role_action` and must never poison
    ``GameState`` with fake reveals.
    """
    if _RE_ACTION_VERB.search(text):
        return None

    # (a) "Вася был Мафией" / "🔪 user_9 был Маньяком"
    for byl_match in _RE_REVEAL_BYL.finditer(text):
        role = _try_role_by_name(byl_match.group("role"), registry)
        player = _clean_name(byl_match.group("name"))
        if role is not None:
            return BotMessageParse(
                kind=BotMessageKind.ROLE_REVEAL,
                confidence=1.0,
                role_id=role.id,
                player_name=player,
                raw_text=text,
            )
        if player is not None and _looks_like_role_word(byl_match.group("role")):
            return BotMessageParse(
                kind=BotMessageKind.ROLE_REVEAL,
                confidence=0.5,
                role_id=None,
                player_name=player,
                raw_text=text,
            )

    # (b) "Вася — 🕵🏼 Комиссар Каттани" / "🕵🏼 user_3 — Комиссар Каттани"
    dash_match = _RE_REVEAL_DASH.match(text)
    if dash_match:
        role = _try_role_by_name(dash_match.group("role"), registry)
        if role is not None:
            player = _clean_name(dash_match.group("name"))
            return BotMessageParse(
                kind=BotMessageKind.ROLE_REVEAL,
                confidence=1.0 if player is not None else 0.9,
                role_id=role.id,
                player_name=player,
                raw_text=text,
            )

    # (c) "🕵🏼 Комиссар Каттани Вася" (role first, name last)
    tokens = _strip_leading_emoji(text).split()
    if len(tokens) >= 2 and _looks_like_player_token(tokens[-1]):
        role = _try_role_by_name(" ".join(tokens[:-1]), registry)
        player = _clean_name(tokens[-1])
        if role is not None and player is not None:
            return BotMessageParse(
                kind=BotMessageKind.ROLE_REVEAL,
                confidence=0.8,
                role_id=role.id,
                player_name=player,
                raw_text=text,
            )

    return None


def _extract_action_target(text: str) -> str | None:
    match = _RE_ACTION_TARGET.search(text)
    if match is None:
        return None
    candidate = match.group("target")
    if not _looks_like_player_token(candidate):
        return None
    return _clean_name(candidate)


def _parse_role_action(text: str, registry: RoleRegistry) -> BotMessageParse | None:
    """Role action notifications: "Роль сделала X" (NOT a player reveal)."""
    emoji_role_id = _try_detect_emoji_role(text)
    role_id: str | None = emoji_role_id
    confidence = 0.9

    if role_id is None:
        found = _find_role_in_text(text, registry)
        if found is None or _RE_ACTION_VERB.search(text) is None:
            return None
        role_id = found.id
        confidence = 0.7

    return BotMessageParse(
        kind=BotMessageKind.ROLE_ACTION,
        confidence=confidence,
        role_id=role_id,
        target_name=_extract_action_target(text),
        raw_text=text,
    )


def _parse_vote_phase(text: str) -> BotMessageParse | None:
    """Vote-start / vote-end announcements (``"Голосование окончено"``).

    These are unambiguous, so there is no length guard here.
    """
    if not (_RE_VOTE_PHASE.search(text) or _RE_VOTE_PHASE_STANDALONE.match(text)):
        return None
    return BotMessageParse(
        kind=BotMessageKind.PHASE_CHANGE,
        to_phase=Phase.VOTE,
        day_number=_extract_day_number(text),
        raw_text=text,
    )


def _parse_phase_change(text: str) -> BotMessageParse | None:
    """Night / day / end transitions.

    Night and day markers only match short messages, or messages that
    *start* with the phase emoji.  A long ``"🌙 Кто-то был убит …"`` is a
    DEATH (caught earlier) and must never flip the phase.
    """
    starts_with_phase_emoji = text.startswith(("🌙", "🌝"))
    if (
        len(text) >= _PHASE_MAX_LEN
        and not starts_with_phase_emoji
        and not _RE_GAME_END.search(text)
    ):
        return None

    to_phase: Phase | None = None
    if _RE_NIGHT_START.search(text):
        to_phase = Phase.NIGHT
    elif _RE_DAY_START.search(text):
        to_phase = Phase.DAY
    elif _RE_GAME_END.search(text):
        to_phase = Phase.END

    if to_phase is None:
        return None

    return BotMessageParse(
        kind=BotMessageKind.PHASE_CHANGE,
        to_phase=to_phase,
        day_number=_extract_day_number(text),
        raw_text=text,
    )


def _parse_system(text: str) -> BotMessageParse | None:
    if _RE_SYSTEM.search(text):
        return BotMessageParse(kind=BotMessageKind.SYSTEM, raw_text=text)
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
    # 1. Guard: empty / None
    if text is None:
        return None
    try:
        if not isinstance(text, str):
            return None
        stripped = text.strip()
        if not stripped:
            return None
    except (TypeError, ValueError, AttributeError, KeyError):  # defensive input guard
        return None

    # 2. Final report FIRST: it wraps its own numbered sections ("Другие
    #    пользователи:") and would otherwise be mistaken for an alive list.
    result = _parse_end_game(stripped)
    if result is not None:
        return result

    # 3. Last words: arbitrary multi-line payload; classify before other patterns.
    result = _parse_last_words(stripped)
    if result is not None:
        return result

    # 4. Alive list: "Живые игроки:" wraps a multi-line body that
    #    contains role emoji (would otherwise look like role reveals).
    result = _parse_alive_list(stripped)
    if result is not None:
        return result

    # 5. Registration
    result = _parse_registration(stripped)
    if result is not None:
        return result

    # 6. Death — BEFORE phase change ("🌙 Кто-то был убит" is a death)
    result = _parse_death(stripped)
    if result is not None:
        return result

    # 5. Vote start / vote end announcements
    result = _parse_vote_phase(stripped)
    if result is not None:
        return result

    # 6. Votes (multi-line supported)
    result = _parse_vote(stripped)
    if result is not None:
        return result

    # 7. Protection notifications ("использовал защиту от линчевания!")
    result = _parse_protection_used(stripped)
    if result is not None:
        return result

    # 8. Abstentions ("решил никого не вешать")
    result = _parse_vote_abstain(stripped)
    if result is not None:
        return result

    # 9. Gifts
    result = _parse_gift(stripped)
    if result is not None:
        return result

    # 10. Grammatical role reveals
    result = _parse_role_reveal(stripped, registry)
    if result is not None:
        return result

    # 10. Role action notifications (previously 1000 false ROLE_REVEALs)
    result = _parse_role_action(stripped, registry)
    if result is not None:
        return result

    # 11. Night / day transitions (short messages only)
    result = _parse_phase_change(stripped)
    if result is not None:
        return result

    # 12. System announcements
    result = _parse_system(stripped)
    if result is not None:
        return result

    # 13. Unknown — bot message that we couldn't classify
    return BotMessageParse(
        kind=BotMessageKind.UNKNOWN,
        confidence=0.0,
        raw_text=text,
    )
