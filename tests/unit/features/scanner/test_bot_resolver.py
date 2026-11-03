"""Tests for ``features.scanner.bot_resolver`` — parsed bot message → domain events.

Pure unit tests: no ``async``, no database, no Telegram.  The roles
registry is loaded from the real ``configs/roles.yaml`` (the single source
of truth) so ``role_name_ru`` assertions stay honest.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from loguru import logger

from domain.events import (
    AliveListEvent,
    AliveListPayload,
    AnyEvent,
    DeathEvent,
    EndGameEvent,
    EndGamePayload,
    MessageEvent,
    PhaseChangeEvent,
    RoleRevealEvent,
    VoteEvent,
)
from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles
from features.parser.bot_messages import BotMessageKind, BotMessageParse
from features.scanner.bot_resolver import _strip_role_prefix, resolve_bot_message

if TYPE_CHECKING:
    from loguru import Record

REGISTRY: RoleRegistry = load_roles(Path(__file__).parents[4] / "configs" / "roles.yaml")
TS: datetime = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)
GAME_ID = 1

NAME_TO_ID: dict[str, int] = {
    "Вася": 100,
    "Петя": 101,
    "Лена": 102,
    "Игорь": 103,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(
    parse: BotMessageParse,
    name_to_id: dict[str, int] | None = None,
    current_phase: Phase = Phase.NIGHT,
) -> list[AnyEvent]:
    """Call ``resolve_bot_message`` with stable test defaults."""
    return resolve_bot_message(
        parse=parse,
        name_to_id=name_to_id if name_to_id is not None else {},
        registry=REGISTRY,
        game_id=GAME_ID,
        ts=TS,
        current_phase=current_phase,
    )


def _parse(kind: BotMessageKind, **kwargs: Any) -> BotMessageParse:
    """Build a ``BotMessageParse`` with test defaults filled in."""
    return BotMessageParse(kind=kind, confidence=kwargs.pop("confidence", 1.0), **kwargs)


@pytest.fixture
def log_records() -> Iterator[list[Record]]:
    """Capture WARNING-and-above loguru records emitted during a test."""
    records: list[Record] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def _messages(records: list[Record]) -> list[str]:
    """Return only the formatted message texts of captured records."""
    return [record["message"] for record in records]


# ===================================================================
# PHASE_CHANGE (1-3)
# ===================================================================


def test_phase_change_builds_event_with_current_phase_as_from_phase() -> None:
    """1. A real transition yields one event whose ``from_phase`` is the current phase."""
    parse = _parse(BotMessageKind.PHASE_CHANGE, to_phase=Phase.NIGHT, day_number=1)

    events = _resolve(parse, current_phase=Phase.REGISTRATION)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PhaseChangeEvent)
    assert event.kind == "phase_change"
    assert event.payload.from_phase is Phase.REGISTRATION
    assert event.payload.to_phase is Phase.NIGHT
    assert event.payload.day_number == 1
    assert event.ts == TS
    assert event.game_id == GAME_ID


def test_game_start_creates_registration_phase_change() -> None:
    """GAME_START emits a registration event even when already in registration."""
    events = _resolve(_parse(BotMessageKind.GAME_START), current_phase=Phase.REGISTRATION)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PhaseChangeEvent)
    assert event.payload.from_phase is Phase.REGISTRATION
    assert event.payload.to_phase is Phase.REGISTRATION
    assert event.payload.day_number == 0
    assert event.ts == TS
    assert event.game_id == GAME_ID


def test_phase_change_to_current_phase_is_skipped_with_warning(
    log_records: list[Record],
) -> None:
    """2. A transition to the phase we are already in is a no-op (GameState would reject it)."""
    parse = _parse(BotMessageKind.PHASE_CHANGE, to_phase=Phase.NIGHT, day_number=1)

    events = _resolve(parse, current_phase=Phase.NIGHT)

    assert events == []
    assert any("phase_change to current phase" in text for text in _messages(log_records))


def test_phase_change_without_to_phase_is_skipped_with_warning(log_records: list[Record]) -> None:
    """3. A phase announcement without a target phase cannot become an event."""
    parse = _parse(BotMessageKind.PHASE_CHANGE, to_phase=None, day_number=2)

    events = _resolve(parse, current_phase=Phase.DAY)

    assert events == []
    assert any("phase_change without to_phase" in text for text in _messages(log_records))


def test_phase_change_without_day_number_defaults_to_zero() -> None:
    """3b. A missing day number degrades to 0 instead of failing validation."""
    parse = _parse(BotMessageKind.PHASE_CHANGE, to_phase=Phase.DAY, day_number=None)

    events = _resolve(parse, current_phase=Phase.NIGHT)

    assert len(events) == 1
    assert isinstance(events[0], PhaseChangeEvent)
    assert events[0].payload.day_number == 0


# ===================================================================
# DEATH (4-6)
# ===================================================================


def test_death_resolves_victim_and_keeps_parsed_cause() -> None:
    """4. A known victim becomes a DeathEvent carrying the parsed cause."""
    parse = _parse(BotMessageKind.DEATH, player_name="Вася", death_cause="lynch")

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID), current_phase=Phase.DAY)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DeathEvent)
    assert event.kind == "death"
    assert event.payload.victim_id == 100
    assert event.payload.victim_name == "Вася"
    assert event.payload.cause == "lynch"


def test_death_with_night_kill_cause() -> None:
    """4b. ``night_kill`` is preserved as-is."""
    parse = _parse(BotMessageKind.DEATH, player_name="Лена", death_cause="night_kill")

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert len(events) == 1
    assert isinstance(events[0], DeathEvent)
    assert events[0].payload.cause == "night_kill"
    assert events[0].payload.victim_id == 102


def test_death_with_unresolved_victim_is_skipped_with_warning(log_records: list[Record]) -> None:
    """5. An unknown nickname produces no event, only a warning."""
    parse = _parse(BotMessageKind.DEATH, player_name="Неизвестный", death_cause="lynch")

    events = _resolve(parse, name_to_id={}, current_phase=Phase.DAY)

    assert events == []
    messages = _messages(log_records)
    assert any("Неизвестный" in text for text in messages)
    assert any("death" in text for text in messages)


def test_death_without_player_name_is_skipped_with_warning(log_records: list[Record]) -> None:
    """5b. A death announcement without a name cannot be attributed."""
    parse = _parse(BotMessageKind.DEATH, player_name=None, death_cause="lynch")

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert events == []
    assert any("death without player_name" in text for text in _messages(log_records))


def test_death_without_cause_falls_back_to_unknown() -> None:
    """6. ``death_cause=None`` degrades to the ``unknown`` literal."""
    parse = _parse(BotMessageKind.DEATH, player_name="Петя", death_cause=None)

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert len(events) == 1
    assert isinstance(events[0], DeathEvent)
    assert events[0].payload.cause == "unknown"


def test_death_with_blank_cause_falls_back_to_unknown() -> None:
    """6b. An empty cause string behaves like a missing one."""
    parse = _parse(BotMessageKind.DEATH, player_name="Игорь", death_cause="")

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert len(events) == 1
    assert isinstance(events[0], DeathEvent)
    assert events[0].payload.cause == "unknown"


def test_death_with_unrecognized_cause_degrades_to_unknown(log_records: list[Record]) -> None:
    """6c. A cause outside the domain literal never raises; it is logged and degraded."""
    parse = _parse(BotMessageKind.DEATH, player_name="Вася", death_cause="застрелен")

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DeathEvent)
    assert event.payload.cause == "unknown"
    messages = _messages(log_records)
    assert any("unrecognized cause" in text and "застрелен" in text for text in messages)


# ===================================================================
# ROLE_REVEAL (7-10)
# ===================================================================


def test_role_reveal_uses_registry_russian_name() -> None:
    """7. The payload carries ``name_ru`` from the registry, not the bot's raw text."""
    parse = _parse(BotMessageKind.ROLE_REVEAL, player_name="Вася", role_id="mafia")

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID), current_phase=Phase.DAY)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, RoleRevealEvent)
    assert event.kind == "role_reveal"
    assert event.payload.player_id == 100
    assert event.payload.player_name == "Вася"
    assert event.payload.role_name_ru == REGISTRY.get("mafia").name_ru
    assert event.payload.role_name_ru == "Мафия"


def test_role_reveal_without_role_id_is_skipped_with_warning(log_records: list[Record]) -> None:
    """8. A low-confidence reveal (``role_id=None``) is not worth an event."""
    parse = _parse(BotMessageKind.ROLE_REVEAL, confidence=0.5, player_name="Вася", role_id=None)

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert events == []
    assert any("role_reveal without role_id" in text for text in _messages(log_records))


def test_role_reveal_without_player_name_is_skipped_with_warning(log_records: list[Record]) -> None:
    """8b. A reveal without a player name is skipped as well."""
    parse = _parse(BotMessageKind.ROLE_REVEAL, player_name=None, role_id="mafia")

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert events == []
    assert any("role_reveal without role_id" in text for text in _messages(log_records))


def test_role_reveal_with_unresolved_player_is_skipped_with_warning(
    log_records: list[Record],
) -> None:
    """9. A known role but an unknown nickname yields no event."""
    parse = _parse(BotMessageKind.ROLE_REVEAL, player_name="Призрак", role_id="mafia")

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert events == []
    messages = _messages(log_records)
    assert any("role_reveal" in text and "Призрак" in text for text in messages)


def test_role_reveal_with_unknown_role_id_is_skipped_with_warning(
    log_records: list[Record],
) -> None:
    """10. ``UnknownRoleError`` from the registry is caught, not propagated."""
    parse = _parse(BotMessageKind.ROLE_REVEAL, player_name="Вася", role_id="not_a_role")

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert events == []
    messages = _messages(log_records)
    assert any("unknown role id" in text and "not_a_role" in text for text in messages)


# ===================================================================
# VOTE (11-14)
# ===================================================================


def test_vote_with_single_pair_builds_one_event() -> None:
    """11. One vote pair → one VoteEvent with both ids resolved."""
    parse = _parse(
        BotMessageKind.VOTE,
        voter_name="Вася",
        target_name="Петя",
        votes=[("Вася", "Петя")],
    )

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID), current_phase=Phase.VOTE)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, VoteEvent)
    assert event.kind == "vote"
    assert event.payload.voter_id == 100
    assert event.payload.voter_name == "Вася"
    assert event.payload.target_id == 101
    assert event.payload.target_name == "Петя"


def test_vote_with_three_pairs_builds_three_events() -> None:
    """12. A multi-line vote message yields one event per pair, in order."""
    parse = _parse(
        BotMessageKind.VOTE,
        votes=[("Вася", "Петя"), ("Лена", "Игорь"), ("Игорь", "Вася")],
    )

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID), current_phase=Phase.VOTE)

    assert len(events) == 3
    assert all(isinstance(event, VoteEvent) for event in events)
    voters = [event.payload.voter_id for event in events if isinstance(event, VoteEvent)]
    targets = [event.payload.target_id for event in events if isinstance(event, VoteEvent)]
    assert voters == [100, 102, 103]
    assert targets == [101, 103, 100]


def test_vote_abstention_keeps_target_none() -> None:
    """13. An abstention is still a vote, with ``target_id=None``."""
    parse = _parse(BotMessageKind.VOTE, voter_name="Вася", votes=[("Вася", None)])

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID), current_phase=Phase.VOTE)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, VoteEvent)
    assert event.payload.voter_id == 100
    assert event.payload.target_id is None
    assert event.payload.target_name is None


def test_vote_with_unknown_voter_skips_only_that_pair(log_records: list[Record]) -> None:
    """14. One unresolved voter does not drop the remaining pairs."""
    parse = _parse(
        BotMessageKind.VOTE,
        votes=[("Неизвестный", "Петя"), ("Вася", "Лена"), ("Игорь", "Петя")],
    )

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID), current_phase=Phase.VOTE)

    assert len(events) == 2
    assert all(isinstance(event, VoteEvent) for event in events)
    first, second = events
    assert isinstance(first, VoteEvent)
    assert isinstance(second, VoteEvent)
    assert first.payload.voter_id == 100
    assert second.payload.voter_id == 103
    assert any("Неизвестный" in text for text in _messages(log_records))


def test_vote_with_unresolved_target_keeps_target_name() -> None:
    """14b. Partial information is preserved: name without id."""
    parse = _parse(BotMessageKind.VOTE, votes=[("Вася", "Призрак")])

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID), current_phase=Phase.VOTE)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, VoteEvent)
    assert event.payload.target_id is None
    assert event.payload.target_name == "Призрак"


def test_vote_falls_back_to_single_pair_when_votes_missing() -> None:
    """14c. ``votes=None`` with ``voter_name`` set still produces one event."""
    parse = _parse(BotMessageKind.VOTE, voter_name="Лена", target_name="Игорь", votes=None)

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID), current_phase=Phase.VOTE)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, VoteEvent)
    assert event.payload.voter_id == 102
    assert event.payload.target_id == 103


def test_vote_without_any_voter_produces_no_events() -> None:
    """14d. A vote-results announcement without names produces nothing."""
    parse = _parse(BotMessageKind.VOTE, confidence=0.9, voter_name=None, votes=None)

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID), current_phase=Phase.VOTE)

    assert events == []


# ===================================================================
# LAST_WORDS (15-17)
# ===================================================================


def test_last_words_become_message_from_the_victim() -> None:
    """15. Last words are attributed to the dying player as a MessageEvent."""
    parse = _parse(
        BotMessageKind.LAST_WORDS,
        player_name="Вася",
        last_words="Мафия это Лена",
    )

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID), current_phase=Phase.NIGHT)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, MessageEvent)
    assert event.kind == "message"
    assert event.payload.author_id == 100
    assert event.payload.author_name == "Вася"
    assert event.payload.text == "Мафия это Лена"


def test_last_words_without_player_name_are_skipped_with_warning(log_records: list[Record]) -> None:
    """16. No author → no event."""
    parse = _parse(BotMessageKind.LAST_WORDS, player_name=None, last_words="Мафия это Лена")

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert events == []
    assert any("last_words without player_name" in text for text in _messages(log_records))


def test_last_words_with_unresolved_player_are_skipped_with_warning(
    log_records: list[Record],
) -> None:
    """16b. An unknown author → no event, warning only."""
    parse = _parse(BotMessageKind.LAST_WORDS, player_name="Призрак", last_words="прощайте")

    events = _resolve(parse, name_to_id={}, current_phase=Phase.NIGHT)

    assert events == []
    messages = _messages(log_records)
    assert any("last_words" in text and "Призрак" in text for text in messages)


@pytest.mark.parametrize("last_words", ["", "   ", None])
def test_last_words_with_empty_text_use_placeholder(last_words: str | None) -> None:
    """17. Empty/blank/absent words degrade to the ``(no words)`` placeholder."""
    parse = _parse(BotMessageKind.LAST_WORDS, player_name="Вася", last_words=last_words)

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, MessageEvent)
    assert event.payload.text == "(no words)"


def test_last_words_text_is_stripped() -> None:
    """17b. Surrounding whitespace never reaches the payload."""
    parse = _parse(BotMessageKind.LAST_WORDS, player_name="Вася", last_words="  я мирный  \n")

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID))

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, MessageEvent)
    assert event.payload.text == "я мирный"


# ===================================================================
# Step 10c-fix-8: ALIVE_LIST (21-23)
# ===================================================================


def test_alive_list_creates_event() -> None:
    """21. A snapshot with names becomes one AliveListEvent (names unresolved)."""
    parse = _parse(
        BotMessageKind.ALIVE_LIST,
        alive_player_names=["A"],
        role_count={"Мафия": 5},
        total_players=10,
    )

    events = _resolve(parse, name_to_id={})

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, AliveListEvent)
    assert isinstance(event.payload, AliveListPayload)
    assert event.kind == "alive_list"
    assert event.game_id == GAME_ID
    assert event.ts == TS
    assert event.payload.alive_player_names == ["A"]
    assert event.payload.role_count == {"Мафия": 5}
    assert event.payload.total_players == 10


def test_alive_list_empty_names_skipped(log_records: list[Record]) -> None:
    """22. An empty name list is meaningless: no event, only a warning."""
    parse = _parse(BotMessageKind.ALIVE_LIST, alive_player_names=[])

    assert _resolve(parse) == []
    assert any("alive_list without player names" in text for text in _messages(log_records))


def test_alive_list_preserves_role_count() -> None:
    """23. role_count is forwarded as-is (no registry mapping in the resolver)."""
    role_count = {"Адвокат": 1, "Мафия": 5, "Мирный Житель": 3}
    parse = _parse(
        BotMessageKind.ALIVE_LIST,
        alive_player_names=["user_1", "user_2"],
        role_count=role_count,
        total_players=9,
    )

    events = _resolve(parse)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, AliveListEvent)
    assert event.payload.role_count == role_count
    assert event.payload.alive_player_names == ["user_1", "user_2"]
    assert event.payload.total_players == 9


# ===================================================================
# Informational kinds (18-20)
# ===================================================================


def test_registration_produces_no_events() -> None:
    """18. Registration announcements are informational."""
    assert _resolve(_parse(BotMessageKind.REGISTRATION), name_to_id=dict(NAME_TO_ID)) == []


def test_role_action_produces_no_events() -> None:
    """19. Role action notifications are not player reveals."""
    parse = _parse(BotMessageKind.ROLE_ACTION, role_id="don", target_name="Вася")

    assert _resolve(parse, name_to_id=dict(NAME_TO_ID)) == []


def test_unknown_kind_produces_no_events() -> None:
    """20. An unclassified bot message produces no events."""
    parse = _parse(BotMessageKind.UNKNOWN, confidence=0.0, raw_text="что-то странное")

    assert _resolve(parse, name_to_id=dict(NAME_TO_ID)) == []


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [
        (BotMessageKind.ALIVE_LIST, {}),
        (BotMessageKind.SYSTEM, {}),
        (BotMessageKind.GIFT, {"player_name": "Вася"}),
        (BotMessageKind.PROTECTION_USED, {"player_name": "Вася"}),
        (BotMessageKind.VOTE_ABSTAIN, {"player_name": "Вася"}),
    ],
)
def test_informational_kinds_produce_no_events(
    kind: BotMessageKind, kwargs: dict[str, Any]
) -> None:
    """18-20b. Every remaining kind is informational and resolves to no events."""
    assert _resolve(_parse(kind, **kwargs), name_to_id=dict(NAME_TO_ID)) == []


# ===================================================================
# Contract-level guarantees
# ===================================================================


def test_warnings_are_bound_to_the_scanner_channel(log_records: list[Record]) -> None:
    """Skip warnings must be filterable by the ``scanner`` log channel."""
    _resolve(_parse(BotMessageKind.DEATH, player_name="Призрак"), name_to_id={})

    assert log_records
    assert all(record["extra"].get("channel") == "scanner" for record in log_records)
    assert all(record["level"].name == "WARNING" for record in log_records)


def test_resolved_events_share_game_id_and_timestamp() -> None:
    """Every event of one message carries the same game id and timestamp."""
    parse = _parse(BotMessageKind.VOTE, votes=[("Вася", "Петя"), ("Лена", "Игорь")])

    events = _resolve(parse, name_to_id=dict(NAME_TO_ID), current_phase=Phase.VOTE)

    assert len(events) == 2
    assert {event.game_id for event in events} == {GAME_ID}
    assert {event.ts for event in events} == {TS}


def test_resolver_does_not_mutate_the_name_mapping() -> None:
    """The resolver is pure: the caller's mapping stays untouched."""
    mapping = dict(NAME_TO_ID)
    parse = _parse(BotMessageKind.DEATH, player_name="Призрак", death_cause="lynch")

    _resolve(parse, name_to_id=mapping)
    _resolve(_parse(BotMessageKind.VOTE, votes=[("Вася", "Призрак")]), name_to_id=mapping)

    assert mapping == NAME_TO_ID


def test_markdown_link_in_death():
    from features.parser.bot_messages import parse_bot_message

    parsed = parse_bot_message("Сегодня повешен [Yulios](tg://user?id=1)", registry=REGISTRY)
    assert parsed is not None
    events = _resolve(parsed, {"Yulios": 1})
    assert len(events) == 1
    assert isinstance(events[0], DeathEvent)
    assert events[0].payload.victim_name == "Yulios"


def test_markdown_link_in_vote():
    parsed = _parse(
        BotMessageKind.VOTE, votes=[("[Вася](tg://user?id=100)", "[Петя](tg://user?id=101)")]
    )
    events = _resolve(parsed, NAME_TO_ID)
    assert len(events) == 1
    assert isinstance(events[0], VoteEvent)
    assert events[0].payload.voter_id == 100
    assert events[0].payload.target_id == 101
    assert events[0].payload.target_name == "Петя"
    fallback = _parse(BotMessageKind.VOTE, voter_name="**Вася**", target_name="*Петя*")
    fallback_event = _resolve(fallback, NAME_TO_ID)[0]
    assert isinstance(fallback_event, VoteEvent)
    assert fallback_event.payload.target_id == 101


def test_bold_name_in_role_reveal():
    from features.parser.bot_messages import parse_bot_message

    parsed = parse_bot_message("**Вася** был Мафией", registry=REGISTRY)
    assert parsed is not None
    events = _resolve(parsed, NAME_TO_ID)
    assert len(events) == 1
    assert isinstance(events[0], RoleRevealEvent)
    assert events[0].payload.player_name == "Вася"


def test_strip_returns_plain_when_no_markdown():
    from features.scanner.bot_resolver import _strip_markdown

    assert _strip_markdown("Вася") == "Вася"
    assert _strip_markdown("[**Вася**](tg://user?id=1)") == "Вася"
    assert _strip_markdown("*Петя*") == "Петя"


# ===================================================================
# Step 10c-fix-10: _strip_role_prefix & fallback resolution
# ===================================================================


def test_death_with_role_prefix_resolved() -> None:
    """Night-kill name carrying '[emoji] RoleName ' resolves to the bare display_name."""
    parse = BotMessageParse(
        kind=BotMessageKind.DEATH,
        confidence=1.0,
        player_name="👨🏼 Мирный Житель ℳ𝒶𝓇𝒾𝓃𝒶 ♕",
        death_cause="night_kill",
        raw_text="",
    )

    events = _resolve(parse, name_to_id={"ℳ𝒶𝓇𝒾𝓃𝒶 ♕": 123})

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DeathEvent)
    assert event.payload.victim_id == 123
    assert event.payload.victim_name == "ℳ𝒶𝓇𝒾𝓃𝒶 ♕"
    assert event.payload.cause == "night_kill"


def test_death_without_role_prefix_resolved() -> None:
    """Plain victim display_name resolves directly without prefix stripping."""
    parse = _parse(BotMessageKind.DEATH, player_name="Вася", death_cause="night_kill")

    events = _resolve(parse, name_to_id={"Вася": 100})

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DeathEvent)
    assert event.payload.victim_id == 100
    assert event.payload.victim_name == "Вася"


def test_death_unresolved_falls_back_to_warning(log_records: list[Record]) -> None:
    """Unresolved plain or role-prefixed victim logs a warning and emits no event."""
    parse_plain = _parse(BotMessageKind.DEATH, player_name="Неизвестный", death_cause="night_kill")
    parse_prefixed = _parse(
        BotMessageKind.DEATH,
        player_name="👨🏼 Мирный Житель Неизвестный",
        death_cause="night_kill",
    )

    assert _resolve(parse_plain, name_to_id={}) == []
    assert _resolve(parse_prefixed, name_to_id={}) == []
    messages = _messages(log_records)
    assert any("Неизвестный" in text for text in messages)


def test_strip_role_prefix_does_not_break_plain_names() -> None:
    """Names without a role prefix (or role-only strings) stay intact."""
    assert _strip_role_prefix("Вася", REGISTRY) == "Вася"
    assert _strip_role_prefix("Магистр", REGISTRY) == "Магистр"
    assert _strip_role_prefix("Мафия", REGISTRY) == "Мафия"
    assert _strip_role_prefix("Мафия   ", REGISTRY) == "Мафия   "


def test_strip_role_prefix_all_roles() -> None:
    """Every registered role's name_ru prefix is stripped cleanly."""
    for role in REGISTRY.all_roles():
        assert _strip_role_prefix(f"{role.name_ru} Имя", REGISTRY) == "Имя"


def test_role_prefix_resolved_in_reveal_last_words_and_votes() -> None:
    """Role prefix stripping also applies to role_reveal, last_words, and vote pairs."""
    reveal = _parse(
        BotMessageKind.ROLE_REVEAL,
        player_name="🤵🏼 Мафия Вася",
        role_id="mafia",
    )
    reveal_events = _resolve(reveal, name_to_id=dict(NAME_TO_ID))
    assert len(reveal_events) == 1
    assert isinstance(reveal_events[0], RoleRevealEvent)
    assert reveal_events[0].payload.player_id == 100
    assert reveal_events[0].payload.player_name == "Вася"

    last_words = _parse(
        BotMessageKind.LAST_WORDS,
        player_name="👨🏼 Мирный Житель Ӡɞёӡɠϰƅɩύ Λγɥ 💬",
        last_words="прощайте",
    )
    lw_events = _resolve(last_words, name_to_id={"Ӡɞёӡɠϰƅɩύ Λγɥ 💬": 555})
    assert len(lw_events) == 1
    assert isinstance(lw_events[0], MessageEvent)
    assert lw_events[0].payload.author_id == 555
    assert lw_events[0].payload.author_name == "Ӡɞёӡɠϰƅɩύ Λγɥ 💬"

    vote = _parse(
        BotMessageKind.VOTE,
        votes=[
            ("👨🏼 Мирный Житель Вася", "🤵🏼 Мафия Петя"),
            ("Лена", "🤵🏼 Мафия Призрак"),
        ],
    )
    vote_events = _resolve(vote, name_to_id=dict(NAME_TO_ID), current_phase=Phase.VOTE)
    assert len(vote_events) == 2
    first_vote, second_vote = vote_events
    assert isinstance(first_vote, VoteEvent)
    assert first_vote.payload.voter_id == 100
    assert first_vote.payload.voter_name == "Вася"
    assert first_vote.payload.target_id == 101
    assert first_vote.payload.target_name == "Петя"
    assert isinstance(second_vote, VoteEvent)
    assert second_vote.payload.voter_id == 102
    assert second_vote.payload.target_id is None
    assert second_vote.payload.target_name == "Призрак"


# ===================================================================
# Step 10c-fix-11: END_GAME (24-25)
# ===================================================================


def test_end_game_creates_event() -> None:
    """24. A final report becomes one EndGameEvent carrying winners and losers."""
    parse = _parse(
        BotMessageKind.END_GAME,
        winners=[("user_01", "Камикадзе"), ("user_07", "Мафия")],
        losers=[("user_08", "Маньяк")],
        duration_minutes=20,
    )

    events = _resolve(parse, name_to_id={})

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, EndGameEvent)
    assert isinstance(event.payload, EndGamePayload)
    assert event.kind == "end_game"
    assert event.game_id == GAME_ID
    assert event.ts == TS
    assert event.payload.winners == [("user_01", "Камикадзе"), ("user_07", "Мафия")]
    assert event.payload.losers == [("user_08", "Маньяк")]
    assert event.payload.duration_minutes == 20


def test_end_game_empty_skipped(log_records: list[Record]) -> None:
    """25. A report without any players is dropped with a warning."""
    parse = _parse(BotMessageKind.END_GAME, winners=[], losers=[])

    assert _resolve(parse) == []
    assert any("end_game without any players" in text for text in _messages(log_records))


def test_strip_role_prefix_strips_parens() -> None:
    assert _strip_role_prefix("Мирный Житель ^‿^", REGISTRY) == "^‿^"
    assert _strip_role_prefix("( ^‿^ )", REGISTRY) == "^‿^"
    assert _strip_role_prefix("Мирный Житель ( ^‿^ )", REGISTRY) == "^‿^"
    assert _strip_role_prefix("Мирный Житель ( ^‿^", REGISTRY) == "^‿^"
    assert _strip_role_prefix("Мирный Житель ^‿^ )", REGISTRY) == "^‿^"


def test_strip_role_prefix_plain_name_unchanged() -> None:
    assert _strip_role_prefix("Вася", REGISTRY) == "Вася"
    assert _strip_role_prefix("Вася Петров", REGISTRY) == "Вася Петров"


def test_death_resolves_decorated_nickname() -> None:
    """A club-marker prefix in a bot death message still finds the Telegram name."""
    parse = _parse(
        BotMessageKind.DEATH,
        player_name="| Дмитрий",
        death_cause="lynch",
    )

    events = _resolve(
        parse,
        name_to_id={"Дмитрий": 1, "Shinichiro": 2},
        current_phase=Phase.DAY,
    )

    assert len(events) == 1
    assert isinstance(events[0], DeathEvent)
    assert events[0].payload.victim_id == 1
    assert events[0].payload.victim_name == "Дмитрий"
