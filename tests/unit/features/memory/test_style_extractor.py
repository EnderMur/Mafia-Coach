"""Tests for speech-style extraction from game messages."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from domain.events import MessageEvent, MessagePayload
from domain.roles import RoleRegistry, load_roles
from features.memory.style_extractor import (
    APOLOGY_THRESHOLD,
    MAX_EVIDENCE_CHARS,
    MAX_EVIDENCE_ITEMS,
    TAG_AGGRESSIVE_ACCUSATION,
    TAG_AVOID_CHECK,
    TAG_FREQUENT_APOLOGY,
    TAG_PASSIVE_SILENCE,
    PlayerStyle,
    extract_styles,
)

ROLES_PATH = Path(__file__).resolve().parents[4] / "configs" / "roles.yaml"
BASE_TS = datetime(2026, 1, 1, tzinfo=UTC)
GAME_ID = 7
ALICE = 101
BORIS = 102
VERA = 103


@pytest.fixture(name="registry")
def registry_fixture() -> RoleRegistry:
    """Load the configured role definitions."""
    return load_roles(ROLES_PATH)


def message(author_id: int, text: str) -> MessageEvent:
    """Build one message event of the test game."""
    return MessageEvent(
        ts=BASE_TS,
        game_id=GAME_ID,
        payload=MessagePayload(
            author_id=author_id,
            author_name=f"player_{author_id}",
            text=text,
        ),
    )


def messages(author_id: int, *texts: str) -> list[MessageEvent]:
    """Build one message per supplied text."""
    return [message(author_id, text) for text in texts]


def tags_of(styles: dict[int, list[PlayerStyle]], player_id: int) -> set[str]:
    """Return the detected tag names of one player."""
    return {style.tag for style in styles.get(player_id, [])}


def test_two_apologies_produce_frequent_apology(registry: RoleRegistry) -> None:
    """Two apology messages cross the apology threshold."""
    events = messages(ALICE, "извините, я запутался", "простите меня за голос")

    styles = extract_styles(events, registry=registry)

    assert tags_of(styles, ALICE) == {TAG_FREQUENT_APOLOGY}
    assert styles[ALICE][0].weight == pytest.approx(0.7)


def test_two_aggressions_produce_aggressive_accusation(registry: RoleRegistry) -> None:
    """Two aggressive accusations cross the aggression threshold."""
    events = messages(BORIS, "его надо вешать сегодня", "давайте вешать Петю")

    styles = extract_styles(events, registry=registry)

    assert tags_of(styles, BORIS) == {TAG_AGGRESSIVE_ACCUSATION}
    assert styles[BORIS][0].weight == pytest.approx(0.8)


def test_single_avoid_check_produces_avoid_check(registry: RoleRegistry) -> None:
    """One refusal to check is enough for the avoid-check tag."""
    events = messages(VERA, "не буду проверять его сегодня", "голосую против Пети")

    styles = extract_styles(events, registry=registry)

    assert TAG_AVOID_CHECK in tags_of(styles, VERA)


def test_single_message_produces_passive_silence(registry: RoleRegistry) -> None:
    """A player with fewer than two messages is marked as passive."""
    events = messages(VERA, "привет, я тут")

    styles = extract_styles(events, registry=registry)

    assert tags_of(styles, VERA) == {TAG_PASSIVE_SILENCE}
    assert styles[VERA][0].weight == pytest.approx(0.4)
    assert styles[VERA][0].evidence == ["привет, я тут"]


def test_player_without_patterns_is_absent_from_result(registry: RoleRegistry) -> None:
    """Neutral speech produces no styles and no dictionary entry."""
    events = messages(ALICE, "я подумаю до вечера", "давайте послушаем остальных")

    styles = extract_styles(events, registry=registry)

    assert ALICE not in styles
    assert styles == {}


def test_three_players_are_grouped_by_author(registry: RoleRegistry) -> None:
    """Each author gets an independent set of styles."""
    events = [
        *messages(ALICE, "извините, ошибся", "простите, был неправ"),
        *messages(BORIS, "его надо вешать", "вешать и точка"),
        message(VERA, "я промолчу"),
    ]

    styles = extract_styles(events, registry=registry)

    assert set(styles) == {ALICE, BORIS, VERA}
    assert tags_of(styles, ALICE) == {TAG_FREQUENT_APOLOGY}
    assert tags_of(styles, BORIS) == {TAG_AGGRESSIVE_ACCUSATION}
    assert tags_of(styles, VERA) == {TAG_PASSIVE_SILENCE}


def test_evidence_is_limited_and_truncated(registry: RoleRegistry) -> None:
    """Evidence keeps at most five messages, each cut to the length limit."""
    long_tail = " и это очень длинное объяснение" * 10
    events = messages(ALICE, *(f"извините, я ошибся{index}{long_tail}" for index in range(7)))

    styles = extract_styles(events, registry=registry)

    evidence = styles[ALICE][0].evidence
    assert len(evidence) == MAX_EVIDENCE_ITEMS
    assert all(len(item) <= MAX_EVIDENCE_CHARS for item in evidence)
    assert evidence[0].startswith("извините, я ошибся0")


def test_emoji_are_stripped_before_matching(registry: RoleRegistry) -> None:
    """Emoji never block a match and never reach the evidence."""
    events = messages(ALICE, "не буду проверять его 🙄🔥", "голосую против Пети")

    styles = extract_styles(events, registry=registry)

    assert tags_of(styles, ALICE) == {TAG_AVOID_CHECK}
    evidence = styles[ALICE][0].evidence
    assert evidence == ["не буду проверять его"]
    assert "🙄" not in evidence[0]


def test_empty_event_list_produces_empty_dict(registry: RoleRegistry) -> None:
    """No events means no styles."""
    assert extract_styles([], registry=registry) == {}


@pytest.mark.parametrize("text", ["Извини, я ошибся", "ИЗВИНИ, я ошибся"])
def test_apology_matching_is_case_insensitive(registry: RoleRegistry, text: str) -> None:
    """Upper- and mixed-case apologies match the same pattern."""
    events = messages(ALICE, text, "ПРОСТИТЕ, был неправ")

    styles = extract_styles(events, registry=registry)

    assert tags_of(styles, ALICE) == {TAG_FREQUENT_APOLOGY}


def test_single_apology_is_below_threshold(registry: RoleRegistry) -> None:
    """One apology is not enough for the apology tag."""
    events = messages(ALICE, "извините, я ошибся", "давайте слушать остальных")

    styles = extract_styles(events, registry=registry)

    assert APOLOGY_THRESHOLD == 2
    assert ALICE not in styles


def test_mixed_patterns_produce_two_styles(registry: RoleRegistry) -> None:
    """An apologetic player who also accuses collects both tags."""
    events = messages(
        ALICE,
        "извините, я ошибся",
        "простите, был неправ",
        "его надо вешать",
        "вешать и точка",
    )

    styles = extract_styles(events, registry=registry)

    assert tags_of(styles, ALICE) == {TAG_FREQUENT_APOLOGY, TAG_AGGRESSIVE_ACCUSATION}
    assert len(styles[ALICE]) == 2


def test_emoji_only_single_message_still_flagged_as_passive(registry: RoleRegistry) -> None:
    """A message without usable text keeps a non-empty evidence list."""
    events = messages(VERA, "🎉🎉🎉")

    styles = extract_styles(events, registry=registry)

    assert tags_of(styles, VERA) == {TAG_PASSIVE_SILENCE}
    assert styles[VERA][0].evidence == ["<нет пригодного текста>"]
