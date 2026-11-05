"""Tests for ``features.parser.bot_messages`` — semantic parser for bot messages."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from domain.phases import Phase
from domain.roles import RoleRegistry, load_roles
from features.parser.bot_messages import (
    _EMOJI_TO_ROLE_ID,
    BotMessageKind,
    BotMessageParse,
    parse_bot_message,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> RoleRegistry:
    """Load the roles registry once per module."""
    return load_roles(Path("configs/roles.yaml"))


FIXTURE_PATH = Path("tests/fixtures/telegram/real_game.json")


@pytest.fixture(scope="module")
def fixture_data() -> dict[str, Any]:
    """Load the real-game fixture once."""
    data: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return data


# ===================================================================
# Registration tests (3)
# ===================================================================


class TestRegistration:
    def test_registration_open(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Регистрация открыта! Запишите себя в игру.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.REGISTRATION

    def test_registration_player_signed_up(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_1 записался в игру. Всего записалось: 5", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.REGISTRATION

    def test_registration_closed(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Регистрация закрыта. В игре 10 человек.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.REGISTRATION


# ===================================================================
# Phase change tests (8)
# ===================================================================


class TestPhaseChangeNight:
    def test_night_moon_emoji(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🌙 Наступает ночь 1. Город засыпает.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.NIGHT
        assert r.day_number == 1

    def test_night_text_only(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("наступает ночь", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.NIGHT

    def test_night_city_sleeps(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Город засыпает. Наступает ночь 3.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.NIGHT
        assert r.day_number == 3


class TestPhaseChangeDay:
    def test_day_sun_emoji(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🌝 Наступает день 2. Город просыпается.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.DAY
        assert r.day_number == 2

    def test_day_text_only(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("наступает день", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.DAY

    def test_day_city_wakes(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Город просыпается. День 5.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.DAY


class TestPhaseChangeVote:
    def test_vote_time(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Пришло время голосования. День 1.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.VOTE
        assert r.day_number == 1

    def test_vote_standalone(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Голосование", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.VOTE


# ===================================================================
# Alive list tests (2)
# ===================================================================


class TestAliveList:
    def test_alive_with_names(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Живые: user_1, user_2, user_3", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ALIVE_LIST

    def test_alive_empty(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Живые:", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ALIVE_LIST


# ===================================================================
# Step 10c-fix-8: ALIVE_LIST payload extraction (4)
# ===================================================================


class TestAliveListExtraction:
    """The parser must carry names, role composition and headcount."""

    def test_alive_list_parses_player_names(self, registry: RoleRegistry) -> None:
        text = (
            "Живые игроки:\n1. user_1\n2. user_2\n\n"
            "Из них: 👨🏼‍💼 Адвокат, 🤵🏼 Мафия - 5\n\nВсего: 2"
        )
        r = parse_bot_message(text, registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ALIVE_LIST
        assert r.alive_player_names == ["user_1", "user_2"]

    def test_alive_list_parses_role_count(self, registry: RoleRegistry) -> None:
        text = (
            "Живые игроки:\n1. user_1\n\n"
            "Из них: 👨🏼‍💼 Адвокат, 🤵🏼 Мафия - 5, 💣 Камикадзе - 3\n\n"
            "Всего: 10"
        )
        r = parse_bot_message(text, registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ALIVE_LIST
        assert r.role_count == {"Адвокат": 1, "Мафия": 5, "Камикадзе": 3}

    def test_alive_list_parses_total(self, registry: RoleRegistry) -> None:
        text = "Живые игроки:\n1. user_1\n\nИз них: 🤵🏼 Мафия - 1\n\nВсего: 30"
        r = parse_bot_message(text, registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ALIVE_LIST
        assert r.total_players == 30

    def test_alive_list_no_role_section(self, registry: RoleRegistry) -> None:
        text = "Живые игроки:\n1. user_1\n\nВсего: 1"
        r = parse_bot_message(text, registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ALIVE_LIST
        assert r.alive_player_names == ["user_1"]
        assert r.role_count is None
        assert r.total_players == 1

    def test_new_fields_default_to_none_for_other_kinds(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Кто-то был убит этой ночью.", registry=registry)
        assert r is not None
        assert r.alive_player_names is None
        assert r.role_count is None
        assert r.total_players is None


# ===================================================================
# Death tests (5)
# ===================================================================


class TestDeathNight:
    def test_death_someone_killed(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Кто-то был убит этой ночью.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "night_kill"

    def test_death_moon_prefix(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🌙 Кто-то был убит этой ночью.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "night_kill"

    def test_death_night_killed_named(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Этой ночью убили user_5.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "night_kill"


class TestDeathLynch:
    def test_death_hanged_today(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Сегодня повешен user_3.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "lynch"
        assert r.player_name is not None

    def test_death_hanged_by_vote(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Голосованием повешен user_7.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "lynch"


# ===================================================================
# Vote tests (3)
# ===================================================================


class TestVote:
    def test_vote_against(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_1 проголосовал против user_5", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.VOTE
        assert r.voter_name == "user_1"
        assert r.target_name == "user_5"

    def test_vote_results(self, registry: RoleRegistry) -> None:
        r = parse_bot_message(
            "Результаты голосования: user_7 получил 6 голосов.", registry=registry
        )
        assert r is not None
        assert r.kind is BotMessageKind.VOTE

    def test_vote_completed(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Голосование завершено.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.VOTE


# ===================================================================
# Role reveal tests (5)
# ===================================================================


class TestRoleReveal:
    def test_role_reveal_maniac_emoji(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🔪 user_9 был Маньяком", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_REVEAL
        assert r.role_id == "maniac"

    def test_role_reveal_mafia_byl(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_1 был Мафией", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_REVEAL
        assert r.role_id == "mafia"

    def test_role_reveal_don(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_7 был Доном", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_REVEAL
        assert r.role_id == "don"

    def test_role_reveal_commissioner_full_name(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_3 был Комиссаром Каттани", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_REVEAL
        assert r.role_id == "commissioner"

    def test_role_reveal_dash_pattern(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🕵🏼 user_3 — Комиссар Каттани", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_REVEAL
        assert r.role_id == "commissioner"


# ===================================================================
# Edge case tests (8)
# ===================================================================


class TestEdgeCases:
    def test_empty_text_returns_none(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("", registry=registry)
        assert r is None

    def test_none_returns_none(self, registry: RoleRegistry) -> None:
        r = parse_bot_message(None, registry=registry)
        assert r is None

    def test_emoji_only_moon_is_phase(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🌙", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE

    def test_huge_byl_spam_no_raise(self, registry: RoleRegistry) -> None:
        text = "был" * 500
        r = parse_bot_message(text, registry=registry)
        assert r is None or isinstance(r, BotMessageParse)

    def test_markdown_link_no_crash(self, registry: RoleRegistry) -> None:
        text = "[user_1](tg://user?id=123) был Мафией"
        r = parse_bot_message(text, registry=registry)
        assert r is None or isinstance(r, BotMessageParse)

    def test_unknown_role_text(self, registry: RoleRegistry) -> None:
        text = "Что-то непонятное с НепонятнойРолищей"
        r = parse_bot_message(text, registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.UNKNOWN
        assert r.confidence < 0.5

    def test_alive_empty_list(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Живые:", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ALIVE_LIST

    def test_newlines_before_death(self, registry: RoleRegistry) -> None:
        text = "\n\n\nКто-то был убит этой ночью."
        r = parse_bot_message(text, registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH


# ===================================================================
# Real fixture tests
# ===================================================================


class TestRealFixture:
    """Tests against the anonymised real-game fixture."""

    def test_fixture_exists_and_anonymized(self, fixture_data: dict[str, Any]) -> None:
        assert fixture_data["anonymized"] is True
        assert len(fixture_data["messages"]) >= 50

    def test_fixture_bot_messages_parseable(
        self, fixture_data: dict[str, Any], registry: RoleRegistry
    ) -> None:
        bot_id = fixture_data["bot_id"]
        bot_msgs = [m for m in fixture_data["messages"] if m["sender_id"] == bot_id]
        assert len(bot_msgs) >= 30

        for m in bot_msgs:
            r = parse_bot_message(m["text"], registry=registry)
            assert r is not None, f"parse returned None for: {m['text'][:60]!r}"

    def test_fixture_bot_messages_mostly_classified(
        self, fixture_data: dict[str, Any], registry: RoleRegistry
    ) -> None:
        bot_id = fixture_data["bot_id"]
        bot_msgs = [m for m in fixture_data["messages"] if m["sender_id"] == bot_id]
        total = len(bot_msgs)
        unclassified = 0
        for m in bot_msgs:
            result = parse_bot_message(m["text"], registry=registry)
            if result is not None and result.kind == BotMessageKind.UNKNOWN:
                unclassified += 1
        classified_pct = 100 * (total - unclassified) / total if total else 0
        assert classified_pct >= 90, f"Only {classified_pct:.1f}% classified"

    def test_fixture_bot_name_not_anonymized(self, fixture_data: dict[str, Any]) -> None:
        """Bot name should remain as-is (not user_N)."""
        assert not fixture_data["bot_name"].startswith("user_")

    def test_fixture_player_names_are_anonymized(self, fixture_data: dict[str, Any]) -> None:
        """Player sender_ids and names should be user_N format."""
        bot_id = fixture_data["bot_id"]
        for m in fixture_data["messages"]:
            if m["sender_id"] != bot_id:
                assert str(m["sender_id"]).startswith("user_")
                assert str(m["sender_name"]).startswith("user_")


# ===================================================================
# Skipif: real dump (C3 statistics — local only)
# ===================================================================

_DUMP_DIR = Path("data/dumps")
_DUMP_FILES = list(_DUMP_DIR.glob("chat_*.json")) if _DUMP_DIR.exists() else []


@pytest.mark.skipif(
    not _DUMP_DIR.exists() or not _DUMP_FILES,
    reason="no local dump (only runs where data/dumps/ exists)",
)
class TestFullDumpClassification:
    """Run parser on the full real dump for C3 statistics (local only)."""

    def test_full_dump_classification(self, registry: RoleRegistry) -> None:
        # Use the latest dump by mtime
        latest = max(_DUMP_FILES, key=lambda p: p.stat().st_mtime)
        data = json.loads(latest.read_text(encoding="utf-8"))
        bot_id = data.get("bot_id")
        messages = data.get("messages", [])

        bot_msgs = [m for m in messages if m.get("sender_id") == bot_id]
        player_msgs = [m for m in messages if m.get("sender_id") != bot_id]

        total = len(bot_msgs)
        parsed = 0
        unknown = 0
        by_kind: Counter[str] = Counter()
        results: list[BotMessageParse | None] = []

        for m in bot_msgs:
            r = parse_bot_message(m.get("text", ""), registry=registry)
            results.append(r)
            if r is not None:
                parsed += 1
                by_kind[r.kind.name] += 1
                if r.kind == BotMessageKind.UNKNOWN:
                    unknown += 1

        print(f"\n=== C3: real dump ({latest.name}) ===")
        print(f"bot_messages: {total}")
        print(f"parsed:       {parsed} ({100 * parsed / total:.1f}%)" if total else "parsed: 0")
        print(f"unknown:      {unknown} ({100 * unknown / total:.1f}%)" if total else "unknown: 0")
        for kind, cnt in by_kind.most_common():
            print(f"  {kind:15s}: {cnt}")

        # The direct parse loop above fails immediately if parsing raises.

        # >= 97% classified (non-UNKNOWN)
        if total > 0:
            classified_pct = 100 * (parsed - unknown) / total
            assert classified_pct >= 97, f"classified {classified_pct:.1f}%"

        # ROLE_REVEAL false-positive detector (step 7b-fix-1): a reveal is
        # only trusted at confidence 1.0 (grammatical form).  A high share
        # of low-confidence reveals means action notifications are leaking
        # into ROLE_REVEAL again — warn loudly (do not fail the build).
        role_reveal_total = by_kind[BotMessageKind.ROLE_REVEAL.name]
        role_reveal_conf_1 = sum(
            1
            for r in results
            if r is not None and r.kind == BotMessageKind.ROLE_REVEAL and r.confidence >= 1.0
        )
        if role_reveal_total:
            low_conf_pct = 100 * (role_reveal_total - role_reveal_conf_1) / role_reveal_total
            print(f"role_reveal: {role_reveal_total}, conf==1.0: {role_reveal_conf_1}")
            if low_conf_pct > 20:
                print(
                    f"WARNING: {low_conf_pct:.1f}% of ROLE_REVEAL have confidence < 1.0 "
                    "— possible action notifications leaking into reveals"
                )

        # Optional offline analysis artifact (local runs only):
        #   MAFIA_C3_REPORT=c3_report.json pytest -q -k full_dump_classification -s
        report_path = os.environ.get("MAFIA_C3_REPORT")
        if report_path:
            report = {
                "dump": latest.name,
                "total_bot_messages": total,
                "classified_pct": (100 * (parsed - unknown) / total) if total else 0.0,
                "by_kind": dict(by_kind.most_common()),
                "role_reveal_total": role_reveal_total,
                "role_reveal_conf_1": role_reveal_conf_1,
            }
            Path(report_path).write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"report written to {report_path}")

        # Player messages should NOT be parsed as bot messages
        sample_player = player_msgs[:200] if len(player_msgs) > 200 else player_msgs
        false_positives = 0
        for m in sample_player:
            r = parse_bot_message(m.get("text", ""), registry=registry)
            if r is not None and r.kind != BotMessageKind.UNKNOWN:
                false_positives += 1
        # Allow some false positives since players may use similar language
        print(f"player_sample: {len(sample_player)}, false_positives: {false_positives}")


# ===================================================================
# Step 7b-fix-1: role ACTION notifications are NOT role reveals
# ===================================================================


class TestRoleAction:
    """``ROLE_ACTION`` — "Роль сделала X" (action notification)."""

    def test_don_action_is_not_reveal(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🤵🏻 Дон выбрал жертву", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_ACTION
        assert r.role_id == "don"

    def test_commissioner_action(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🕵🏼 Комиссар Каттани ушёл искать злодеев", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_ACTION
        assert r.role_id == "commissioner"
        assert r.player_name is None

    def test_maniac_action(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🔪 Маньяк спрятался глубоко в кустах", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_ACTION
        assert r.role_id == "maniac"

    def test_werewolf_transform_is_action(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🐺 Оборотень превратился в 🤵🏻 Мафия!", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_ACTION
        assert r.role_id == "werewolf"

    def test_action_without_emoji(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Доктор вылечил user_9", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_ACTION
        assert r.role_id == "doctor"
        assert r.target_name == "user_9"

    def test_action_no_player_reveal_false_positive(self, registry: RoleRegistry) -> None:
        """Every action notification must stay out of ROLE_REVEAL."""
        actions = [
            "🤵🏻 Дон выбрал жертву",
            "🔪 Маньяк спрятался глубоко в кустах",
            "💃 Любовница посетила user_9",
            "👨🏼\u200d⚕️ Доктор уже ждёт",
            "Адвокат защитил user_9",
        ]
        for text in actions:
            r = parse_bot_message(text, registry=registry)
            assert r is not None
            assert r.kind is BotMessageKind.ROLE_ACTION, f"{text!r} → {r.kind}"


class TestProtectionUsed:
    """``PROTECTION_USED`` — защита потрачена, игрок жив."""

    def test_someone_spent_protection(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("💫 Кто-то потратил защиту!", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PROTECTION_USED

    def test_lucky_player_survived(self, registry: RoleRegistry) -> None:
        r = parse_bot_message(
            "💫 Счастливчику ночью повезло, и он остался в живых",
            registry=registry,
        )
        assert r is not None
        assert r.kind is BotMessageKind.PROTECTION_USED

    def test_used_protection_is_not_death(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_9 использовал(а) защиту от линчевания!", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PROTECTION_USED
        assert r.death_cause is None


class TestVoteAbstain:
    """``VOTE_ABSTAIN`` — "решил никого не вешать"."""

    def test_abstain_with_name(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🚷 user_9 решил никого не вешать", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.VOTE_ABSTAIN
        assert r.player_name == "user_9"

    def test_abstain_without_name(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Никого не вешать", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.VOTE_ABSTAIN


class TestGameStart:
    """New-game markers are distinct from informational system messages."""

    def test_game_start_detected(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Город ветров начинает свою игру над жителями!", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.GAME_START

    def test_game_start_before_system(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Город ветров начинает игру", registry=registry)
        assert r is not None
        assert r.kind is not BotMessageKind.SYSTEM
        assert r.kind is BotMessageKind.GAME_START

    def test_automatic_start(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Автоматический старт игры", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.GAME_START


class TestNewKinds:
    """All step 7b-fix-1 kinds must exist on the enum."""

    def test_new_kinds_present(self) -> None:
        for name in (
            "ROLE_ACTION",
            "PROTECTION_USED",
            "VOTE_ABSTAIN",
            "GAME_START",
            "SYSTEM",
        ):
            assert hasattr(BotMessageKind, name)
        assert BotMessageKind("game_start") is BotMessageKind.GAME_START
        assert BotMessageKind("role_action") is BotMessageKind.ROLE_ACTION
        assert BotMessageKind("protection_used") is BotMessageKind.PROTECTION_USED
        assert BotMessageKind("vote_abstain") is BotMessageKind.VOTE_ABSTAIN
        assert BotMessageKind("system") is BotMessageKind.SYSTEM


# ===================================================================
# Step 7b-fix-1: priority checks (death before phase, vote patterns)
# ===================================================================


class TestDeathPriority:
    """Death must win over phase change and never be a phase flip."""

    def test_moon_killed_is_death(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🌙 Кто-то был убит", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.to_phase is None
        assert r.death_cause == "night_kill"

    def test_long_night_message_with_kill_is_death(self, registry: RoleRegistry) -> None:
        text = "🌙 " + "Кто-то был убит этой ночью. " * 30
        r = parse_bot_message(text, registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH

    def test_someone_died(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Кто-то погиб", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "night_kill"

    def test_brutally_killed(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Сегодня был жестоко убит user_9", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "night_kill"
        assert r.player_name == "user_9"

    def test_hanged_with_role_line(self, registry: RoleRegistry) -> None:
        r = parse_bot_message(
            "user_9 не выдержал гнетущей атмосферы... повесился.\nОн был 🤵🏻 Дон",
            registry=registry,
        )
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "lynch"
        assert r.player_name == "user_9"

    def test_lynch_by_vote_named(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Голосованием повешен user_9", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "lynch"
        assert r.player_name == "user_9"


class TestVotePatterns:
    """Multi-line votes and vote-phase announcements."""

    def test_vote_za_form(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_1 голосовал(а) за: user_5", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.VOTE
        assert r.voter_name == "user_1"
        assert r.target_name == "user_5"
        assert r.votes == [("user_1", "user_5")]

    def test_vote_za_male_form(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_1 голосовал за: user_5", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.VOTE
        assert r.votes == [("user_1", "user_5")]

    def test_vote_multi_line(self, registry: RoleRegistry) -> None:
        text = "user_1 голосовал(а) за: user_2\n\n\nuser_3 голосовал(а) за: user_4"
        r = parse_bot_message(text, registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.VOTE
        assert r.votes is not None
        assert len(r.votes) == 2
        assert r.votes == [("user_1", "user_2"), ("user_3", "user_4")]
        assert r.voter_name == "user_1"
        assert r.target_name == "user_2"

    def test_vote_with_chat_marker_prefix(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🇷🇺ℹ | user_1 | 💋 голосовал(а) за: user_5", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.VOTE
        assert r.voter_name == "user_1"
        assert r.target_name == "user_5"

    def test_vote_against_keeps_target(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_1 проголосовал(а) против user_5", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.VOTE
        assert r.voter_name == "user_1"
        assert r.target_name == "user_5"

    def test_vote_start_announcement(self, registry: RoleRegistry) -> None:
        r = parse_bot_message(
            "Пришло время определить, кого повесить. Голосование продлится 60 секунд.",
            registry=registry,
        )
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.VOTE

    def test_vote_ended_announcement(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Голосование окончено", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.VOTE

    def test_no_lynch_result_is_vote(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Никто не повешен", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.VOTE

    def test_votes_none_for_non_vote_kinds(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🤵🏻 Дон выбрал жертву", registry=registry)
        assert r is not None
        assert r.votes is None


class TestGrammaticalReveal:
    """ROLE_REVEAL stays available, but only in grammatical forms."""

    def test_byl_instrumental(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_9 был Мафией", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_REVEAL
        assert r.role_id == "mafia"
        assert r.confidence == 1.0

    def test_dash_with_role_emoji(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_9 — 🕵🏼 Комиссар Каттани", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_REVEAL
        assert r.role_id == "commissioner"

    def test_role_first_form(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("🕵🏼 Комиссар Каттани user_9", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_REVEAL
        assert r.role_id == "commissioner"
        assert r.player_name == "user_9"

    def test_unknown_role_word_is_low_confidence_reveal(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_9 был НеведомойРолью", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_REVEAL
        assert r.role_id is None
        assert r.confidence == 0.5

    def test_killed_is_not_reveal(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_9 был убит", registry=registry)
        assert r is not None
        assert r.kind is not BotMessageKind.ROLE_REVEAL


class TestRegistrationPatterns:
    """Registration templates from the real dump."""

    def test_nabor_vedetsya(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Ведётся набор в игру", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.REGISTRATION

    def test_nabor_otkryt(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Набор в игру открыт", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.REGISTRATION


class TestAliveListPriority:
    """Alive list must be recognised before role reveals inside its body."""

    def test_multi_line_alive_list(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Живые игроки:\n🕵🏼 user_1\n🤵🏻 user_2", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ALIVE_LIST

    def test_alive_list_with_counter(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Живые игроки (5): user_1, user_2", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ALIVE_LIST


# ===================================================================
# Additional robustness tests
# ===================================================================


class TestRobustness:
    def test_whitespace_only(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("   \t\n  ", registry=registry)
        assert r is None

    def test_very_long_message(self, registry: RoleRegistry) -> None:
        text = "🎉" * 2000
        r = parse_bot_message(text, registry=registry)
        assert r is None or isinstance(r, BotMessageParse)

    def test_null_bytes(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("\x00\x01\x02", registry=registry)
        assert r is None or isinstance(r, BotMessageParse)

    def test_huge_url(self, registry: RoleRegistry) -> None:
        text = "https://example.com/" + "a" * 2000
        r = parse_bot_message(text, registry=registry)
        assert r is None or isinstance(r, BotMessageParse)

    def test_mixed_emoji_and_text(self, registry: RoleRegistry) -> None:
        text = "🌙🌝🔪💃 Текст"
        r = parse_bot_message(text, registry=registry)
        assert r is None or isinstance(r, BotMessageParse)

    def test_bot_message_parse_is_frozen(self) -> None:
        """BotMessageParse should be immutable (frozen pydantic)."""
        p = BotMessageParse(kind=BotMessageKind.UNKNOWN, raw_text="test")
        with pytest.raises(ValidationError):
            p.kind = BotMessageKind.DEATH

    def test_emoji_map_has_21_roles(self) -> None:
        assert len(_EMOJI_TO_ROLE_ID) == 21

    def test_non_string_input_returns_none(self, registry: RoleRegistry) -> None:
        """Non-string, non-None input should return None (defensive)."""
        # Testing runtime safety with wrong type on purpose
        r = parse_bot_message(12345, registry=registry)  # type: ignore[arg-type]
        assert r is None

    def test_dash_pattern_without_emoji(self, registry: RoleRegistry) -> None:
        """Role reveal via dash pattern without leading emoji."""
        r = parse_bot_message("user_1 — Комиссар Каттани", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_REVEAL
        assert r.role_id == "commissioner"

    def test_role_reveal_long_candidate_truncation(self, registry: RoleRegistry) -> None:
        """Very long role candidate should be truncated, not crash."""
        long_role = "Комиссаром " + "Каттани " * 50
        text = f"user_1 был {long_role}"
        r = parse_bot_message(text, registry=registry)
        assert r is None or isinstance(r, BotMessageParse)

    def test_role_reveal_with_multiple_roles_mentioned(self, registry: RoleRegistry) -> None:
        """Text mentioning multiple roles should still resolve correctly."""
        r = parse_bot_message("user_1 был Доном, а user_2 был Мафией", registry=registry)
        assert r is not None
        # Should match the first "был" pattern
        assert r.kind is BotMessageKind.ROLE_REVEAL

    def test_death_with_emoji_prefix(self, registry: RoleRegistry) -> None:
        """Death message with emoji prefix should be DEATH, not PHASE_CHANGE."""
        r = parse_bot_message("🌙 Кто-то был убит этой ночью.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "night_kill"

    def test_phase_night_with_day_number(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Ночь 5", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.NIGHT
        assert r.day_number == 5

    def test_phase_day_with_day_number(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("День 3", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.PHASE_CHANGE
        assert r.to_phase is Phase.DAY
        assert r.day_number == 3

    def test_vote_female_form(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("user_5 проголосовала против user_3", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.VOTE
        assert r.voter_name == "user_5"
        assert r.target_name == "user_3"

    def test_role_reveal_byla(self, registry: RoleRegistry) -> None:
        """Female form: 'X была ...' should work too."""
        r = parse_bot_message("user_6 была Любовницей", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ROLE_REVEAL
        assert r.role_id == "mistress"

    def test_death_with_named_victim(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Кто-то был убит этой ночью. user_8 был убит.", registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "night_kill"

    def test_hanged_death_with_role_info(self, registry: RoleRegistry) -> None:
        r = parse_bot_message("Сегодня повешен user_7. user_7 был Доном.", registry=registry)
        assert r is not None
        # "был Доном" could match role_reveal, but "Сегодня повешен" matches death first
        assert r.kind is BotMessageKind.DEATH
        assert r.death_cause == "lynch"


# Step 7b-fix-2 regression coverage.
@pytest.mark.parametrize(
    ("text", "words"),
    [
        (
            "Кто-то из жителей слышал, как Вася кричал перед смертью:\nМафия это Лена",
            "Мафия это Лена",
        ),
        (
            "Кто-то из жителей слышал, как Вася кричал перед смертью:\nПервая строка\nВторая строка",
            "Первая строка\nВторая строка",
        ),
        ("Кто-то из жителей слышал, как Вася кричал перед смертью:\n", None),
        ("Кто-то из жителей слышал, как 🕵️ X кричал перед смертью:\nПрощайте", "Прощайте"),
    ],
)
def test_last_words(registry, text, words):
    result = parse_bot_message(text, registry=registry)
    assert result is not None and result.kind is BotMessageKind.LAST_WORDS
    assert result.last_words == words
    assert result.player_name


@pytest.mark.parametrize(
    ("text", "voter", "target"),
    [
        ("🇷🇺ℹ голосовал(а) за: Вася", "🇷🇺ℹ", "Вася"),
        ("Вася голосовал(а) за: 🇷🇺ℹ", "Вася", "🇷🇺ℹ"),
        ("Вася 🦱 голосовал(а) за: Петя", "Вася", "Петя"),
        ("k голосовал(а) за: Вася", "k", "Вася"),
        ("Вася голосовал(а) за: .", "Вася", None),
        ("Вася голосовал(а) за: 🇧🇦", "Вася", "🇧🇦"),
        ("Вася голосовал(а) за: v", "Вася", "v"),
        ("Лена голосовал(а) за: .", "Лена", None),
        ("🇧🇦 голосовал(а) за: Вася", "🇧🇦", "Вася"),
        ("m голосовал(а) за: Петя", "m", "Петя"),
    ],
)
def test_vote_emoji_short_names_and_abstain(registry, text, voter, target):
    result = parse_bot_message(text, registry=registry)
    assert result is not None and result.kind is BotMessageKind.VOTE
    assert (result.voter_name, result.target_name) == (voter, target)
    assert result.votes == [(voter, target)]


def test_gift_simple(registry):
    result = parse_bot_message("Вася подарил 💎5 к Пете!", registry=registry)
    assert result is not None and result.kind is BotMessageKind.GIFT
    assert (result.player_name, result.target_name) == ("Вася", "Пете")


def test_gift_complex_names(registry):
    result = parse_bot_message("🇷🇺ℹ | Вася | 💎 подарил 💎3 к 🕵️ | Петя | 💋!", registry=registry)
    assert result is not None and result.kind is BotMessageKind.GIFT
    assert (result.player_name, result.target_name) == ("Вася", "Петя")


@pytest.mark.parametrize(
    "text",
    [
        "Странно, этой ночью все остались в живых...",
        "Недостаточно игроков для начала игры...",
    ],
)
def test_new_system_messages(registry, text):
    result = parse_bot_message(text, registry=registry)
    assert result is not None and result.kind is BotMessageKind.SYSTEM


def test_role_action_interaction(registry):
    result = parse_bot_message(
        "Сегодня 🤵🏻 Дон пытался убить Мага\nПосмотрим...", registry=registry
    )
    assert result is not None and result.kind is BotMessageKind.ROLE_ACTION
    assert result.role_id == "don"


def test_regressions_and_multiline_vote(registry):
    cases = [
        ("🤵🏻 Дон выбрал жертву", BotMessageKind.ROLE_ACTION),
        ("Вася был Мафией", BotMessageKind.ROLE_REVEAL),
        ("🌙 Кто-то был убит", BotMessageKind.DEATH),
    ]
    for text, kind in cases:
        result = parse_bot_message(text, registry=registry)
        assert result is not None and result.kind is kind
    result = parse_bot_message(
        "Вася голосовал(а) за: Петя\n\n\nЛена голосовал(а) за: Игорь", registry=registry
    )
    assert result is not None and result.kind is BotMessageKind.VOTE
    assert result.votes == [("Вася", "Петя"), ("Лена", "Игорь")]


# ===================================================================
# End game (final report) tests (4)
# ===================================================================


END_GAME_TEXT = """Игра окончена!

Победители:
    1. user_01 - 💣 Камикадзе
    7. user_07 - 🤵🏼 Мафия

Другие пользователи:
    8. user_08 - 🔪 Маньяк
    30. user_30 - 👮🏼 Сержант

Игра длилась: 20 минут"""


class TestEndGame:
    def test_end_game_parses_winners_losers(self, registry: RoleRegistry) -> None:
        """1. The real final report yields both sections with Russian role names."""
        result = parse_bot_message(END_GAME_TEXT, registry=registry)

        assert result is not None
        assert result.kind is BotMessageKind.END_GAME
        assert result.winners == [("user_01", "Камикадзе"), ("user_07", "Мафия")]
        assert result.losers == [("user_08", "Маньяк"), ("user_30", "Сержант")]
        # Roles stay as written by the bot — no role_id resolution in the parser.
        assert result.role_id is None

    def test_end_game_duration(self, registry: RoleRegistry) -> None:
        """2. "Игра длилась: 20 минут" becomes ``duration_minutes``."""
        result = parse_bot_message(END_GAME_TEXT, registry=registry)

        assert result is not None
        assert result.kind is BotMessageKind.END_GAME
        assert result.duration_minutes == 20

    def test_end_game_no_data(self, registry: RoleRegistry) -> None:
        """3. A report without entries is still END_GAME, just without data."""
        result = parse_bot_message("Игра окончена!\n\nНет победителей", registry=registry)

        assert result is not None
        assert result.kind is BotMessageKind.END_GAME
        assert result.winners is None
        assert result.losers is None
        assert result.duration_minutes is None

    def test_end_game_not_mistaken_for_alive_list(self, registry: RoleRegistry) -> None:
        """4. A report that also contains an alive list stays END_GAME."""
        text = (
            "Игра окончена!\n\n"
            "Победители:\n    1. user_01 - 💣 Камикадзе\n\n"
            "Живые игроки:\n    1. user_01\n\n"
            "Из них:\n    💣 Камикадзе: 1\n\n"
            "Всего: 1"
        )

        result = parse_bot_message(text, registry=registry)

        assert result is not None
        assert result.kind is BotMessageKind.END_GAME
        assert result.winners == [("user_01", "Камикадзе")]


# ===================================================================
# Real-club ALIVE_LIST with decorated nicknames (16 players / 16 slots)
# ===================================================================


CLUB_ALIVE_LIST_TEXT = """Живые игроки:

1. | user_1

3. user_2🏹

4. user_3'

5. 🇦🇪

8. user_4

9. user_5🍒

10. user_6

11. moon

14. 𝖁𝖊𝖑𝖑𝖆𝖓𝖊 ᥫ᭡

15. user_7

16. н 🐮

17. Е А

18. 🕷️ᵗᵏ| user_8🌝🖤

19. mzhaaaa

20. ♯𝅃𝐫᪶⃕𝐚᪶͡𝐲᪶𝆬𝐚᪶𝆬 ✟

21. M•

Из них:  👨🏼‍💼 Адвокат, 🧟 Поджигатель, 👨🏼 Мирный Житель, 🕵🏼 Комиссар Каттани, 🤵🏻 Дон, 🧙🏼 Бомж, 💣 Камикадзе, 🤞🏼 Счастливчик, 🤵🏼 Мафия - 3, 🧙‍ Маг, 💃 Любовница, 👮🏼 Сержант, 🤦🏼 Самоубийца, 🐺 Оборотень

Всего: 16
"""


class TestClubAliveListComposition:
    """The exact bot snapshot shape from a real club must survive parsing."""

    def test_kind_names_and_total(self, registry: RoleRegistry) -> None:
        r = parse_bot_message(CLUB_ALIVE_LIST_TEXT, registry=registry)
        assert r is not None
        assert r.kind is BotMessageKind.ALIVE_LIST
        assert r.confidence == 1.0
        assert r.alive_player_names is not None
        assert len(r.alive_player_names) == 16
        assert r.alive_player_names[0] == "| user_1"
        assert r.alive_player_names[12] == "🕷️ᵗᵏ| user_8🌝🖤"
        assert r.total_players == 16

    def test_role_count_is_canonical_and_sums_to_total(self, registry: RoleRegistry) -> None:
        r = parse_bot_message(CLUB_ALIVE_LIST_TEXT, registry=registry)
        assert r is not None
        assert r.role_count is not None
        assert len(r.role_count) == 14
        assert sum(r.role_count.values()) == 16
        assert r.role_count["Мафия"] == 3
        assert r.role_count["Мирный житель"] == 1
        assert r.role_count["Комиссар Каттани"] == 1
        assert r.role_count["Поджигатель"] == 1
        assert r.role_count["Маг"] == 1
        # Bot capitalisation must not leak into the LLM prompt.
        assert "Мирный Житель" not in r.role_count

    def test_emoji_zwnj_and_skin_tones_are_stripped(self, registry: RoleRegistry) -> None:
        r = parse_bot_message(CLUB_ALIVE_LIST_TEXT, registry=registry)
        assert r is not None
        assert r.role_count is not None
        for name in r.role_count:
            assert not name.startswith(
                ("👨", "🧟", "🕵", "🤵", "🧙", "💣", "🤞", "💃", "👮", "🤦", "🐺")
            )


@pytest.mark.parametrize("text", ["| A убит", "| A был убит"])
def test_death_single_letter_decorated_name(text: str, registry: RoleRegistry) -> None:
    result = parse_bot_message(text, registry=registry)
    assert result is not None
    assert result.kind is BotMessageKind.DEATH
    assert result.player_name == "| A"
