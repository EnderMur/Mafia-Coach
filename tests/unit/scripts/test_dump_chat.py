"""Network-free tests for the read-only Telegram history dump CLI."""

from __future__ import annotations

import ast
import importlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from loguru import logger

from domain.exceptions import (
    ConfigError,
    TelegramAccessError,
    TelegramFloodWaitError,
    TelegramNotConnectedError,
)
from infra.telegram.client import TelegramClientWrapper

# scripts is a CLI directory, not an installed package. Dynamic import also keeps
# mypy from assigning its standalone file a second, conflicting module name.
dump_chat = importlib.import_module("scripts.dump_chat")

API_ID = 123456
API_HASH = "0" * 32
CHAT_ID = -1001234567890
BOT_ID = 888002
BOT_NAME = "Fixture Game Bot"
PLAYER_IDS = (1001, 1002, 1003)


def make_message(
    number: int,
    sender_id: int = PLAYER_IDS[0],
    *,
    text: str | None = None,
    date: datetime | None = None,
    reply_to_msg_id: int | None = None,
) -> SimpleNamespace:
    """Build a synthetic message with the attributes returned by Telethon."""
    return SimpleNamespace(
        id=number,
        sender_id=sender_id,
        sender=SimpleNamespace(first_name="Fixture", last_name=f"Player {sender_id}"),
        date=date or datetime(2026, 10, 31, 12, 0, tzinfo=UTC) + timedelta(seconds=number),
        message=f"сообщение {number}" if text is None else text,
        reply_to=(
            SimpleNamespace(reply_to_msg_id=reply_to_msg_id)
            if reply_to_msg_id is not None
            else None
        ),
    )


def write_settings(path: Path, telegram: dict[str, Any]) -> None:
    """Write synthetic settings; no real credentials or chat identifiers are used."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"telegram": telegram}, allow_unicode=True), encoding="utf-8")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Replace the wrapper constructor and logging setup before any CLI invocation."""
    monkeypatch.chdir(tmp_path)
    session = tmp_path / "data" / "sessions" / "fixture"
    session.parent.mkdir(parents=True)
    session.with_suffix(".session").write_bytes(b"")
    config_path = tmp_path / "configs" / "settings.yaml"
    telegram: dict[str, Any] = {
        "session_file": str(session),
        "api_id": API_ID,
        "api_hash": API_HASH,
        "chat_id": CHAT_ID,
        "bot_id": BOT_ID,
        "bot_username": None,
        "read_only": True,
        "ignore_own_messages": True,
    }
    write_settings(config_path, telegram)

    wrapper = AsyncMock(spec=TelegramClientWrapper)
    wrapper.bot_id = BOT_ID
    wrapper.connect = AsyncMock()
    wrapper.aclose = AsyncMock()
    wrapper.list_recent_messages = AsyncMock(return_value=[])
    entities = {
        CHAT_ID: SimpleNamespace(id=CHAT_ID, title="Fixture Mafia Chat"),
        BOT_ID: SimpleNamespace(id=BOT_ID, first_name="Fixture Game", last_name="Bot"),
        "@fixture_chat": SimpleNamespace(id=555111, title="Username Fixture Chat"),
    }

    async def get_chat(reference: int | str) -> Any:
        return entities[reference]

    wrapper.get_chat = AsyncMock(side_effect=get_chat)
    with (
        patch("scripts.dump_chat.TelegramClientWrapper", return_value=wrapper) as constructor,
        patch("scripts.dump_chat.setup_logging") as setup_logging,
        patch("scripts.dump_chat.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        yield SimpleNamespace(
            root=tmp_path,
            config_path=config_path,
            telegram=telegram,
            wrapper=wrapper,
            constructor=constructor,
            setup_logging=setup_logging,
            sleep=sleep,
            entities=entities,
        )


def read_json(path: Path) -> dict[str, Any]:
    """Read a generated synthetic dump."""
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def update_settings(env: SimpleNamespace, **changes: Any) -> None:
    """Update the temporary settings file for a test."""
    env.telegram.update(changes)
    write_settings(env.config_path, env.telegram)


def test_limit_output_and_real_sender_ids(env: SimpleNamespace) -> None:
    """1. A raw dump preserves 100 sender IDs and passes the requested limit."""
    newest_first = [make_message(i, PLAYER_IDS[i % 3]) for i in range(100, 0, -1)]
    env.wrapper.list_recent_messages.return_value = newest_first
    output = env.root / "tmp" / "x.json"

    assert dump_chat.run_cli(["--limit", "100", "--output", str(output), "--no-anonymize"]) == 0

    payload = read_json(output)
    assert payload["message_count"] == len(payload["messages"]) == 100
    assert payload["anonymized"] is False
    assert [msg["sender_id"] for msg in payload["messages"]] == [
        PLAYER_IDS[i % 3] for i in range(1, 101)
    ]
    assert payload["chat_title"] == "Fixture Mafia Chat"
    env.wrapper.list_recent_messages.assert_awaited_once_with(chat_id=CHAT_ID, limit=100)
    env.wrapper.aclose.assert_awaited_once()


async def test_anonymization_defaults_on_and_preserves_bot(
    env: SimpleNamespace, tmp_path: Path
) -> None:
    """2. Default anonymization changes players, but never the game bot."""
    output = tmp_path / "data" / "anonymous.json"
    env.wrapper.list_recent_messages.return_value = [
        make_message(2, BOT_ID),
        make_message(1, PLAYER_IDS[0]),
    ]

    await dump_chat.main(["--output", str(output)])

    payload = read_json(output)
    assert payload["anonymized"] is True
    assert payload["bot_id"] == BOT_ID
    assert payload["bot_name"] == BOT_NAME
    assert payload["messages"][0]["sender_id"] == "user_1"
    assert payload["messages"][0]["sender_name"] == "user_1"
    assert payload["messages"][1]["sender_id"] == BOT_ID
    assert payload["messages"][1]["sender_name"] == BOT_NAME
    assert payload["chat_title"] == ""


async def test_same_sender_gets_same_alias(env: SimpleNamespace, tmp_path: Path) -> None:
    """3. Two messages from one player share one alias, without rewriting text."""
    output = tmp_path / "anonymous.json"
    env.wrapper.list_recent_messages.return_value = [
        make_message(2, text="Fixture Player 1001 говорит снова"),
        make_message(1, text="Fixture Player 1001 начинает"),
    ]

    await dump_chat.main(["--anonymize", "--output", str(output)])

    messages = read_json(output)["messages"]
    assert [msg["sender_id"] for msg in messages] == ["user_1", "user_1"]
    assert [msg["sender_name"] for msg in messages] == ["user_1", "user_1"]
    assert [msg["text"] for msg in messages] == [
        "Fixture Player 1001 начинает",
        "Fixture Player 1001 говорит снова",
    ]


async def test_three_senders_numbered_by_first_chronological_appearance(
    env: SimpleNamespace, tmp_path: Path
) -> None:
    """4. The bot does not consume an alias and repeated players do not increment N."""
    output = tmp_path / "anonymous.json"
    env.wrapper.list_recent_messages.return_value = [
        make_message(6, PLAYER_IDS[2]),
        make_message(5, PLAYER_IDS[0]),
        make_message(4, PLAYER_IDS[1]),
        make_message(3, BOT_ID),
        make_message(2, PLAYER_IDS[1]),
        make_message(1, PLAYER_IDS[0]),
    ]

    await dump_chat.main(["--output", str(output)])

    assert [msg["sender_id"] for msg in read_json(output)["messages"]] == [
        "user_1",
        "user_2",
        BOT_ID,
        "user_2",
        "user_1",
        "user_3",
    ]


async def test_newest_first_history_is_reversed(env: SimpleNamespace, tmp_path: Path) -> None:
    """5. Telethon's newest-first result becomes an oldest-first JSON chronology."""
    output = tmp_path / "chronological.json"
    env.wrapper.list_recent_messages.return_value = [
        make_message(3, PLAYER_IDS[2]),
        make_message(2, PLAYER_IDS[1]),
        make_message(1, PLAYER_IDS[0]),
    ]

    await dump_chat.main(["--output", str(output)])

    messages = read_json(output)["messages"]
    assert [msg["text"] for msg in messages] == ["сообщение 1", "сообщение 2", "сообщение 3"]
    assert [msg["sender_id"] for msg in messages] == ["user_1", "user_2", "user_3"]
    assert [msg["ts"] for msg in messages] == sorted(msg["ts"] for msg in messages)


async def test_flood_wait_logs_waits_and_retries_once(
    env: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """6. A history FloodWait is logged, waited out, and retried exactly once."""
    output = tmp_path / "retried.json"
    env.wrapper.list_recent_messages.side_effect = [
        TelegramFloodWaitError(7),
        [make_message(1)],
    ]
    records: list[Any] = []
    sink = logger.add(records.append, level="WARNING")
    try:
        await dump_chat.main(["--output", str(output)])
    finally:
        logger.remove(sink)

    assert read_json(output)["message_count"] == 1
    assert env.wrapper.list_recent_messages.await_count == 2
    env.sleep.assert_awaited_once_with(7)
    assert any(
        record.record["level"].name == "WARNING"
        and "waiting 7 seconds (FloodWait), retrying once..." in record.record["message"]
        for record in records
    )
    assert "waiting 7 seconds (FloodWait), retrying once..." in capsys.readouterr().err
    env.wrapper.aclose.assert_awaited_once()


def test_missing_chat_exits_with_actionable_hint(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """7. No CLI chat and a null configured chat fail before constructing a client."""
    update_settings(env, chat_id=None)

    assert dump_chat.run_cli([]) == 1

    assert "--chat-id" in capsys.readouterr().err
    env.constructor.assert_not_called()
    env.wrapper.connect.assert_not_awaited()


def test_raw_dump_inside_data_still_warns(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """8. Even the nominal data/dumps location is not a license to commit raw PII."""
    output = Path("data/dumps/raw.json")
    env.wrapper.list_recent_messages.return_value = [make_message(1)]

    assert dump_chat.run_cli(["--output", str(output), "--no-anonymize"]) == 0

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "DO NOT COMMIT" in captured.err
    assert read_json(output)["messages"][0]["sender_id"] == PLAYER_IDS[0]
    assert f"OK: wrote 1 messages to {output}" in captured.out


def test_parse_defaults_and_quiet_logging(env: SimpleNamespace) -> None:
    """The advertised defaults are stable and the CLI config path is relative to cwd."""
    args = dump_chat.parse_args([])
    assert args.limit == 10_000
    assert args.anonymize is True
    assert args.chat_id is args.chat_username is args.bot_id is None
    assert args.config == Path("configs/settings.yaml")
    assert args.output.parent == Path("data/dumps")
    assert args.output.name.startswith("chat_")
    assert args.output.suffix == ".json"

    assert dump_chat.run_cli([]) == 0
    env.setup_logging.assert_called_once_with(level="WARNING", log_dir="logs")
    env.wrapper.list_recent_messages.assert_awaited_once_with(chat_id=CHAT_ID, limit=10_000)
    assert len(list(Path("data/dumps").glob("chat_*.json"))) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--chat-id", "1", "--chat-username", "@fixture_chat"],
        ["--anonymize", "--no-anonymize"],
        ["--limit", "0"],
        ["--limit", "-1"],
        ["--limit", "many"],
        ["--bot-id", "0"],
    ],
)
def test_invalid_arguments_exit_two(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """Conflicting selectors/privacy flags and nonpositive counts are rejected by argparse."""
    with pytest.raises(SystemExit) as exc:
        dump_chat.parse_args(argv)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


async def test_cli_chat_id_overrides_settings(env: SimpleNamespace, tmp_path: Path) -> None:
    """A numeric CLI chat overrides YAML, including in the validated wrapper config."""
    other_id = -100999000111
    env.entities[other_id] = SimpleNamespace(id=other_id, title="Other Fixture Chat")
    output = tmp_path / "override.json"

    await dump_chat.main(["--chat-id", str(other_id), "--output", str(output)])

    assert read_json(output)["chat_id"] == other_id
    assert env.constructor.call_args.kwargs["config"].chat_id == other_id
    env.wrapper.list_recent_messages.assert_awaited_once_with(chat_id=other_id, limit=10_000)


@pytest.mark.parametrize("from_cli", [True, False])
async def test_chat_username_resolves_through_wrapper(
    env: SimpleNamespace, tmp_path: Path, from_cli: bool
) -> None:
    """Both username sources use get_chat, not direct Telethon entity resolution."""
    output = tmp_path / "username.json"
    if from_cli:
        argv = ["--chat-username", "@fixture_chat", "--output", str(output)]
    else:
        update_settings(env, chat_id="@fixture_chat")
        argv = ["--output", str(output)]

    await dump_chat.main(argv)

    assert read_json(output)["chat_id"] == 555111
    assert env.constructor.call_args.kwargs["config"].chat_id is None
    env.wrapper.get_chat.assert_any_await("@fixture_chat")
    env.wrapper.list_recent_messages.assert_awaited_once_with(chat_id=555111, limit=10_000)


async def test_bot_id_cli_override(env: SimpleNamespace, tmp_path: Path) -> None:
    """An explicit bot ID reaches the wrapper and remains unanonymized in output."""
    other_bot = 999002
    env.entities[other_bot] = SimpleNamespace(id=other_bot, first_name="Override Bot")
    env.wrapper.bot_id = other_bot
    env.wrapper.list_recent_messages.return_value = [make_message(1, other_bot)]
    output = tmp_path / "bot.json"

    await dump_chat.main(["--bot-id", str(other_bot), "--output", str(output)])

    assert env.constructor.call_args.kwargs["config"].bot_id == other_bot
    payload = read_json(output)
    assert payload["bot_id"] == other_bot
    assert payload["bot_name"] == "Override Bot"
    assert payload["messages"][0]["sender_id"] == other_bot


async def test_bot_username_is_resolved_by_wrapper_connect(
    env: SimpleNamespace, tmp_path: Path
) -> None:
    """With no numeric bot ID, connect supplies the resolved wrapper.bot_id."""
    update_settings(env, bot_id=None, bot_username="@FixtureGameBot")
    env.wrapper.list_recent_messages.return_value = [make_message(1, BOT_ID)]
    output = tmp_path / "resolved-bot.json"

    await dump_chat.main(["--output", str(output)])

    config = env.constructor.call_args.kwargs["config"]
    assert config.bot_id is None
    assert config.bot_username == "@FixtureGameBot"
    env.wrapper.connect.assert_awaited_once()
    env.wrapper.get_chat.assert_any_await(BOT_ID)
    assert read_json(output)["bot_id"] == BOT_ID


def test_unresolved_bot_is_a_domain_error_and_closes_wrapper(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """A wrapper without a resolved bot ID cannot produce an ambiguous dump."""
    env.wrapper.bot_id = None

    assert dump_chat.run_cli([]) == 1

    assert "cannot resolve bot_id" in capsys.readouterr().err
    env.wrapper.list_recent_messages.assert_not_awaited()
    env.wrapper.aclose.assert_awaited_once()


def test_missing_session_fails_before_wrapper_construction(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing session produces the exact recovery command and never opens a client."""
    update_settings(env, session_file=str(env.root / "missing-session"))

    assert dump_chat.run_cli([]) == 1

    assert "session file missing. Run: python scripts/make_session.py" in capsys.readouterr().err
    env.constructor.assert_not_called()


@pytest.mark.parametrize("session_file", [None, "", "   ", 123])
def test_invalid_session_path_is_rejected(
    env: SimpleNamespace, session_file: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Session paths must be nonblank strings, not arbitrary YAML values."""
    update_settings(env, session_file=session_file)
    assert dump_chat.run_cli([]) == 1
    assert "session_file" in capsys.readouterr().err
    env.constructor.assert_not_called()


@pytest.mark.parametrize("content", ["", "[]", "telegram: []", "telegram: [invalid"])
def test_invalid_settings_are_domain_errors(
    env: SimpleNamespace, content: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing/malformed settings never fall back to example credentials or a real client."""
    env.config_path.write_text(content, encoding="utf-8")
    assert dump_chat.run_cli([]) == 1
    assert "ERROR:" in capsys.readouterr().err
    env.constructor.assert_not_called()


def test_missing_config_and_custom_config_path(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """The explicit --config path is honored and its absence is explained."""
    missing = env.root / "missing.yaml"
    assert dump_chat.run_cli(["--config", str(missing)]) == 1
    assert str(missing) in capsys.readouterr().err

    custom = env.root / "custom.yaml"
    write_settings(custom, env.telegram)
    assert dump_chat.run_cli(["--config", str(custom)]) == 0
    env.constructor.assert_called_once()


def test_invalid_wrapper_settings_do_not_echo_credentials(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pydantic failures become ConfigError without printing the API hash input."""
    update_settings(env, api_hash="sensitive-short")
    assert dump_chat.run_cli([]) == 1
    stderr = capsys.readouterr().err
    assert "invalid telegram settings" in stderr
    assert "sensitive-short" not in stderr
    env.constructor.assert_not_called()


@pytest.mark.parametrize("chat_id", [True, [], {}, "   "])
def test_invalid_configured_chat_is_rejected(
    env: SimpleNamespace, chat_id: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only integer IDs, nonblank usernames, and null are valid configured selectors."""
    update_settings(env, chat_id=chat_id)
    assert dump_chat.run_cli([]) == 1
    assert "chat_id" in capsys.readouterr().err
    env.constructor.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        TelegramNotConnectedError("fixture disconnected"),
        TelegramFloodWaitError(9),
        TelegramAccessError("fixture access denied"),
        ConfigError("fixture configuration failure"),
    ],
)
def test_connect_domain_errors_exit_one_and_cleanup(
    env: SimpleNamespace, error: Exception, capsys: pytest.CaptureFixture[str]
) -> None:
    """Connection failures are readable CLI errors, not unhandled async tracebacks."""
    env.wrapper.connect.side_effect = error
    assert dump_chat.run_cli([]) == 1
    captured = capsys.readouterr()
    assert str(error) in captured.err
    assert "Traceback" not in captured.err
    env.wrapper.aclose.assert_awaited_once()
    env.wrapper.list_recent_messages.assert_not_awaited()


def test_second_flood_wait_is_not_retried_again(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second FloodWait terminates the dump rather than entering a retry loop."""
    env.wrapper.list_recent_messages.side_effect = [
        TelegramFloodWaitError(2),
        TelegramFloodWaitError(3),
    ]
    assert dump_chat.run_cli([]) == 1
    assert env.wrapper.list_recent_messages.await_count == 2
    env.sleep.assert_awaited_once_with(2)
    assert "retry after 3s" in capsys.readouterr().err
    env.wrapper.aclose.assert_awaited_once()


def test_unexpected_error_prints_traceback_and_closes(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unexpected failures retain their traceback on stderr and still return failure."""
    env.wrapper.list_recent_messages.side_effect = RuntimeError("fixture unexpected failure")
    assert dump_chat.run_cli([]) == 1
    stderr = capsys.readouterr().err
    assert "Traceback" in stderr
    assert "RuntimeError: fixture unexpected failure" in stderr
    env.wrapper.aclose.assert_awaited_once()


def test_write_failure_is_nonzero_and_closes(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """Filesystem write failures follow the generic error path and release the wrapper."""
    blocked = env.root / "not-a-directory"
    blocked.write_text("fixture", encoding="utf-8")
    assert dump_chat.run_cli(["--output", str(blocked / "dump.json")]) == 1
    assert "Traceback" in capsys.readouterr().err
    env.wrapper.aclose.assert_awaited_once()


@pytest.mark.parametrize("anonymize", [True, False])
def test_raw_outside_data_warning_and_fixture_notice(
    env: SimpleNamespace, anonymize: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """Path warnings use resolved paths and fixture advice applies to either privacy mode."""
    output = Path("tests/fixtures/synthetic.json")
    argv = ["--output", str(output), "--anonymize" if anonymize else "--no-anonymize"]
    assert dump_chat.run_cli(argv) == 0
    stderr = capsys.readouterr().err
    assert "INFO: place for test fixtures. Verify manually there is no PII" in stderr
    assert ("WARNING" in stderr) is (not anonymize)

    escaping = Path("data/../outside.json")
    assert dump_chat.run_cli(["--output", str(escaping), "--no-anonymize"]) == 0
    assert (
        "WARNING: output is outside data/ AND not anonymized — DO NOT COMMIT"
        in capsys.readouterr().err
    )


async def test_default_output_unicode_reply_and_utc_timestamps(
    env: SimpleNamespace, tmp_path: Path
) -> None:
    """The default file is under data/dumps and JSON preserves Unicode and reply IDs."""
    local_date = datetime(2026, 10, 31, 15, 30, tzinfo=timezone(timedelta(hours=3)))
    env.wrapper.list_recent_messages.return_value = [
        make_message(7, BOT_ID, text="Игра начинается 🎲", date=local_date, reply_to_msg_id=6),
        make_message(6, text="готов к игре"),
    ]

    await dump_chat.main([])

    output = next((tmp_path / "data" / "dumps").glob("chat_*.json"))
    raw = output.read_text(encoding="utf-8")
    assert "Игра начинается 🎲" in raw
    assert '\n  "messages": [' in raw
    payload = read_json(output)
    assert set(payload) == {
        "chat_id",
        "chat_title",
        "dump_at",
        "message_count",
        "anonymized",
        "bot_id",
        "bot_name",
        "messages",
    }
    assert payload["message_count"] == 2
    assert payload["dump_at"].endswith("Z")
    assert datetime.fromisoformat(payload["dump_at"]).tzinfo == UTC
    assert payload["messages"][0]["reply_to_msg_id"] is None
    assert payload["messages"][1]["reply_to_msg_id"] == 6
    assert payload["messages"][1]["ts"] == "2026-10-31T12:30:00Z"
    assert set(payload["messages"][0]) == {
        "ts",
        "sender_id",
        "sender_name",
        "text",
        "reply_to_msg_id",
    }


@pytest.mark.parametrize("naive", [True, False])
async def test_naive_and_offset_timestamps_normalize_to_utc(
    env: SimpleNamespace, tmp_path: Path, naive: bool
) -> None:
    """Telethon UTC timestamps and offset-aware values serialize consistently."""
    date = datetime(2026, 10, 31, 12, 0)
    if not naive:
        date = date.replace(tzinfo=UTC)
    env.wrapper.list_recent_messages.return_value = [make_message(1, date=date)]
    output = tmp_path / "timestamp.json"
    await dump_chat.main(["--output", str(output)])
    assert read_json(output)["messages"][0]["ts"] == "2026-10-31T12:00:00Z"


async def test_empty_media_and_unknown_sender_have_stable_representation(
    env: SimpleNamespace, tmp_path: Path
) -> None:
    """Media/service-like records retain empty text and unknown senders still get aliases."""
    media = make_message(2, BOT_ID, text="")
    media.message = None
    media.text = None
    anonymous = make_message(1)
    anonymous.sender_id = None
    anonymous.from_id = None
    anonymous.sender = None
    anonymous.post_author = "Fixture signature"
    env.wrapper.list_recent_messages.return_value = [media, anonymous]
    output = tmp_path / "media.json"

    await dump_chat.main(["--no-anonymize", "--output", str(output)])
    messages = read_json(output)["messages"]
    assert messages[0] == {
        "ts": "2026-10-31T12:00:01Z",
        "sender_id": 0,
        "sender_name": "Fixture signature",
        "text": "сообщение 1",
        "reply_to_msg_id": None,
    }
    assert messages[1]["text"] == ""  # Media without a caption is retained.

    await dump_chat.main(["--output", str(output)])
    assert read_json(output)["messages"][0]["sender_id"] == "user_1"


async def test_peer_user_fallback_and_missing_names(env: SimpleNamespace, tmp_path: Path) -> None:
    """PeerUser IDs work if sender_id is absent; absent display names remain empty."""
    message = make_message(1)
    message.sender_id = None
    message.from_id = SimpleNamespace(user_id=PLAYER_IDS[1])
    message.sender = None
    message.reply_to = None
    message.reply_to_msg_id = 10
    env.wrapper.list_recent_messages.return_value = [message]
    output = tmp_path / "fallback.json"

    await dump_chat.main(["--no-anonymize", "--output", str(output)])

    assert read_json(output)["messages"][0] == {
        "ts": "2026-10-31T12:00:01Z",
        "sender_id": PLAYER_IDS[1],
        "sender_name": "",
        "text": "сообщение 1",
        "reply_to_msg_id": 10,
    }


@pytest.mark.parametrize(
    "entity",
    [None, SimpleNamespace(username="FixtureName"), SimpleNamespace(title="Fixture Title")],
)
def test_entity_name_fallbacks(entity: Any) -> None:
    """Name extraction uses available metadata without calling get_sender or get_entity."""
    expected = "" if entity is None else getattr(entity, "title", None) or entity.username
    assert dump_chat._entity_name(entity) == expected


def test_invalid_message_timestamp_is_a_domain_error(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed message cannot silently receive a fabricated timestamp."""
    message = make_message(1)
    message.date = None
    env.wrapper.list_recent_messages.return_value = [message]
    assert dump_chat.run_cli([]) == 1
    assert "without a valid timestamp" in capsys.readouterr().err
    env.wrapper.aclose.assert_awaited_once()


def test_script_has_no_mutating_calls_or_application_imports() -> None:
    """The script's own AST stays read-only and independent of app/features/Telethon."""
    source = Path(__file__).resolve().parents[3] / "scripts" / "dump_chat.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = {
        "send_message",
        "send_file",
        "send_photo",
        "send_document",
        "forward_messages",
        "vote",
        "send_reaction",
        "edit_message",
        "delete_messages",
        "join_channel",
        "leave_channel",
        "get_entity",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            assert name not in forbidden
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"app", "features", "telethon"}
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "telethon" for alias in node.names)
