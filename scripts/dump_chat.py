"""Export Telegram chat history as JSON without sending or modifying anything.

All network reads go through TelegramClientWrapper. Anonymization replaces player
identifiers and omits the chat title, but does not redact names or other PII in
message texts. Review every dump manually before committing it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from domain.exceptions import (
    ConfigError,
    TelegramAccessError,
    TelegramFloodWaitError,
    TelegramNotConnectedError,
)
from infra.logging.setup import get_logger, setup_logging
from infra.telegram.client import TelegramClientWrapper, TelegramConfig
from infra.telegram.session import session_exists

_DEFAULT_LIMIT = 10_000
_SESSION_MISSING = "session file missing. Run: python scripts/make_session.py"
_CHAT_MISSING = "chat is not configured: укажи --chat-id или заполни settings.telegram.chat_id"


def _positive_int(value: str) -> int:
    """Parse a strictly positive CLI integer."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse dump options; anonymization is enabled unless explicitly disabled."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    parser = argparse.ArgumentParser(
        description="Read-only Telegram history dump. Never sends messages or votes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    chat = parser.add_mutually_exclusive_group()
    chat.add_argument("--chat-id", type=int, help="Numeric chat ID (overrides settings)")
    chat.add_argument("--chat-username", help="Chat @username (overrides settings)")
    parser.add_argument("--limit", type=_positive_int, default=_DEFAULT_LIMIT, help="Message limit")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"data/dumps/chat_{timestamp}.json"),
        help="Destination JSON file",
    )
    privacy = parser.add_mutually_exclusive_group()
    privacy.add_argument(
        "--anonymize", dest="anonymize", action="store_true", help="Replace player IDs/names"
    )
    privacy.add_argument(
        "--no-anonymize", dest="anonymize", action="store_false", help="Keep player IDs/names"
    )
    parser.set_defaults(anonymize=True)
    parser.add_argument("--bot-id", type=_positive_int, help="Game bot ID (overrides settings)")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/settings.yaml"), help="Settings YAML path"
    )
    return parser.parse_args(argv)


def _load_telegram_settings(path: Path) -> dict[str, Any]:
    """Load the Telegram settings mapping without falling back to example credentials."""
    try:
        loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read settings {path}: {exc}") from exc
    if not isinstance(loaded, dict) or not isinstance(loaded.get("telegram"), dict):
        raise ConfigError(f"settings must contain a telegram mapping: {path}")
    telegram: dict[str, Any] = loaded["telegram"]
    return telegram


def _chat_reference(args: argparse.Namespace, telegram: dict[str, Any]) -> int | str:
    """Choose the CLI chat reference, falling back to an ID or username in settings."""
    if args.chat_id is not None:
        return int(args.chat_id)
    if args.chat_username is not None:
        return str(args.chat_username)
    configured = telegram.get("chat_id")
    if isinstance(configured, int) and not isinstance(configured, bool):
        return configured
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if configured is None or configured == "":
        raise ConfigError(_CHAT_MISSING)
    raise ConfigError("settings.telegram.chat_id must be an integer, a username, or null")


def _make_config(
    args: argparse.Namespace, telegram: dict[str, Any], chat_ref: int | str
) -> TelegramConfig:
    """Validate wrapper settings, leaving username-based chat resolution until connect."""
    session_file = telegram.get("session_file")
    if not isinstance(session_file, str) or not session_file.strip():
        raise ConfigError("settings.telegram.session_file must be a non-empty path")
    if not session_exists(session_file):
        raise ConfigError(_SESSION_MISSING)

    # read_only is documentation in the example YAML; the wrapper is always read-only.
    values = {key: value for key, value in telegram.items() if key != "read_only"}
    values["chat_id"] = chat_ref if isinstance(chat_ref, int) else None
    if args.bot_id is not None:
        values["bot_id"] = args.bot_id
    try:
        return TelegramConfig(**values)
    except ValidationError as exc:
        # Do not echo validation input values: they include API credentials.
        fields = ", ".join(".".join(str(part) for part in error["loc"]) for error in exc.errors())
        raise ConfigError(f"invalid telegram settings ({fields}); check {args.config}") from exc


async def _list_messages(wrapper: TelegramClientWrapper, *, chat_id: int, limit: int) -> list[Any]:
    """Read history, honoring one FloodWait delay and making at most one retry."""
    try:
        return await wrapper.list_recent_messages(chat_id=chat_id, limit=limit)
    except TelegramFloodWaitError as exc:
        seconds = max(0, exc.wait_seconds)
        warning = f"waiting {seconds} seconds (FloodWait), retrying once..."
        get_logger("dump_chat").warning(warning)
        print(f"WARNING: {warning}", file=sys.stderr)
        await asyncio.sleep(seconds)
        return await wrapper.list_recent_messages(chat_id=chat_id, limit=limit)


def _utc_iso(value: datetime) -> str:
    """Normalize a Telegram timestamp to ISO8601 UTC, treating naive values as UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _entity_name(entity: Any) -> str:
    """Read an already-fetched entity's display name without additional network calls."""
    if entity is None:
        return ""
    title = getattr(entity, "title", None)
    if title:
        return str(title)
    name = " ".join(
        str(part)
        for part in (getattr(entity, "first_name", None), getattr(entity, "last_name", None))
        if part
    )
    return name or str(getattr(entity, "username", None) or "")


def _sender_id(message: Any) -> int:
    """Use Telethon's marked sender ID, with a PeerUser fallback and 0 for unknown."""
    sender_id = getattr(message, "sender_id", None)
    if sender_id is None:
        sender_id = getattr(getattr(message, "from_id", None), "user_id", None)
    return int(sender_id) if sender_id is not None else 0


def _message_record(
    message: Any,
    *,
    anonymize: bool,
    aliases: dict[int, str],
    bot_id: int,
    bot_name: str,
) -> dict[str, Any]:
    """Convert one message, assigning player aliases in chronological encounter order."""
    sender_id = _sender_id(message)
    is_bot = sender_id == bot_id
    sender_name = bot_name if is_bot else _entity_name(getattr(message, "sender", None))
    if not sender_name and not is_bot:
        sender_name = str(getattr(message, "post_author", None) or "")
    output_id: int | str = sender_id
    if anonymize and not is_bot:
        if sender_id not in aliases:
            aliases[sender_id] = f"user_{len(aliases) + 1}"
        output_id = sender_name = aliases[sender_id]

    timestamp = getattr(message, "date", None)
    if not isinstance(timestamp, datetime):
        raise ConfigError("Telegram returned a message without a valid timestamp")
    text = getattr(message, "message", None)
    if text is None:
        text = getattr(message, "text", None)
    reply_to = getattr(message, "reply_to", None)
    reply_id = getattr(reply_to, "reply_to_msg_id", None)
    if reply_id is None:
        reply_id = getattr(message, "reply_to_msg_id", None)
    return {
        "ts": _utc_iso(timestamp),
        "sender_id": output_id,
        "sender_name": sender_name,
        "text": str(text or ""),
        "reply_to_msg_id": int(reply_id) if reply_id is not None else None,
    }


def _warn_about_output(output: Path, *, anonymize: bool) -> None:
    """Print privacy reminders, including for raw dumps nominally stored under data/."""
    resolved = output.resolve()
    if not anonymize:
        if resolved.is_relative_to(Path("data").resolve()):
            print(
                "WARNING: dump is not anonymized — DO NOT COMMIT; message texts may contain PII.",
                file=sys.stderr,
            )
        else:
            print(
                "WARNING: output is outside data/ AND not anonymized — DO NOT COMMIT",
                file=sys.stderr,
            )
    if resolved.is_relative_to((Path("tests") / "fixtures").resolve()):
        print(
            "INFO: place for test fixtures. Verify manually there is no PII "
            "in message texts before committing.",
            file=sys.stderr,
        )


async def main(argv: Sequence[str] | None = None) -> None:
    """Fetch, anonymize, and save a chat dump; always close an initialized wrapper."""
    args = parse_args(argv)
    setup_logging(level="WARNING", log_dir="logs")
    telegram = _load_telegram_settings(args.config)
    chat_ref = _chat_reference(args, telegram)
    config = _make_config(args, telegram, chat_ref)

    wrapper: TelegramClientWrapper | None = None
    try:
        wrapper = TelegramClientWrapper(config=config)
        await wrapper.connect()
        bot_id = wrapper.bot_id
        if bot_id is None:
            raise ConfigError("cannot resolve bot_id; check --bot-id or settings.telegram.bot_*")
        chat = await wrapper.get_chat(chat_ref)
        chat_id = chat_ref if isinstance(chat_ref, int) else int(chat.id)
        bot = await wrapper.get_chat(bot_id)
        bot_name = _entity_name(bot) or config.bot_username or ""
        messages = await _list_messages(wrapper, chat_id=chat_id, limit=args.limit)

        aliases: dict[int, str] = {}
        records = [
            _message_record(
                message,
                anonymize=args.anonymize,
                aliases=aliases,
                bot_id=bot_id,
                bot_name=bot_name,
            )
            # Telethon yields newest first; aliases also follow the reversed order.
            for message in reversed(messages)
        ]
        output: Path = args.output
        payload = {
            "chat_id": chat_id,
            # Group titles can contain personal data, so omit them in anonymized dumps.
            "chat_title": "" if args.anonymize else _entity_name(chat),
            "dump_at": _utc_iso(datetime.now(UTC)),
            "message_count": len(records),
            "anonymized": args.anonymize,
            "bot_id": bot_id,
            "bot_name": bot_name,
            "messages": records,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"OK: wrote {len(records)} messages to {output}")
        _warn_about_output(output, anonymize=args.anonymize)
    finally:
        if wrapper is not None:
            await wrapper.aclose()


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Synchronous CLI boundary with readable domain errors and nonzero failure status."""
    try:
        asyncio.run(main(argv))
    except (
        ConfigError,
        TelegramNotConnectedError,
        TelegramFloodWaitError,
        TelegramAccessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_cli())
