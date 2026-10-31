"""Static AST and surface read-only audit tests for infra/telegram."""

from __future__ import annotations

import ast
from pathlib import Path

from infra.telegram.client import TelegramClientWrapper

FORBIDDEN_METHODS = frozenset(
    {
        "send_message",
        "send_file",
        "send_photo",
        "send_document",
        "send_audio",
        "send_video",
        "send_voice",
        "send_video_note",
        "send_sticker",
        "send_gif",
        "forward_messages",
        "vote",
        "send_reaction",
        "edit_message",
        "edit_admin",
        "delete_messages",
        "delete_dialog",
        "pin_message",
        "unpin_message",
        "join_channel",
        "leave_channel",
        "invite_to_channel",
        "kick_participant",
        "edit_2fa",
        "log_out",
        "sign_in",
    }
)


def _scan_forbidden_calls(path: Path) -> list[tuple[int, str]]:
    """Parse AST and return (lineno, attr) pairs for any forbidden method calls."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_METHODS:
                bad.append((node.lineno, func.attr))
    return bad


def test_no_forbidden_calls_in_infra_telegram() -> None:
    """No Python module in infra/telegram invokes any mutating Telethon method."""
    for path in Path("infra/telegram").rglob("*.py"):
        bad = _scan_forbidden_calls(path)
        assert not bad, f"{path}: forbidden calls {bad}"


def test_readonly_surface_on_wrapper() -> None:
    """Public attributes of TelegramClientWrapper do not include forbidden methods."""
    public = {name for name in dir(TelegramClientWrapper) if not name.startswith("_")}
    overlap = public & FORBIDDEN_METHODS
    assert not overlap, f"wrapper exposes forbidden methods: {overlap}"


def test_no_send_tokens_in_source() -> None:
    """Source files in infra/telegram contain no '.<forbidden>(' call tokens."""
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in Path("infra/telegram").rglob("*.py")
    )
    for token in FORBIDDEN_METHODS:
        assert f".{token}(" not in text, f"forbidden call token: .{token}("


def test_events_only_newmessage() -> None:
    """Only events.NewMessage is parsed; deleted, chat, and user events are absent."""
    text = Path("infra/telegram/client.py").read_text(encoding="utf-8")
    assert "events.NewMessage" in text
    for forbidden_event in (
        "events.MessageDeleted",
        "events.ChatAction",
        "events.UserUpdate",
    ):
        assert forbidden_event not in text, f"forbidden event: {forbidden_event}"


def test_edited_message_only_for_warning() -> None:
    """events.MessageEdited may appear only as a warning-log watcher, never as parser input."""
    import re

    text = Path("infra/telegram/client.py").read_text(encoding="utf-8")
    matches = list(re.finditer(r"events\.MessageEdited", text))
    assert matches, "expected a log-only handler for edited messages"
    for match in matches:
        snippet = text[match.start() : match.start() + 400]
        lowered = snippet.lower()
        assert (
            "warn" in lowered or "warning" in lowered
        ), "events.MessageEdited without warning context"
        # The edited event must never be handed to a user handler directly.
        assert "self._handlers" not in snippet


def test_readonly_docstring_present() -> None:
    """TelegramClientWrapper docstring explicitly documents its read-only contract."""
    assert TelegramClientWrapper.__doc__ is not None
    doc_lower = TelegramClientWrapper.__doc__.lower()
    assert "read-only" in doc_lower or "read only" in doc_lower
