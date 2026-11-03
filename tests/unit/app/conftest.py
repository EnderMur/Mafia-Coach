"""Fakes and settings helpers shared by application unit tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from infra.llm.client import LLMClient, ReasoningSink
from infra.llm.schemas import AnalysisResponse
from infra.telegram.client import TelegramConfig

ROOT = Path(__file__).resolve().parents[3]


class FakeLLMClient(LLMClient):
    """Fake streaming LLM client with a configurable response and call counter."""

    def __init__(self, response: AnalysisResponse | None = None) -> None:
        self.response = response or AnalysisResponse(
            analysis=[],
            game_read="Текущая картина партии.",
            reasoning="Краткий разбор.",
        )
        self.calls = 0
        self.aclose_calls = 0

    async def healthcheck(self) -> None:
        """Implement a no-op fake health check."""

    async def analyze(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
    ) -> AnalysisResponse:
        """Delegate to the fake streaming implementation."""
        return await self.analyze_stream(
            messages=messages,
            known_player_ids=known_player_ids,
            role_names_ru=role_names_ru,
            role_alias_resolver=role_alias_resolver,
        )

    async def analyze_stream(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        on_reasoning_chunk: ReasoningSink | None = None,
        log_reasoning_chunks: bool = True,
    ) -> AnalysisResponse:
        """Return the canned response and record the model invocation."""
        self.calls += 1
        if on_reasoning_chunk is not None:
            on_reasoning_chunk("Проверяю игровые события.")
        return self.response

    async def aclose(self) -> None:
        """Record that the fake transport was closed."""
        self.aclose_calls += 1


class FakeTelegram:
    """Offline stand-in for the subset of Telegram used by the app/scanner."""

    def __init__(
        self,
        *,
        dialogs: list[dict[str, object]] | None = None,
        get_chat_error: Exception | None = None,
    ) -> None:
        self._config = TelegramConfig(
            session_file="unused-test-session",
            api_id=1,
            api_hash="a" * 32,
            bot_id=42,
        )
        self._me_id: int | None = 7
        self._bot_id: int | None = 42
        self._dialogs = (
            dialogs
            if dialogs is not None
            else [
                {"id": -1001, "title": "Игровой чат 1", "kind": "group"},
                {"id": -1002, "title": "Игровой чат 2", "kind": "group"},
                {"id": -1003, "title": "Игровой чат 3", "kind": "channel"},
            ]
        )
        self.get_chat_error = get_chat_error
        self.get_chat_calls: list[int | str] = []
        self.iter_dialogs_calls: list[dict[str, object]] = []
        self.handler: Callable[[Any], Awaitable[None]] | None = None
        self.handler_chat_id: int | None = None
        self.connected = False
        self.aclose_calls = 0
        self.disconnect_calls = 0
        self.remove_handlers_calls = 0

    @property
    def config(self) -> TelegramConfig:
        """Return the fake client's immutable configuration."""
        return self._config

    @property
    def me_id(self) -> int | None:
        """Return the fake userbot id."""
        return self._me_id

    @property
    def bot_id(self) -> int | None:
        """Return the fake game-bot id."""
        return self._bot_id

    async def connect(self) -> None:
        """Mark the fake client connected."""
        self.connected = True

    async def disconnect(self) -> None:
        """Mark the fake client disconnected."""
        self.connected = False
        self.disconnect_calls += 1

    async def get_chat(self, chat_ref: int | str) -> Any:
        """Return a lightweight chat or raise the configured failure."""
        self.get_chat_calls.append(chat_ref)
        if self.get_chat_error is not None:
            raise self.get_chat_error
        return SimpleNamespace(id=chat_ref, title="test chat")

    async def iter_dialogs(
        self,
        *,
        dialogs_limit: int = 200,
        groups_only: bool = True,
    ) -> list[dict[str, object]]:
        """Return fake dialogs and record requested listing options."""
        self.iter_dialogs_calls.append({"dialogs_limit": dialogs_limit, "groups_only": groups_only})
        dialogs = self._dialogs[:dialogs_limit]
        if groups_only:
            dialogs = [dialog for dialog in dialogs if dialog.get("kind") in {"group", "channel"}]
        return dialogs

    def add_message_handler(
        self,
        handler: Callable[[Any], Awaitable[None]],
        *,
        chat_id: int,
        bot_only: bool = False,
        deduplicate: bool = True,
    ) -> None:
        """Save the scanner handler without registering any network callback."""
        self.handler = handler
        self.handler_chat_id = chat_id

    async def remove_all_handlers(self) -> None:
        """Forget any callback retained by the fake client."""
        self.handler = None
        self.remove_handlers_calls += 1

    async def list_recent_messages(
        self,
        *,
        chat_id: int,
        limit: int = 50,
    ) -> list[Any]:
        """Return no historical messages."""
        return []

    async def aclose(self) -> None:
        """Record closure and mark the fake disconnected."""
        self.aclose_calls += 1
        self.connected = False


def create_settings_file(
    tmp_path: Path,
    *,
    chat_id: int | None = -100123,
    trigger_mode: str = "queue",
    db_path: str | None = None,
) -> Path:
    """Write a valid temporary settings file derived from the example YAML."""
    example = ROOT / "configs" / "settings.example.yaml"
    settings = yaml.safe_load(example.read_text(encoding="utf-8"))
    settings["telegram"].update(
        {
            "api_id": 12345,
            "api_hash": "0123456789abcdef0123456789abcdef",
            "chat_id": chat_id,
            "bot_username": "@FakeMafiaBot",
            "bot_id": 42,
        }
    )
    settings["llm"]["trigger_mode"] = trigger_mode
    settings["db"]["path"] = db_path or str(tmp_path / "mafia.db")
    settings["roles_file"] = str(ROOT / "configs" / "roles.yaml")
    # Never read a developer's private club file when bootstrapping tests.
    settings["club_patterns_file"] = str(tmp_path / "patterns.txt")
    settings["logging"]["dir"] = str(tmp_path / "logs")

    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        yaml.safe_dump(settings, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path
