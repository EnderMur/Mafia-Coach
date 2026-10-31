"""Unit tests for TelegramConfig and TelegramClientWrapper using fake clients."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger
from pydantic import ValidationError
from telethon import events  # type: ignore[import-untyped]
from telethon.errors import RPCError  # type: ignore[import-untyped]
from telethon.tl.types import Chat, User  # type: ignore[import-untyped]

from domain.exceptions import ConfigError, DomainError, TelegramNotConnectedError
from infra.telegram.client import TelegramClientWrapper, TelegramConfig

API_ID = 123456
API_HASH = "0" * 32
ME_ID = 777001
BOT_ID = 888002
CHAT_ID = -1001234567890


def make_session_file(tmp_path: Path, name: str = "mafia_coach") -> str:
    """Create a dummy .session file in tmp_path and return the base session path."""
    (tmp_path / f"{name}.session").write_bytes(b"")
    return str(tmp_path / name)


def make_config(tmp_path: Path, **overrides: Any) -> TelegramConfig:
    """Create a valid TelegramConfig backed by a dummy session file."""
    data: dict[str, Any] = {
        "session_file": make_session_file(tmp_path),
        "api_id": API_ID,
        "api_hash": API_HASH,
        "chat_id": CHAT_ID,
        "bot_id": BOT_ID,
        "ignore_own_messages": True,
    }
    data.update(overrides)
    return TelegramConfig(**data)


def make_fake_client(
    *,
    authorized: bool = True,
    me: Any = None,
    entity: Any = None,
    messages: list[Any] | None = None,
) -> AsyncMock:
    """Build a mocked Telethon client without touching any network or real session."""
    fake = AsyncMock()
    fake.connect = AsyncMock(return_value=None)
    fake.is_user_authorized = AsyncMock(return_value=authorized)
    fake.disconnect = AsyncMock(return_value=None)
    fake.get_me = AsyncMock(return_value=User(id=ME_ID) if me is None else me)
    fake.get_entity = AsyncMock(return_value=User(id=BOT_ID) if entity is None else entity)
    fake.add_event_handler = MagicMock()

    items = list(messages or [])

    async def _iter(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        for item in items:
            yield item

    fake.iter_messages = MagicMock(side_effect=_iter)
    return fake


def test_telegram_error_hierarchy() -> None:
    """TelegramNotConnectedError inherits from DomainError."""
    assert issubclass(TelegramNotConnectedError, DomainError)


def test_init_missing_session_raises_config_error(tmp_path: Path) -> None:
    """1. Constructor raises ConfigError when the session file does not exist."""
    config = TelegramConfig(
        session_file=str(tmp_path / "nonexistent"),
        api_id=API_ID,
        api_hash=API_HASH,
        bot_id=BOT_ID,
    )
    with pytest.raises(ConfigError, match="session file missing"):
        TelegramClientWrapper(config=config, client=make_fake_client())


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"api_id": 0},
        {"session_file": "   "},
        {"api_hash": " " * 16},
    ],
)
def test_config_rejects_invalid_fields(tmp_path: Path, invalid_kwargs: dict[str, Any]) -> None:
    """2. TelegramConfig rejects non-positive api_id and blank credentials."""
    session_file = make_session_file(tmp_path)
    base: dict[str, Any] = {
        "session_file": session_file,
        "api_id": API_ID,
        "api_hash": API_HASH,
        "bot_id": BOT_ID,
    }
    base.update(invalid_kwargs)
    with pytest.raises(ValidationError):
        TelegramConfig(**base)


@pytest.mark.parametrize("bot_username", [None, "", "   "])
def test_config_requires_bot_identifier(tmp_path: Path, bot_username: str | None) -> None:
    """3. TelegramConfig requires either bot_username or bot_id."""
    session_file = make_session_file(tmp_path)
    with pytest.raises(ValidationError, match="bot_username or telegram.bot_id required"):
        TelegramConfig(
            session_file=session_file,
            api_id=API_ID,
            api_hash=API_HASH,
            bot_username=bot_username,
            bot_id=None,
        )


def test_init_with_injected_client_sets_owns_client_false(tmp_path: Path) -> None:
    """4. Valid config with an injected fake client sets _owns_client=False."""
    config = make_config(tmp_path)
    fake = make_fake_client()
    wrapper = TelegramClientWrapper(config=config, client=fake)
    assert wrapper._owns_client is False
    assert wrapper.config == config
    assert wrapper.is_connected is False
    assert wrapper.me_id is None
    assert wrapper.bot_id is None


async def test_connect_authorized_marks_connected_and_logs(tmp_path: Path) -> None:
    """5. connect() marks wrapper connected, resolves IDs, and logs to telegram channel."""
    config = make_config(tmp_path, bot_id=None, bot_username="@TrueMafiaBot")
    fake = make_fake_client(authorized=True, entity=User(id=BOT_ID))
    wrapper = TelegramClientWrapper(config=config, client=fake)

    records: list[tuple[str, str | None]] = []
    sink_id = logger.add(
        lambda msg: records.append((msg.record["message"], msg.record["extra"].get("channel"))),
        level="INFO",
    )
    try:
        await wrapper.connect()
    finally:
        logger.remove(sink_id)

    assert wrapper.is_connected is True
    assert wrapper.me_id == ME_ID
    assert wrapper.bot_id == BOT_ID
    assert ("connected to Telegram (read-only)", "telegram") in records


async def test_connect_unauthorized_raises_not_connected(tmp_path: Path) -> None:
    """6. connect() raises TelegramNotConnectedError when session is unauthorized."""
    config = make_config(tmp_path)
    fake = make_fake_client(authorized=False)
    wrapper = TelegramClientWrapper(config=config, client=fake)

    with pytest.raises(TelegramNotConnectedError, match="not authorized"):
        await wrapper.connect()
    assert wrapper.is_connected is False


async def test_connect_is_idempotent(tmp_path: Path) -> None:
    """7. Calling connect() twice connects the underlying client only once."""
    config = make_config(tmp_path)
    fake = make_fake_client()
    wrapper = TelegramClientWrapper(config=config, client=fake)

    await wrapper.connect()
    await wrapper.connect()
    fake.connect.assert_awaited_once()


@pytest.mark.parametrize(
    "entity_or_error",
    [
        RPCError(None, "USERNAME_NOT_OCCUPIED"),
        ValueError("Cannot find any entity"),
        Chat(id=555, title="NotABot", photo=None, participants_count=3, date=None, version=1),
    ],
)
async def test_connect_raises_config_error_when_bot_unresolvable(
    tmp_path: Path, entity_or_error: Any
) -> None:
    """8. connect() raises ConfigError when bot_username cannot resolve to a User."""
    config = make_config(tmp_path, bot_id=None, bot_username="@BadBot")
    fake = make_fake_client()
    if isinstance(entity_or_error, Exception):
        fake.get_entity = AsyncMock(side_effect=entity_or_error)
    else:
        fake.get_entity = AsyncMock(return_value=entity_or_error)

    wrapper = TelegramClientWrapper(config=config, client=fake)
    with pytest.raises(ConfigError, match="cannot resolve bot_username/bot_id"):
        await wrapper.connect()
    assert wrapper.is_connected is False


async def test_connect_handles_transport_rpc_error_and_me_rpc_error(tmp_path: Path) -> None:
    """connect() wraps connect RPCError and tolerates get_me failure cleanly."""
    config = make_config(tmp_path)
    failing_connect = make_fake_client()
    failing_connect.connect = AsyncMock(side_effect=RPCError(None, "NETWORK_DOWN"))
    wrapper_fail = TelegramClientWrapper(config=config, client=failing_connect)
    with pytest.raises(TelegramNotConnectedError, match="failed to connect"):
        await wrapper_fail.connect()

    me_rpc_fail = make_fake_client()
    me_rpc_fail.get_me = AsyncMock(side_effect=RPCError(None, "ME_FAILED"))
    wrapper_ok = TelegramClientWrapper(config=config, client=me_rpc_fail)
    await wrapper_ok.connect()
    assert wrapper_ok.is_connected is True
    assert wrapper_ok.me_id is None

    me_none = make_fake_client()
    me_none.get_me = AsyncMock(return_value=None)
    wrapper_none = TelegramClientWrapper(config=config, client=me_none)
    await wrapper_none.connect()
    assert wrapper_none.me_id is None


async def test_disconnect_closes_connected_client(tmp_path: Path) -> None:
    """9. disconnect() disconnects a connected client and is a no-op afterwards."""
    config = make_config(tmp_path)
    fake = make_fake_client()
    wrapper = TelegramClientWrapper(config=config, client=fake)

    await wrapper.connect()
    await wrapper.disconnect()
    await wrapper.disconnect()

    fake.disconnect.assert_awaited_once()
    assert wrapper.is_connected is False


def test_add_message_handler_before_connect_raises(tmp_path: Path) -> None:
    """10. add_message_handler() before connect() raises TelegramNotConnectedError."""
    config = make_config(tmp_path)
    wrapper = TelegramClientWrapper(config=config, client=make_fake_client())

    async def handler(_event: Any) -> None:
        return None

    with pytest.raises(TelegramNotConnectedError):
        wrapper.add_message_handler(handler, chat_id=CHAT_ID)


async def test_add_message_handler_registers_new_message_filter(tmp_path: Path) -> None:
    """11. add_message_handler() registers handler with events.NewMessage filter."""
    config = make_config(tmp_path, ignore_own_messages=True)
    fake = make_fake_client()
    wrapper = TelegramClientWrapper(config=config, client=fake)
    await wrapper.connect()

    async def handler(_event: Any) -> None:
        return None

    wrapper.add_message_handler(handler, chat_id=CHAT_ID)
    fake.add_event_handler.assert_called_once()
    registered_handler, builder = fake.add_event_handler.call_args.args
    assert registered_handler is handler
    assert isinstance(builder, events.NewMessage)
    assert builder.chats == [CHAT_ID]
    assert builder.incoming is True
    assert builder.func is not None
    assert builder.func(SimpleNamespace(out=True, sender_id=999)) is False
    assert builder.func(SimpleNamespace(out=False, sender_id=ME_ID)) is False
    assert builder.func(SimpleNamespace(out=False, sender_id=BOT_ID)) is True

    wrapper.add_message_handler(handler, chat_id=CHAT_ID, bot_only=True)
    _, bot_builder = fake.add_event_handler.call_args.args
    assert isinstance(bot_builder, events.NewMessage)
    assert bot_builder.from_users == [BOT_ID]


async def test_add_message_handler_allows_own_messages_when_configured(tmp_path: Path) -> None:
    """add_message_handler() omits own-message exclusion when ignore_own_messages=False."""
    config = make_config(tmp_path, ignore_own_messages=False)
    fake = make_fake_client()
    wrapper = TelegramClientWrapper(config=config, client=fake)
    await wrapper.connect()

    async def handler(_event: Any) -> None:
        return None

    wrapper.add_message_handler(handler, chat_id=CHAT_ID)
    _, builder = fake.add_event_handler.call_args.args
    assert isinstance(builder, events.NewMessage)
    assert builder.incoming is None
    assert builder.func is None


async def test_get_chat_resolves_entity_and_handles_errors(tmp_path: Path) -> None:
    """get_chat() requires connect(), returns entity, and maps errors to ConfigError."""
    config = make_config(tmp_path)
    fake = make_fake_client()
    wrapper = TelegramClientWrapper(config=config, client=fake)

    with pytest.raises(TelegramNotConnectedError):
        await wrapper.get_chat(CHAT_ID)

    await wrapper.connect()
    target_chat = SimpleNamespace(id=CHAT_ID, title="Mafia Club")
    fake.get_entity = AsyncMock(return_value=target_chat)
    resolved = await wrapper.get_chat(CHAT_ID)
    assert resolved is target_chat

    fake.get_entity = AsyncMock(side_effect=RPCError(None, "CHANNEL_PRIVATE"))
    with pytest.raises(ConfigError, match="cannot resolve chat"):
        await wrapper.get_chat(CHAT_ID)


async def test_list_recent_messages_before_connect_raises(tmp_path: Path) -> None:
    """12. list_recent_messages() before connect() raises TelegramNotConnectedError."""
    config = make_config(tmp_path)
    wrapper = TelegramClientWrapper(config=config, client=make_fake_client())

    with pytest.raises(TelegramNotConnectedError):
        await wrapper.list_recent_messages(chat_id=CHAT_ID)


async def test_list_recent_messages_returns_messages_and_handles_edge_cases(
    tmp_path: Path,
) -> None:
    """13. list_recent_messages() returns items from fake client and handles RPCError."""
    sample = [SimpleNamespace(id=1, text="hello"), SimpleNamespace(id=2, text="world")]
    config = make_config(tmp_path)
    fake = make_fake_client(messages=sample)
    wrapper = TelegramClientWrapper(config=config, client=fake)
    await wrapper.connect()

    result = await wrapper.list_recent_messages(chat_id=CHAT_ID, limit=10)
    assert result == sample
    fake.iter_messages.assert_called_once_with(CHAT_ID, limit=10)

    assert await wrapper.list_recent_messages(chat_id=CHAT_ID, limit=0) == []

    async def _failing_iter(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        raise RPCError(None, "CHAT_ID_INVALID")
        yield  # pragma: no cover

    fake.iter_messages = MagicMock(side_effect=_failing_iter)
    with pytest.raises(TelegramNotConnectedError, match="failed to list recent messages"):
        await wrapper.list_recent_messages(chat_id=CHAT_ID, limit=5)


async def test_aclose_does_not_disconnect_external_client(tmp_path: Path) -> None:
    """14. aclose() does not close an externally injected client (_owns_client=False)."""
    config = make_config(tmp_path)
    fake = make_fake_client()
    wrapper = TelegramClientWrapper(config=config, client=fake)
    await wrapper.connect()

    await wrapper.aclose()
    fake.disconnect.assert_not_called()
    assert wrapper.is_connected is False


async def test_aclose_disconnects_owned_client_idempotently(tmp_path: Path) -> None:
    """15. aclose() closes an internally owned client once (_owns_client=True)."""
    config = make_config(tmp_path)
    owned_fake = make_fake_client()
    owned_sync_disconnect = make_fake_client()
    owned_sync_disconnect.disconnect = MagicMock(return_value=None)

    with patch("infra.telegram.client.TelegramClient", return_value=owned_fake) as ctor:
        wrapper = TelegramClientWrapper(config=config)
        ctor.assert_called_once_with(
            session=str(config.session_file),
            api_id=config.api_id,
            api_hash=config.api_hash,
        )
        assert wrapper._owns_client is True
        await wrapper.connect()
        await wrapper.aclose()
        await wrapper.aclose()
        owned_fake.disconnect.assert_awaited_once()

    with patch("infra.telegram.client.TelegramClient", return_value=owned_sync_disconnect):
        wrapper2 = TelegramClientWrapper(config=config)
        await wrapper2.aclose()
        owned_sync_disconnect.disconnect.assert_called_once()
