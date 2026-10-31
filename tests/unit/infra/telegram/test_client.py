"""Unit tests for TelegramConfig and TelegramClientWrapper using fake clients."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger
from pydantic import ValidationError
from telethon import events  # type: ignore[import-untyped]
from telethon.errors import (  # type: ignore[import-untyped]
    ChannelInvalidError,
    ChannelPrivateError,
    ChatAdminRequiredError,
    FloodWaitError,
    RPCError,
    UserNotParticipantError,
)
from telethon.tl.types import Chat, User  # type: ignore[import-untyped]

from domain.exceptions import (
    ConfigError,
    DomainError,
    TelegramAccessError,
    TelegramFloodWaitError,
    TelegramNotConnectedError,
)
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
    new_message_calls = [
        call.args
        for call in fake.add_event_handler.call_args_list
        if isinstance(call.args[1], events.NewMessage)
    ]
    assert len(new_message_calls) == 1
    registered_handler, builder = new_message_calls[0]
    # The user handler is wrapped by the de-duplicator, not registered directly.
    assert registered_handler is not handler
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


# ---------------------------------------------------------------------------
# step-6b: rate-limit / access mapping, iter_dialogs, dedup, cleanup, edited
# ---------------------------------------------------------------------------


def make_flood(wait: int) -> FloodWaitError:
    """Build a Telethon FloodWaitError carrying ``wait`` seconds without network I/O."""
    return FloodWaitError(request=None, capture=wait)


def make_dialog(
    dialog_id: int,
    title: str = "Chat",
    *,
    is_group: bool = False,
    is_channel: bool = False,
) -> SimpleNamespace:
    """Build a minimal dialog-like object accepted by iter_dialogs()."""
    return SimpleNamespace(id=dialog_id, title=title, is_group=is_group, is_channel=is_channel)


def dialog_iter(items: list[Any]) -> Any:
    """Return a callable producing an async iterator over ``items``."""

    async def _iter(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        for item in items:
            yield item

    return _iter


def failing_dialog_iter(error: Exception) -> Any:
    """Return a callable producing an async iterator that raises ``error``."""

    async def _iter(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        raise error
        yield  # pragma: no cover - keeps this an async generator

    return _iter


async def make_connected_wrapper(tmp_path: Path, **overrides: Any) -> TelegramClientWrapper:
    """Build a connected wrapper backed by a fake client (no real Telethon calls)."""
    config = make_config(tmp_path, **overrides)
    fake = make_fake_client()
    wrapper = TelegramClientWrapper(config=config, client=fake)
    await wrapper.connect()
    return wrapper


async def test_get_chat_flood_wait_maps_to_domain_error(tmp_path: Path) -> None:
    """1. get_chat() maps FloodWaitError to TelegramFloodWaitError with wait_seconds."""
    wrapper = await make_connected_wrapper(tmp_path)
    wrapper._client.get_entity = AsyncMock(side_effect=make_flood(30))

    with pytest.raises(TelegramFloodWaitError) as excinfo:
        await wrapper.get_chat(CHAT_ID)
    assert excinfo.value.wait_seconds == 30
    assert "30" in str(excinfo.value)


async def test_list_recent_messages_flood_wait_maps_to_domain_error(tmp_path: Path) -> None:
    """2. list_recent_messages() maps FloodWaitError to TelegramFloodWaitError."""
    wrapper = await make_connected_wrapper(tmp_path)
    wrapper._client.iter_messages = MagicMock(side_effect=failing_dialog_iter(make_flood(5)))

    with pytest.raises(TelegramFloodWaitError) as excinfo:
        await wrapper.list_recent_messages(chat_id=CHAT_ID, limit=5)
    assert excinfo.value.wait_seconds == 5


async def test_iter_dialogs_flood_wait_maps_to_domain_error(tmp_path: Path) -> None:
    """3. iter_dialogs() maps FloodWaitError to TelegramFloodWaitError."""
    wrapper = await make_connected_wrapper(tmp_path)
    wrapper._client.iter_dialogs = MagicMock(side_effect=failing_dialog_iter(make_flood(60)))

    with pytest.raises(TelegramFloodWaitError) as excinfo:
        await wrapper.iter_dialogs()
    assert excinfo.value.wait_seconds == 60


@pytest.mark.parametrize(
    "error", [ChannelPrivateError(request=None), ChatAdminRequiredError(request=None)]
)
async def test_get_chat_access_errors_map_to_access_error(tmp_path: Path, error: Exception) -> None:
    """4-5. get_chat() maps private-channel and admin-required errors to TelegramAccessError."""
    wrapper = await make_connected_wrapper(tmp_path)
    wrapper._client.get_entity = AsyncMock(side_effect=error)

    with pytest.raises(TelegramAccessError, match="access denied"):
        await wrapper.get_chat(CHAT_ID)


async def test_list_recent_messages_access_error_maps_to_access_error(tmp_path: Path) -> None:
    """6. list_recent_messages() maps UserNotParticipantError to TelegramAccessError."""
    wrapper = await make_connected_wrapper(tmp_path)
    wrapper._client.iter_messages = MagicMock(
        side_effect=failing_dialog_iter(UserNotParticipantError(request=None))
    )

    with pytest.raises(TelegramAccessError, match="UserNotParticipantError"):
        await wrapper.list_recent_messages(chat_id=CHAT_ID, limit=3)


async def test_iter_dialogs_access_error_maps_to_access_error(tmp_path: Path) -> None:
    """iter_dialogs() maps ChannelInvalidError to TelegramAccessError."""
    wrapper = await make_connected_wrapper(tmp_path)
    wrapper._client.iter_dialogs = MagicMock(
        side_effect=failing_dialog_iter(ChannelInvalidError(request=None))
    )

    with pytest.raises(TelegramAccessError, match="ChannelInvalidError"):
        await wrapper.iter_dialogs()


async def test_iter_dialogs_rpc_error_maps_to_config_error(tmp_path: Path) -> None:
    """iter_dialogs() maps a generic RPCError to ConfigError."""
    wrapper = await make_connected_wrapper(tmp_path)
    wrapper._client.iter_dialogs = MagicMock(
        side_effect=failing_dialog_iter(RPCError(None, "FLOOD_PREMIUM"))
    )

    with pytest.raises(ConfigError, match="telegram RPC error"):
        await wrapper.iter_dialogs()


async def test_iter_dialogs_keeps_only_groups_and_channels(tmp_path: Path) -> None:
    """7. iter_dialogs(groups_only=True) drops private chats and bots."""
    wrapper = await make_connected_wrapper(tmp_path)
    wrapper._client.iter_dialogs = MagicMock(
        side_effect=dialog_iter(
            [
                make_dialog(-1001, "Mafia Group", is_group=True),
                make_dialog(-1002, "Mafia Channel", is_channel=True),
                make_dialog(42, "Private chat"),
                make_dialog(43, "TrueMafiaBot", is_group=False),
            ]
        )
    )

    dialogs = await wrapper.iter_dialogs(groups_only=True)
    assert dialogs == [
        {"id": -1001, "title": "Mafia Group", "kind": "group"},
        {"id": -1002, "title": "Mafia Channel", "kind": "channel"},
    ]


async def test_iter_dialogs_groups_only_false_returns_all_kinds(tmp_path: Path) -> None:
    """8. iter_dialogs(groups_only=False) keeps every dialog kind."""
    wrapper = await make_connected_wrapper(tmp_path)
    wrapper._client.iter_dialogs = MagicMock(
        side_effect=dialog_iter(
            [
                make_dialog(-1001, "Mafia Group", is_group=True),
                make_dialog(42, "Private chat"),
            ]
        )
    )

    dialogs = await wrapper.iter_dialogs(groups_only=False)
    assert [d["kind"] for d in dialogs] == ["group", "other"]
    assert dialogs[1]["title"] == "Private chat"


async def test_iter_dialogs_passes_limit_to_client(tmp_path: Path) -> None:
    """9. iter_dialogs() forwards dialogs_limit to the Telethon client."""
    wrapper = await make_connected_wrapper(tmp_path)
    captured: dict[str, Any] = {}

    async def _iter(*_args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        captured.update(kwargs)
        for item in [make_dialog(-1001, "Mafia Group", is_group=True)]:
            yield item

    wrapper._client.iter_dialogs = MagicMock(side_effect=_iter)
    await wrapper.iter_dialogs(dialogs_limit=7)
    assert captured["limit"] == 7


async def test_iter_dialogs_empty_result(tmp_path: Path) -> None:
    """10. iter_dialogs() returns an empty list when Telegram yields nothing."""
    wrapper = await make_connected_wrapper(tmp_path)
    wrapper._client.iter_dialogs = MagicMock(side_effect=dialog_iter([]))

    assert await wrapper.iter_dialogs() == []


async def test_iter_dialogs_before_connect_raises(tmp_path: Path) -> None:
    """11. iter_dialogs() before connect() raises TelegramNotConnectedError."""
    config = make_config(tmp_path)
    wrapper = TelegramClientWrapper(config=config, client=make_fake_client())

    with pytest.raises(TelegramNotConnectedError):
        await wrapper.iter_dialogs()


def make_event(msg_id: int | None, chat_id: int | None) -> SimpleNamespace:
    """Build a minimal NewMessage-like event with the given ids."""
    return SimpleNamespace(message=SimpleNamespace(id=msg_id), chat_id=chat_id)


def last_registered_wrapper(fake: AsyncMock) -> Any:
    """Return the most recently registered NewMessage handler wrapper."""
    calls = [
        call.args
        for call in fake.add_event_handler.call_args_list
        if isinstance(call.args[1], events.NewMessage)
    ]
    return calls[-1][0]


async def test_add_message_handler_deduplicates_by_message_id(tmp_path: Path) -> None:
    """12. duplicate (chat_id, msg_id) events invoke the handler only once."""
    wrapper = await make_connected_wrapper(tmp_path)
    calls: list[Any] = []

    async def handler(event: Any) -> None:
        calls.append(event)

    wrapper.add_message_handler(handler, chat_id=CHAT_ID, deduplicate=True)
    wrapped = last_registered_wrapper(wrapper._client)

    event = make_event(10, CHAT_ID)
    await wrapped(event)
    assert await wrapped(event) is None  # duplicate: handler is NOT called
    assert len(calls) == 1

    await wrapped(make_event(11, CHAT_ID))
    assert len(calls) == 2


async def test_add_message_handler_without_dedup_forwards_duplicates(tmp_path: Path) -> None:
    """13. deduplicate=False forwards every duplicate event to the handler."""
    wrapper = await make_connected_wrapper(tmp_path)
    calls: list[Any] = []

    async def handler(event: Any) -> None:
        calls.append(event)

    wrapper.add_message_handler(handler, chat_id=CHAT_ID, deduplicate=False)
    wrapped = last_registered_wrapper(wrapper._client)

    event = make_event(10, CHAT_ID)
    await wrapped(event)
    await wrapped(event)

    assert len(calls) == 2
    assert wrapper._seen_ids == set()


async def test_dedup_window_is_trimmed_after_limit(tmp_path: Path) -> None:
    """14. the seen-id window is cleared once it grows past 10_000 entries."""
    wrapper = await make_connected_wrapper(tmp_path)

    async def handler(_event: Any) -> None:
        return None

    wrapper.add_message_handler(handler, chat_id=CHAT_ID)
    wrapped = last_registered_wrapper(wrapper._client)

    for msg_id in range(1, 10_001):
        await wrapped(make_event(msg_id, CHAT_ID))
    assert len(wrapper._seen_ids) == 10_000

    await wrapped(make_event(10_001, CHAT_ID))
    assert wrapper._seen_ids == {(CHAT_ID, 10_001)}


async def test_dedup_forwards_events_without_ids(tmp_path: Path) -> None:
    """Events without message id or chat id are forwarded without bookkeeping."""
    wrapper = await make_connected_wrapper(tmp_path)
    calls: list[Any] = []

    async def handler(event: Any) -> None:
        calls.append(event)

    wrapper.add_message_handler(handler, chat_id=CHAT_ID)
    wrapped = last_registered_wrapper(wrapper._client)

    await wrapped(SimpleNamespace(message=None, chat_id=CHAT_ID))
    await wrapped(make_event(5, None))

    assert len(calls) == 2
    assert wrapper._seen_ids == set()


async def test_synchronous_user_handler_is_supported(tmp_path: Path) -> None:
    """A plain synchronous callback is invoked and its None result is tolerated."""
    wrapper = await make_connected_wrapper(tmp_path)
    calls: list[Any] = []

    def handler(event: Any) -> None:
        calls.append(event)

    wrapper.add_message_handler(cast(Any, handler), chat_id=CHAT_ID)
    wrapped = last_registered_wrapper(wrapper._client)

    assert await wrapped(make_event(21, CHAT_ID)) is None
    assert len(calls) == 1


async def test_remove_all_handlers_removes_each_handler(tmp_path: Path) -> None:
    """15. remove_all_handlers() detaches every registered handler exactly once."""
    wrapper = await make_connected_wrapper(tmp_path)
    fake = wrapper._client
    fake.remove_event_handler = MagicMock()

    async def handler(_event: Any) -> None:
        return None

    wrapper.add_message_handler(handler, chat_id=CHAT_ID)
    wrapper.add_message_handler(handler, chat_id=CHAT_ID)
    wrapped_handlers = [wrapped for _user, wrapped in wrapper._handlers]
    edited_handler = wrapper._edited_warn_handler

    await wrapper.remove_all_handlers()
    await wrapper.remove_all_handlers()

    assert wrapper._handlers == []
    assert wrapper._edited_warn_handler is None
    assert wrapper._seen_ids == set()
    removed = [call.args[0] for call in fake.remove_event_handler.call_args_list]
    for expected in wrapped_handlers:
        assert removed.count(expected) == 1
    assert removed.count(edited_handler) == 1


async def test_remove_all_handlers_swallows_cleanup_errors(tmp_path: Path) -> None:
    """remove_all_handlers() ignores errors raised by the client during cleanup."""
    wrapper = await make_connected_wrapper(tmp_path)
    fake = wrapper._client
    fake.remove_event_handler = MagicMock(side_effect=RuntimeError("boom"))

    async def handler(_event: Any) -> None:
        return None

    wrapper.add_message_handler(handler, chat_id=CHAT_ID)
    await wrapper.remove_all_handlers()
    assert wrapper._handlers == []


async def test_remove_all_handlers_awaits_async_removal(tmp_path: Path) -> None:
    """remove_all_handlers() awaits an awaitable remove_event_handler result."""
    wrapper = await make_connected_wrapper(tmp_path)
    fake = wrapper._client
    fake.remove_event_handler = AsyncMock()

    async def handler(_event: Any) -> None:
        return None

    wrapper.add_message_handler(handler, chat_id=CHAT_ID)
    await wrapper.remove_all_handlers()
    assert fake.remove_event_handler.await_count == 2


async def test_remove_all_handlers_works_before_connect(tmp_path: Path) -> None:
    """16. remove_all_handlers() is a safe no-op before connect() is called."""
    config = make_config(tmp_path)
    wrapper = TelegramClientWrapper(config=config, client=make_fake_client())

    await wrapper.remove_all_handlers()
    await wrapper.remove_all_handlers()
    assert wrapper._handlers == []


async def test_connect_registers_edited_message_warning_handler(tmp_path: Path) -> None:
    """17. connect() registers a log-only handler for events.MessageEdited."""
    wrapper = await make_connected_wrapper(tmp_path)
    edited_calls = [
        call.args
        for call in wrapper._client.add_event_handler.call_args_list
        if call.args[1] is events.MessageEdited
    ]

    assert len(edited_calls) == 1
    registered, builder = edited_calls[0]
    assert registered is wrapper._edited_warn_handler
    assert builder is events.MessageEdited


async def test_reconnect_does_not_duplicate_edited_handler(tmp_path: Path) -> None:
    """Reconnecting keeps exactly one MessageEdited warning handler registered."""
    wrapper = await make_connected_wrapper(tmp_path)
    await wrapper.disconnect()
    await wrapper.connect()

    edited_calls = [
        call.args
        for call in wrapper._client.add_event_handler.call_args_list
        if call.args[1] is events.MessageEdited
    ]
    assert len(edited_calls) == 1


async def test_edited_message_handler_logs_warning(tmp_path: Path) -> None:
    """18. the edited-message handler logs a warning and never touches user handlers."""
    wrapper = await make_connected_wrapper(tmp_path)
    handler = wrapper._edited_warn_handler
    assert handler is not None

    records: list[str] = []
    sink_id = logger.add(lambda msg: records.append(msg.record["message"]), level="WARNING")
    try:
        await handler(SimpleNamespace(chat_id=CHAT_ID, message=SimpleNamespace(id=99)))
    finally:
        logger.remove(sink_id)

    assert any("edited message" in record and "99" in record for record in records)


@pytest.mark.parametrize("error", [make_flood(12), ChannelPrivateError(request=None)])
async def test_connect_maps_rpc_errors_to_domain_errors(tmp_path: Path, error: Exception) -> None:
    """connect() maps flood/access RPC failures to the dedicated domain errors."""
    config = make_config(tmp_path)
    fake = make_fake_client()
    fake.connect = AsyncMock(side_effect=error)
    wrapper = TelegramClientWrapper(config=config, client=fake)

    expected = TelegramFloodWaitError if isinstance(error, FloodWaitError) else TelegramAccessError
    with pytest.raises(expected):
        await wrapper.connect()
    assert wrapper.is_connected is False


@pytest.mark.parametrize("error", [make_flood(9), ChannelPrivateError(request=None)])
async def test_resolve_me_id_maps_rpc_errors_to_domain_errors(
    tmp_path: Path, error: Exception
) -> None:
    """_resolve_me_id() re-raises flood/access errors instead of degrading to None."""
    config = make_config(tmp_path)
    fake = make_fake_client()
    fake.get_me = AsyncMock(side_effect=error)
    wrapper = TelegramClientWrapper(config=config, client=fake)

    expected = TelegramFloodWaitError if isinstance(error, FloodWaitError) else TelegramAccessError
    with pytest.raises(expected):
        await wrapper.connect()


@pytest.mark.parametrize("error", [make_flood(4), ChannelInvalidError(request=None)])
async def test_resolve_bot_id_maps_rpc_errors_to_domain_errors(
    tmp_path: Path, error: Exception
) -> None:
    """_resolve_bot_id() re-raises flood/access errors from the entity lookup."""
    config = make_config(tmp_path, bot_id=None, bot_username="@TrueMafiaBot")
    fake = make_fake_client()
    fake.get_entity = AsyncMock(side_effect=error)
    wrapper = TelegramClientWrapper(config=config, client=fake)

    expected = TelegramFloodWaitError if isinstance(error, FloodWaitError) else TelegramAccessError
    with pytest.raises(expected):
        await wrapper.connect()
