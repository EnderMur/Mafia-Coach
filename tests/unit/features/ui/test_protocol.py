"""Tests for the UI JSON-lines protocol."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from features.ui.protocol import (
    IPCEnvelope,
    ReasoningMessage,
    StatsMessage,
    StatsRow,
    decode,
    encode,
)


def test_stats_encode_decode_roundtrip() -> None:
    """Stats payloads survive JSON encoding and decoding."""
    payload = StatsMessage(
        game_id=3,
        phase="day",
        day=2,
        rows=[StatsRow(player_name="Анна", role="Мафия", confidence=0.7)],
    )
    envelope = IPCEnvelope(type="stats", payload=payload.model_dump())

    result = decode(encode(envelope))

    assert result == envelope
    assert StatsMessage.model_validate(result.payload) == payload


def test_reasoning_encode_decode_roundtrip() -> None:
    """Reasoning payloads retain truncation metadata."""
    message = ReasoningMessage(text="Рассуждение", truncated=True, total_chars=5000)
    envelope = IPCEnvelope(type="reasoning", payload=message.model_dump())

    result = decode(encode(envelope))

    assert ReasoningMessage.model_validate(result.payload) == message


def test_shutdown_envelope_roundtrip() -> None:
    """Shutdown messages can omit the optional payload."""
    envelope = IPCEnvelope(type="shutdown")

    assert decode(encode(envelope)) == envelope
    assert envelope.payload == {}


def test_decode_invalid_json_raises_value_error() -> None:
    """Malformed JSON is normalized to ValueError."""
    with pytest.raises(ValueError, match=re.escape("invalid IPC line")):
        decode("not json")


def test_decode_missing_type_raises_value_error() -> None:
    """Missing required wire fields are rejected."""
    with pytest.raises(ValueError, match=re.escape("invalid IPC line")):
        decode('{"payload": {}}')


def test_stats_row_rejects_confidence_above_one() -> None:
    """Confidence is constrained to probability bounds."""
    with pytest.raises(ValidationError):
        StatsRow(player_name="Анна", role="Мирный", confidence=1.1)


def test_stats_message_accepts_empty_rows() -> None:
    """An empty prediction set is a valid stats frame."""
    assert StatsMessage(game_id=1, phase="registration", day=0, rows=[]).rows == []


def test_envelope_rejects_unknown_type() -> None:
    """Only protocol-declared message types are accepted."""
    with pytest.raises(ValidationError):
        IPCEnvelope(type="unknown")  # type: ignore[arg-type]
