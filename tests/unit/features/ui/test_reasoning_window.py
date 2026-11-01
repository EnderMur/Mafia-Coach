"""Reasoning-window rendering and stdin-loop tests."""

from __future__ import annotations

import io

import pytest

from features.ui.protocol import IPCEnvelope, ReasoningMessage, encode
from features.ui.reasoning_window import main


def _line(text: str, *, truncated: bool = False) -> str:
    """Encode one reasoning envelope for the child loop."""
    return encode(
        IPCEnvelope(
            type="reasoning",
            payload=ReasoningMessage(
                text=text,
                truncated=truncated,
                total_chars=len(text),
            ).model_dump(),
        )
    )


def test_main_renders_three_reasoning_envelopes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each valid reasoning frame is printed to the output window."""
    content = "\n".join((_line("Первое"), _line("Второе"), _line("Третье"), ""))

    assert main(io.StringIO(content)) == 0
    output = capsys.readouterr().out
    assert all(text in output for text in ("Первое", "Второе", "Третье"))


def test_main_marks_truncated_reasoning(capsys: pytest.CaptureFixture[str]) -> None:
    """Truncated content is labeled for the user."""
    assert main(io.StringIO(_line("Сокращённый текст", truncated=True) + "\n")) == 0

    assert "усечено" in capsys.readouterr().out


def test_main_exits_on_shutdown() -> None:
    """Shutdown causes a clean exit code."""
    assert main(io.StringIO(encode(IPCEnvelope(type="shutdown")) + "\n")) == 0


def test_main_skips_invalid_line() -> None:
    """Invalid lines are ignored."""
    content = "not json\n" + encode(IPCEnvelope(type="shutdown")) + "\n"

    assert main(io.StringIO(content)) == 0
