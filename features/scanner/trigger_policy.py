"""Pure decision logic for when the scanner flushes the LLM analysis.

No ``async``, no I/O, no timers: the debounce timer itself lives in the
scanner (step 9b).  This module only answers questions — "should this
event force an immediate analysis?", "is the buffer big enough?" — so the
policy can be unit-tested without an event loop.

Modes (``configs/settings.yaml`` → ``llm.trigger_mode``)
-------------------------------------------------------
``every_message``    — analyze on every single event.
``queue``            — the orchestrator drains the queue manually.
``batch_debounce``   — accumulate messages, but phase events (deaths,
                       phase changes, votes, role reveals) flush at once.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from domain.events import AnyEvent


class TriggerMode(str, Enum):
    """How the scanner decides when to flush the LLM analysis."""

    EVERY_MESSAGE = "every_message"
    QUEUE = "queue"
    BATCH_DEBOUNCE = "batch_debounce"


#: ``AnyEvent.kind`` values that force an immediate flush in
#: ``batch_debounce`` mode (AI_INSTRUCTIONS.md §13.1 п.6).
_FORCE_FLUSH_KINDS: Final[frozenset[str]] = frozenset(
    {
        "death",
        "phase_change",
        "vote",
        "role_reveal",
    }
)


class TriggerPolicy(BaseModel):
    """Pure logic for flush decisions. Scanner (9b) owns the timer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: TriggerMode
    batch_debounce_sec: float = Field(default=3.0, ge=0.0)
    batch_max_messages: int = Field(default=10, ge=1)

    def should_force_now(self, event: AnyEvent) -> bool:
        """Return whether this event bypasses the debounce timer.

        Args:
            event: a domain event produced by the resolver.

        Returns:
            ``True`` when analysis must start immediately:

            - ``every_message`` — ``True`` for every event;
            - ``queue`` — ``False`` always (the orchestrator manages the
              queue manually);
            - ``batch_debounce`` — ``True`` only for phase events
              (``death``, ``phase_change``, ``vote``, ``role_reveal``) and
              ``False`` for plain messages.
        """
        if self.mode is TriggerMode.EVERY_MESSAGE:
            return True
        if self.mode is TriggerMode.QUEUE:
            return False
        # BATCH_DEBOUNCE
        return event.kind in _FORCE_FLUSH_KINDS

    def should_force_on_size(self, buffer_count: int) -> bool:
        """Return whether the buffer has reached ``batch_max_messages``.

        For ``every_message`` this is irrelevant (nothing is buffered); for
        ``queue`` it is also ``True`` because the queue must drain.

        Args:
            buffer_count: number of messages currently accumulated.

        Returns:
            ``True`` when the buffer is full and must be flushed.
        """
        return buffer_count >= self.batch_max_messages

    def is_batch(self) -> bool:
        """Return whether the mode is ``batch_debounce``."""
        return self.mode is TriggerMode.BATCH_DEBOUNCE

    def is_every_message(self) -> bool:
        """Return whether the mode is ``every_message``."""
        return self.mode is TriggerMode.EVERY_MESSAGE

    def is_queue(self) -> bool:
        """Return whether the mode is ``queue``."""
        return self.mode is TriggerMode.QUEUE
