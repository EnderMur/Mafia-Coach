"""Domain-specific exception types."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for domain errors."""


class ConfigError(DomainError):
    """Base class for configuration errors."""


class RolesConfigError(ConfigError):
    """Raised when the roles configuration is invalid."""


class UnknownRoleError(DomainError):
    """Raised when a role identifier is not registered."""


class LLMError(DomainError):
    """Raised when the language-model layer fails."""


class LLMConnectionError(LLMError):
    """Raised when the language-model service cannot be reached."""


class TelegramNotConnectedError(DomainError):
    """Raised when a Telegram operation requires an active authorized session."""


class TelegramFloodWaitError(DomainError):
    """Telegram rate-limited the request; retry after ``wait_seconds``."""

    __slots__ = ("wait_seconds",)

    def __init__(self, wait_seconds: int, message: str = "") -> None:
        """Initialize the error with the server-provided retry delay.

        Args:
            wait_seconds: Number of seconds Telegram asks the client to wait.
            message: Optional custom message; a default hint is built when empty.
        """
        self.wait_seconds = wait_seconds
        super().__init__(message or f"Telegram FloodWait: retry after {wait_seconds}s")


class TelegramAccessError(DomainError):
    """Telegram refused access (private chat, admin rights, kicked from chat, etc.)."""


class TrainingError(DomainError):
    """Raised when offline fine-tune training fails or its dependencies are missing."""
