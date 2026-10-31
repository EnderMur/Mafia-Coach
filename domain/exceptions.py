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
