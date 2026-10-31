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
