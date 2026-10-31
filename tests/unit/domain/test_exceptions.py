"""Tests for domain exception types."""

from __future__ import annotations

from domain.exceptions import ConfigError, DomainError, RolesConfigError, UnknownRoleError


def test_exception_inheritance() -> None:
    """Configuration errors descend from the domain error root."""
    assert issubclass(RolesConfigError, ConfigError)
    assert issubclass(ConfigError, DomainError)


def test_unknown_role_is_domain_error() -> None:
    """Unknown role errors are domain errors."""
    assert isinstance(UnknownRoleError("x"), DomainError)


def test_exceptions_preserve_message() -> None:
    """Domain exceptions preserve their supplied message."""
    error = RolesConfigError("invalid roles")
    assert str(error) == "invalid roles"
