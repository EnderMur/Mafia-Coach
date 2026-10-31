"""Role models, registry, and configuration loading."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Self

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from domain.exceptions import RolesConfigError, UnknownRoleError


class Faction(str, Enum):
    """A role's faction."""

    TOWN = "town"
    MAFIA = "mafia"
    NEUTRAL = "neutral"


class RoleCategory(str, Enum):
    """Whether a role has an active or passive gameplay category."""

    ACTIVE = "active"
    PASSIVE = "passive"


class NightAction(str, Enum):
    """A night action identifier defined by the roles configuration."""

    BLOCK = "block"
    BURN = "burn"
    CHECK = "check"
    CHOOSE_TARGET = "choose_target"
    HEAL = "heal"
    INVESTIGATE = "investigate"
    KILL = "kill"
    LEARN_ROLE = "learn_role"
    PROTECT = "protect"
    VISIT = "visit"


class Role(BaseModel):
    """Validated, immutable role definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name_ru: str = Field(min_length=1)
    faction: Faction
    category: RoleCategory
    night_actions: list[NightAction] = Field(default_factory=list)
    can_kill: bool
    can_check: bool
    can_heal: bool
    notes: str = ""
    bot_aliases: list[str] = Field(min_length=1)

    @field_validator("name_ru")
    @classmethod
    def validate_name_ru(cls, value: str) -> str:
        """Reject role names that contain only whitespace."""
        if not value.strip():
            raise ValueError("name_ru must not be empty or whitespace")
        return value

    @field_validator("bot_aliases")
    @classmethod
    def validate_bot_aliases(cls, values: list[str]) -> list[str]:
        """Reject blank or duplicate aliases within one role."""
        if any(not alias.strip() for alias in values):
            raise ValueError("bot_aliases must not contain empty or whitespace values")
        normalized = [alias.casefold() for alias in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("bot_aliases must be unique within a role")
        return values


class RolesConfig(BaseModel):
    """Validated collection of role definitions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roles: list[Role] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_roles(self) -> Self:
        """Reject duplicate identifiers and Russian role names."""
        ids = [role.id for role in self.roles]
        names = [role.name_ru for role in self.roles]
        if len(ids) != len(set(ids)):
            raise ValueError("role ids must be unique")
        if len(names) != len(set(names)):
            raise ValueError("role name_ru values must be unique")
        return self


class RoleRegistry:
    """Lookup roles by identifier, Russian name, or bot alias."""

    def __init__(self, roles: list[Role] | tuple[Role, ...]) -> None:
        self._roles = tuple(roles)
        self._by_id: dict[str, Role] = {role.id: role for role in self._roles}
        self._by_name_ru: dict[str, Role] = {role.name_ru.casefold(): role for role in self._roles}
        self._by_alias: dict[str, Role] = {}
        for role in self._roles:
            for alias in role.bot_aliases:
                key = alias.casefold()
                existing = self._by_alias.get(key)
                if existing is not None and existing.id != role.id:
                    raise RolesConfigError(f"bot alias {alias!r} belongs to multiple roles")
                self._by_alias[key] = role

    def get(self, role_id: str) -> Role:
        """Return a role by id, raising when it is not registered."""
        try:
            return self._by_id[role_id]
        except KeyError as error:
            raise UnknownRoleError(f"unknown role: {role_id}") from error

    def get_by_name_ru(self, name: str) -> Role | None:
        """Return a role by its Russian name, or None if absent."""
        return self._by_name_ru.get(name.casefold())

    def resolve_alias(self, text: str) -> Role | None:
        """Resolve a Russian role name or bot alias without case sensitivity."""
        normalized = text.casefold()
        role = self._by_name_ru.get(normalized)
        if role is not None:
            return role
        return self._by_alias.get(normalized)

    def all_roles(self) -> tuple[Role, ...]:
        """Return all roles in configuration order."""
        return self._roles

    def by_faction(self, faction: Faction) -> tuple[Role, ...]:
        """Return roles belonging to a faction in configuration order."""
        return tuple(role for role in self._roles if role.faction == faction)

    def by_category(self, category: RoleCategory) -> tuple[Role, ...]:
        """Return roles belonging to a category in configuration order."""
        return tuple(role for role in self._roles if role.category == category)

    def all_ids(self) -> frozenset[str]:
        """Return all registered role identifiers."""
        return frozenset(self._by_id)

    def __len__(self) -> int:
        """Return the number of registered roles."""
        return len(self._roles)


def load_roles(path: Path | str) -> RoleRegistry:
    """Load and validate role definitions from a UTF-8 YAML file."""
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
        config = RolesConfig.model_validate(raw_config)
        registry = RoleRegistry(config.roles)
    except (FileNotFoundError, yaml.YAMLError, ValidationError, RolesConfigError) as error:
        raise RolesConfigError(f"ошибка в {config_path}: {error}") from error

    logger.info("roles loaded: {}", len(registry))
    return registry
