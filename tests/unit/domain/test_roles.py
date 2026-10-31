"""Tests for role validation, loading, and registry lookups."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from domain.exceptions import RolesConfigError, UnknownRoleError
from domain.roles import (
    Faction,
    NightAction,
    Role,
    RoleCategory,
    RoleRegistry,
    load_roles,
)


ROLES_PATH = Path(__file__).parents[3] / "configs" / "roles.yaml"


def make_role(**overrides: Any) -> dict[str, Any]:
    """Create a minimal valid role mapping for configuration tests."""
    role: dict[str, Any] = {
        "id": "sample_role",
        "name_ru": "Тестовая роль",
        "faction": "town",
        "category": "active",
        "night_actions": [],
        "can_kill": False,
        "can_check": False,
        "can_heal": False,
        "notes": "",
        "bot_aliases": ["Тест"],
    }
    role.update(overrides)
    return role


def write_config(tmp_path: Path, roles: list[dict[str, Any]]) -> Path:
    """Write a role config fixture into a temporary directory."""
    path = tmp_path / "roles.yaml"
    path.write_text(json.dumps({"roles": roles}, ensure_ascii=False), encoding="utf-8")
    return path


def load_real_roles() -> RoleRegistry:
    """Load the repository's real roles configuration."""
    return load_roles(ROLES_PATH)


def test_load_real_config_has_21_roles() -> None:
    """The real configuration contains the contracted number of roles."""
    assert len(load_real_roles()) == 21


def test_real_role_ids_follow_pattern() -> None:
    """Every configured identifier follows the stable id pattern."""
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", role.id) for role in load_real_roles().all_roles())


def test_real_role_ids_and_names_are_unique() -> None:
    """The real configuration has unique ids and Russian names."""
    roles = load_real_roles().all_roles()
    assert len({role.id for role in roles}) == len(roles)
    assert len({role.name_ru for role in roles}) == len(roles)


def test_real_aliases_are_unique_within_each_role() -> None:
    """Each role's bot aliases are unique without regard to case."""
    for role in load_real_roles().all_roles():
        aliases = [alias.casefold() for alias in role.bot_aliases]
        assert len(aliases) == len(set(aliases))


def test_role_is_frozen() -> None:
    """Role attributes cannot be reassigned after construction."""
    role = load_real_roles().all_roles()[0]
    with pytest.raises(ValidationError):
        role.name_ru = "Изменение"


@pytest.mark.parametrize(
    "role_fields",
    [
        {"night_actions": ["future_action"]},
        {"faction": "unknown"},
        {"name_ru": ""},
        {"name_ru": "   "},
        {"id": "Invalid-id"},
        {"bot_aliases": [" "]},
        {"bot_aliases": ["same", "SAME"]},
    ],
)
def test_invalid_role_fields_raise_roles_config_error(
    tmp_path: Path, role_fields: dict[str, Any]
) -> None:
    """Invalid enum values and malformed role fields are wrapped on load."""
    path = write_config(tmp_path, [make_role(**role_fields)])
    with pytest.raises(RolesConfigError):
        load_roles(path)


def test_duplicate_role_id_raises_roles_config_error(tmp_path: Path) -> None:
    """Duplicate role ids invalidate a configuration."""
    path = write_config(tmp_path, [make_role(), make_role(name_ru="Вторая роль")])
    with pytest.raises(RolesConfigError, match="unique"):
        load_roles(path)


def test_duplicate_role_name_raises_roles_config_error(tmp_path: Path) -> None:
    """Duplicate Russian role names invalidate a configuration."""
    first = make_role()
    second = make_role(id="second_role", bot_aliases=["Второй алиас"])
    path = write_config(tmp_path, [first, second])
    with pytest.raises(RolesConfigError, match="unique"):
        load_roles(path)


def test_duplicate_alias_between_roles_raises_roles_config_error(tmp_path: Path) -> None:
    """An alias cannot identify two separate roles."""
    first = make_role()
    second = make_role(id="second_role", name_ru="Вторая роль", bot_aliases=["тЕсТ"])
    path = write_config(tmp_path, [first, second])
    with pytest.raises(RolesConfigError, match="multiple roles"):
        load_roles(path)


def test_yaml_and_missing_file_errors_are_wrapped(tmp_path: Path) -> None:
    """Missing files and malformed YAML become contextual config errors."""
    missing = tmp_path / "missing.yaml"
    with pytest.raises(RolesConfigError, match="ошибка в"):
        load_roles(missing)
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("roles: [", encoding="utf-8")
    with pytest.raises(RolesConfigError, match="ошибка в"):
        load_roles(malformed)


def test_registry_get_by_id_and_unknown_id() -> None:
    """Registry id lookup returns roles and reports missing ids."""
    registry = load_real_roles()
    role = registry.all_roles()[0]
    assert registry.get(role.id) == role
    with pytest.raises(UnknownRoleError, match="missing"):
        registry.get("missing")


def test_registry_get_by_name_ru() -> None:
    """Russian names can be looked up case-insensitively."""
    registry = load_real_roles()
    role = registry.all_roles()[0]
    assert registry.get_by_name_ru(role.name_ru.swapcase()) == role
    assert registry.get_by_name_ru("несуществующая роль") is None


def test_resolve_alias_prefers_russian_name_and_ignores_case() -> None:
    """Russian names resolve case-insensitively before aliases."""
    registry = load_real_roles()
    role = registry.all_roles()[0]
    assert registry.resolve_alias(role.name_ru.swapcase()) == role


def test_every_real_bot_alias_resolves_to_its_role() -> None:
    """Each alias in the source config resolves to its owning role."""
    registry = load_real_roles()
    for role in registry.all_roles():
        for alias in role.bot_aliases:
            assert registry.resolve_alias(alias) == role
            assert registry.resolve_alias(alias.swapcase()) == role


def test_registry_filters_and_collection_methods() -> None:
    """Registry filters and immutable collection views are correct."""
    registry = load_real_roles()
    assert len(registry.by_faction(Faction.TOWN)) == 10
    assert len(registry.by_faction(Faction.MAFIA)) == 5
    assert len(registry.by_faction(Faction.NEUTRAL)) == 6
    assert registry.by_category(RoleCategory.ACTIVE)
    assert registry.by_category(RoleCategory.PASSIVE)
    assert isinstance(registry.all_roles(), tuple)
    assert isinstance(registry.all_ids(), frozenset)
    assert len(registry.all_ids()) == len(registry)


def test_commissioner_contract_snapshot() -> None:
    """The commissioner retains the configured check and kill contract."""
    commissioner = load_real_roles().get("commissioner")
    assert commissioner.can_check is True
    assert commissioner.can_kill is True
    assert NightAction.CHECK in commissioner.night_actions
    assert NightAction.KILL in commissioner.night_actions


def test_registry_rejects_alias_collision_directly() -> None:
    """Direct registry construction also rejects cross-role alias conflicts."""
    first = Role.model_validate(make_role())
    second = Role.model_validate(
        make_role(id="second_role", name_ru="Вторая роль", bot_aliases=["тЕсТ"])
    )
    with pytest.raises(RolesConfigError, match="multiple roles"):
        RoleRegistry([first, second])


def test_config_rejects_empty_roles_and_extra_fields(tmp_path: Path) -> None:
    """Empty role lists and unknown configuration fields are rejected."""
    empty_path = write_config(tmp_path, [])
    with pytest.raises(RolesConfigError):
        load_roles(empty_path)
    extra_path = tmp_path / "extra.yaml"
    extra_path.write_text(
        json.dumps({"roles": [make_role()], "unexpected": True}), encoding="utf-8"
    )
    with pytest.raises(RolesConfigError):
        load_roles(extra_path)
