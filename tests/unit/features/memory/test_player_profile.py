"""Tests for player profile loading and note updates."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest
from pydantic import ValidationError

from domain.exceptions import ConfigError
from features.memory.player_profile import PlayerProfile, load_profile, update_notes
from infra.db.repositories.players import PlayersRepository

PLAYER_ID = 5001


@pytest.fixture
def players_repo(conn: sqlite3.Connection) -> PlayersRepository:
    """Provide a players repository with one stored player."""
    repository = PlayersRepository(conn)
    repository.upsert_player(
        player_id=PLAYER_ID, username="nostradamus", display_name="Нострадамус"
    )
    return repository


def test_load_profile_returns_stored_player(players_repo: PlayersRepository) -> None:
    """An existing player is returned with every field populated."""
    profile = load_profile(players_repo, PLAYER_ID)

    assert profile is not None
    assert isinstance(profile, PlayerProfile)
    assert profile.player_id == PLAYER_ID
    assert profile.username == "nostradamus"
    assert profile.notes is None
    assert isinstance(profile.first_seen_at, datetime)
    assert isinstance(profile.last_seen_at, datetime)


def test_load_profile_returns_none_for_unknown_player(players_repo: PlayersRepository) -> None:
    """An unknown player produces None instead of an exception."""
    assert load_profile(players_repo, 404404) is None


def test_update_notes_is_visible_on_next_load(players_repo: PlayersRepository) -> None:
    """Updated notes are persisted and read back."""
    update_notes(players_repo, player_id=PLAYER_ID, notes="блефует в первую ночь")

    profile = load_profile(players_repo, PLAYER_ID)
    assert profile is not None
    assert profile.notes == "блефует в первую ночь"


def test_update_notes_for_unknown_player_raises_config_error(
    players_repo: PlayersRepository,
) -> None:
    """Updating notes of a missing player raises a configuration error."""
    with pytest.raises(ConfigError, match="player 404404 not found"):
        update_notes(players_repo, player_id=404404, notes="нет такого игрока")


def test_player_profile_is_frozen(players_repo: PlayersRepository) -> None:
    """The profile model rejects mutation."""
    profile = load_profile(players_repo, PLAYER_ID)
    assert profile is not None

    with pytest.raises(ValidationError):
        profile.notes = "подмена"


def test_load_profile_keeps_null_timestamps(
    conn: sqlite3.Connection, players_repo: PlayersRepository
) -> None:
    """Rows without timestamps produce a profile with None dates."""
    conn.execute("INSERT INTO players(id, username) VALUES (?, ?)", (7007, "no_dates"))

    profile = load_profile(players_repo, 7007)

    assert profile is not None
    assert profile.first_seen_at is None
    assert profile.last_seen_at is None


def test_load_profile_rejects_corrupted_timestamp(
    conn: sqlite3.Connection, players_repo: PlayersRepository
) -> None:
    """A non-ISO timestamp stored in the database surfaces as a domain error."""
    conn.execute("UPDATE players SET first_seen_at = ? WHERE id = ?", ("не дата", PLAYER_ID))

    with pytest.raises(ConfigError, match="invalid timestamp in players table"):
        load_profile(players_repo, PLAYER_ID)
