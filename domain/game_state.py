"""Mutable game aggregate built from immutable domain models and events."""

from __future__ import annotations

import re

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

_USER_ID_RE = re.compile(r"^user_(\d+)$")

from domain.events import (
    AliveListEvent,
    AnyEvent,
    DeathEvent,
    EndGameEvent,
    MessageEvent,
    PhaseChangeEvent,
    RoleRevealEvent,
    VoteEvent,
)
from domain.exceptions import ConfigError, UnknownRoleError
from domain.phases import Phase
from domain.roles import Role, RoleRegistry


class Player(BaseModel):
    """Immutable snapshot of a player's current game state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: int
    username: str | None
    display_name: str = Field(min_length=1)
    alive: bool = True
    known_role: Role | None = None
    revealed: bool = False
    #: Final role from the bot's end-of-game report (B2-3: takes precedence
    #: over ``known_role`` from a mid-game ``ROLE_REVEAL``).
    final_role: Role | None = None
    #: ``True`` for winners, ``False`` for losers, ``None`` while unknown.
    is_winner: bool | None = None


class GameState:
    """Track players, phase, and successfully applied events for one game."""

    def __init__(self, game_id: int, registry: RoleRegistry) -> None:
        self._game_id = game_id
        self._registry = registry
        self._players: dict[int, Player] = {}
        self._events: list[AnyEvent] = []
        self._phase = Phase.REGISTRATION
        self._day_number = 0
        self._role_count: dict[str, int] | None = None
        self._total_players: int | None = None

    @property
    def game_id(self) -> int:
        """Public accessor for the game id."""
        return self._game_id

    def register_player(
        self, player_id: int, display_name: str, username: str | None = None
    ) -> Player:
        """Register a player or refresh their display details.

        A real (positive) id collapses the placeholder that the earlier
        ``ALIVE_LIST`` registered for the same display name: the placeholder is
        dropped and any ground truth it already carries (``known_role``,
        ``final_role``, ``is_winner``) moves to the real player. Without this
        merge the same person exists twice in the game and ``END_GAME``
        matching by display name writes the final role into the placeholder,
        leaving the real id labeled "unknown" in retro.
        """
        existing = self._players.get(player_id)
        if player_id > 0:
            # Real id arrives: drop any placeholder with the same
            # display_name (registered from ALIVE_LIST before speech).
            placeholder_ids = [
                pid
                for pid, p in self._players.items()
                if pid < 0 and p.display_name == display_name
            ]
            for pid in placeholder_ids:
                placeholder = self._players.pop(pid)
                # Transfer any known_role/final_role from placeholder
                if existing is None and (
                    placeholder.known_role
                    or placeholder.final_role
                    or placeholder.is_winner is not None
                ):
                    existing = placeholder
        if existing is None:
            player = Player(
                player_id=player_id,
                username=username,
                display_name=display_name,
            )
        else:
            player = Player(
                player_id=player_id,
                username=username,
                display_name=display_name,
                alive=existing.alive,
                known_role=existing.known_role,
                revealed=existing.revealed,
                final_role=existing.final_role,
                is_winner=existing.is_winner,
            )
        self._players[player_id] = player
        return player

    def apply(self, event: AnyEvent) -> None:
        """Validate and apply an event, recording it only after success."""
        if event.game_id != self._game_id:
            raise ConfigError(f"event belongs to game {event.game_id}, not game {self._game_id}")

        updated_player: Player | None = None
        if isinstance(event, DeathEvent):
            player = self._get_player(event.payload.victim_id)
            if not player.alive:
                raise ConfigError(f"player {player.player_id} is already dead")
            updated_player = player.model_copy(update={"alive": False})
        elif isinstance(event, RoleRevealEvent):
            player = self._get_player(event.payload.player_id)
            role = self._registry.resolve_alias(event.payload.role_name_ru)
            if role is None:
                raise UnknownRoleError(f"unknown role name: {event.payload.role_name_ru}")
            updated_player = player.model_copy(update={"known_role": role, "revealed": True})
        elif isinstance(event, PhaseChangeEvent):
            payload = event.payload

            # A transition to REGISTRATION marks a new game and resets the
            # aggregate, including when the previous game had already ended.
            if payload.to_phase == Phase.REGISTRATION:
                if self._phase != Phase.REGISTRATION or self._players:
                    logger.bind(channel="game_state").info(
                        "new game detected: phase {} → REGISTRATION, "
                        "resetting state ({} players dropped)",
                        self._phase.value,
                        len(self._players),
                    )
                self._phase = Phase.REGISTRATION
                self._day_number = 0
                self._players = {}
                self._role_count = None
                self._total_players = None
                self._events.append(event)
                return

            if self._phase == Phase.END:
                # Post-END phase chatter is expected in backfills/replays:
                # keep it out of WARNING logs (noise on every finished game).
                logger.bind(channel="game_state").debug(
                    "phase_change to {} after END — ignored", event.payload.to_phase
                )
                return
            if payload.from_phase == payload.to_phase:
                raise ConfigError("phase change must transition to a different phase")
            if payload.from_phase != self._phase:
                raise ConfigError(
                    f"phase change starts at {payload.from_phase.value}, "
                    f"but current phase is {self._phase.value}"
                )
        elif isinstance(event, AliveListEvent):
            for name in event.payload.alive_player_names:
                pid = _placeholder_player_id(name)
                if pid in self._players:
                    # Player registered earlier (real id from speech or
                    # previously from ALIVE_LIST). Don't overwrite
                    # display_name, just refresh alive.
                    continue
                # A real (positive) id already known by this name must never
                # get a second, placeholder identity.
                if any(p.player_id > 0 and p.display_name == name for p in self._players.values()):
                    continue
                self.register_player(pid, name, username=None)
            alive_names = set(event.payload.alive_player_names)
            for name in event.payload.alive_player_names:
                pid = _placeholder_player_id(name)
                if pid in self._players:
                    alive_names.add(self._players[pid].display_name)
            updated: dict[int, Player] = {}
            for pid, player in self._players.items():
                if player.display_name in alive_names:
                    updated[pid] = player
                elif player.alive:
                    updated[pid] = player.model_copy(update={"alive": False})
                else:
                    updated[pid] = player
            self._players = updated
            self._role_count = (
                dict(event.payload.role_count) if event.payload.role_count is not None else None
            )
            self._total_players = event.payload.total_players
        elif isinstance(event, (MessageEvent, VoteEvent)):
            pass
        elif isinstance(event, EndGameEvent):
            self._apply_end_game(event)
        else:
            raise ConfigError(f"unsupported event type: {type(event).__name__}")

        if updated_player is not None:
            self._players[updated_player.player_id] = updated_player
        if isinstance(event, PhaseChangeEvent):
            self._phase = event.payload.to_phase
            if event.payload.day_number > 0:
                # Явный номер дня из бота (например, "День 2").
                self._day_number = event.payload.day_number
            else:
                # Бот не указал номер — инкрементим на каждый переход в новую
                # фазу (DAY/NIGHT/VOTE). REGISTRATION/END — не инкрементят.
                if event.payload.to_phase in (
                    Phase.DAY,
                    Phase.NIGHT,
                    Phase.VOTE,
                ):
                    self._day_number += 1
        self._events.append(event)

    def _apply_end_game(self, event: EndGameEvent) -> None:
        """Write ``final_role`` / ``is_winner`` from the bot's final report.

        B2-3: the final report is ground truth and takes precedence over a
        ``ROLE_REVEAL`` seen earlier in the game. A mismatch means the bot or
        the parser is wrong, so it is reported as a WARNING instead of being
        silently overwritten.

        Names match display names first, then usernames; unknown players get
        deterministic placeholder ids. Roles that do not resolve to a
        configured ``Role`` leave ``final_role`` untouched (the winner flag is
        still recorded — it does not depend on the role).

        The phase always moves to :attr:`~domain.phases.Phase.END`.
        """
        winners_map = dict(event.payload.winners)
        losers_map = dict(event.payload.losers)

        for name in winners_map.keys() | losers_map.keys():
            player = next((p for p in self._players.values() if p.display_name == name), None)
            if player is None:
                player = next((p for p in self._players.values() if p.username == name), None)
            if player is None:
                player = self.register_player(_placeholder_player_id(name), name)
            player_id = player.player_id
            is_winner = name in winners_map
            final_role_ru = winners_map[name] if is_winner else losers_map[name]

            final_role = self._resolve_final_role(final_role_ru)
            if player.known_role is not None:
                mismatch = (
                    player.known_role.id != final_role.id
                    if final_role is not None
                    else player.known_role.name_ru != final_role_ru
                )
                if mismatch:
                    logger.bind(channel="game_state").warning(
                        "final_role mismatch for {}: known={} final={}",
                        player.display_name,
                        player.known_role.name_ru,
                        final_role_ru,
                    )

            update: dict[str, object] = {"is_winner": is_winner, "revealed": True}
            if final_role is not None:
                update["final_role"] = final_role
            self._players[player_id] = player.model_copy(update=update)

        self._phase = Phase.END

    def _resolve_final_role(self, role_name_ru: str) -> Role | None:
        """Resolve the bot's role name (``name_ru``) to a registered role.

        The bot capitalises names inconsistently (``"Мирный Житель"`` vs
        ``"Мирный житель"``), so matching is case-insensitive; surrounding
        quotes and trailing punctuation are tolerated as well.
        """
        candidate = role_name_ru.strip().strip("«»\"'()")
        role = self._registry.resolve_alias(candidate)
        if role is not None:
            return role
        collapsed = " ".join(candidate.split())
        role = self._registry.resolve_alias(collapsed)
        if role is None:
            logger.bind(channel="game_state").warning("unknown final role name: {!r}", role_name_ru)
        return role

    def player(self, player_id: int) -> Player:
        """Return a registered player or raise a configuration error."""
        return self._get_player(player_id)

    def players(self) -> tuple[Player, ...]:
        """Return all players in registration order."""
        return tuple(self._players.values())

    def alive_players(self) -> tuple[Player, ...]:
        """Return all living players in registration order."""
        return tuple(player for player in self._players.values() if player.alive)

    def dead_players(self) -> tuple[Player, ...]:
        """Return all dead players in registration order."""
        return tuple(player for player in self._players.values() if not player.alive)

    def current_phase(self) -> Phase:
        """Return the current game phase."""
        return self._phase

    def current_day(self) -> int:
        """Return the current game day number."""
        return self._day_number

    @property
    def role_count(self) -> dict[str, int] | None:
        """Bot-reported role composition (name_ru → count), or None."""
        return dict(self._role_count) if self._role_count is not None else None

    @property
    def total_players(self) -> int | None:
        """Bot-reported total alive count, or None."""
        return self._total_players

    def events(self) -> tuple[AnyEvent, ...]:
        """Return successfully applied events in append order."""
        return tuple(self._events)

    def events_by_kind(self, kind: str) -> tuple[AnyEvent, ...]:
        """Return successfully applied events matching a kind identifier."""
        return tuple(event for event in self._events if event.kind == kind)

    def _get_player(self, player_id: int) -> Player:
        """Return a registered player for internal state transitions."""
        try:
            return self._players[player_id]
        except KeyError as error:
            raise ConfigError(f"player {player_id} not registered") from error


def _placeholder_player_id(name: str) -> int:
    """Resolve player id from name.

    If name matches "user_<id>" (bot fallback for users without
    username), return the real id. Otherwise return a stable
    negative int64 placeholder (Telegram user ids are positive).
    """
    import hashlib

    m = _USER_ID_RE.match(name)
    if m:
        return int(m.group(1))
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    n = int.from_bytes(digest[:8], "big", signed=False)
    return -(n % (2**62)) - 1
