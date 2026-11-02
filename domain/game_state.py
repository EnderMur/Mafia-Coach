"""Mutable game aggregate built from immutable domain models and events."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from domain.events import (
    AliveListEvent,
    AnyEvent,
    DeathEvent,
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
        """Register a player or refresh their display details."""
        existing = self._players.get(player_id)
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
            if payload.from_phase == payload.to_phase:
                raise ConfigError("phase change must transition to a different phase")
            if payload.from_phase != self._phase:
                raise ConfigError(
                    f"phase change starts at {payload.from_phase.value}, "
                    f"but current phase is {self._phase.value}"
                )
        elif isinstance(event, AliveListEvent):
            # A2-1: every registered player absent from the bot's snapshot is
            # dead. Names are display_name strings; unknown names are ignored
            # (the bot's list never introduces players on its own).
            alive_names = set(event.payload.alive_player_names)
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
        else:
            raise ConfigError(f"unsupported event type: {type(event).__name__}")

        if updated_player is not None:
            self._players[updated_player.player_id] = updated_player
        if isinstance(event, PhaseChangeEvent):
            self._phase = event.payload.to_phase
            self._day_number = event.payload.day_number
        self._events.append(event)

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
