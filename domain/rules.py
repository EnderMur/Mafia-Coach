"""Pure predicates for club meta-rules."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

LYNCH_BAN_DAYS: int = 2
MIN_PLAYERS_FOR_LYNCH_BAN: int = 29
ANTITARGET_NIGHT: int = 1
ANTITARGET_MIN_ALIVE_EXCEPTION: int = 8
ANTITARGET_ROLE_IDS: frozenset[str] = frozenset(
    {"mistress", "commissioner", "don", "maniac", "killer"}
)


class RuleCheck(BaseModel):
    """The result of evaluating a meta-rule predicate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    rule_id: str | None = None
    reason: str | None = None

    @classmethod
    def ok(cls) -> RuleCheck:
        """Return a successful rule check."""
        return cls(allowed=True)

    @classmethod
    def forbid(cls, rule_id: str, reason: str) -> RuleCheck:
        """Return a failed rule check with its identifier and reason."""
        return cls(allowed=False, rule_id=rule_id, reason=reason)


class MetaRule(BaseModel):
    """Static metadata describing a club meta-rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    description: str


class RulesEngine:
    """Evaluate stateless predicates for club meta-rules."""

    @staticmethod
    def check_lynch(
        *,
        day: int,
        alive_count: int,
        has_argument: bool,
        is_rule_breaker: bool = False,
    ) -> RuleCheck:
        """Check whether a lynch is allowed during the opening game days.

        During the first LYNCH_BAN_DAYS game days, lynching without an argument is
        prohibited, except for rule breakers and games with at least
        MIN_PLAYERS_FOR_LYNCH_BAN living players. Any lynch is allowed afterward.
        """
        restricted_day = 1 <= day <= LYNCH_BAN_DAYS
        exception_applies = (
            has_argument or is_rule_breaker or alive_count >= MIN_PLAYERS_FOR_LYNCH_BAN
        )
        if restricted_day and not exception_applies:
            return RuleCheck.forbid(
                "lynch_no_argument",
                "Lynch without an argument is prohibited during the opening days.",
            )
        return RuleCheck.ok()

    @staticmethod
    def check_night_action(
        *,
        night: int,
        role_id: str,
        target_has_antitarget: bool,
        alive_count: int,
    ) -> RuleCheck:
        """Check the first-night anti-target restriction.

        During ANTITARGET_NIGHT, roles in ANTITARGET_ROLE_IDS may not target a
        player with an anti-target marker, unless at most
        ANTITARGET_MIN_ALIVE_EXCEPTION players remain alive.
        """
        restricted = (
            night == ANTITARGET_NIGHT
            and role_id in ANTITARGET_ROLE_IDS
            and target_has_antitarget
            and alive_count > ANTITARGET_MIN_ALIVE_EXCEPTION
        )
        if restricted:
            return RuleCheck.forbid(
                "antitarget_night",
                "This role cannot target an anti-target player on the first night.",
            )
        return RuleCheck.ok()

    @staticmethod
    def registry() -> tuple[MetaRule, ...]:
        """Return the static registry of meta-rules for display."""
        return (
            MetaRule(
                id="lynch_no_argument",
                description="Restricts unsupported lynches during the opening game days.",
            ),
            MetaRule(
                id="antitarget_night",
                description="Restricts selected first-night actions against anti-targets.",
            ),
        )
