"""Club pattern detection and role-override application.

Detects which documented club patterns (from ``configs/club_patterns.yaml``)
have fired based on game events (messages, votes, role reveals, deaths) and
applies pattern-based role overrides to LLM predictions.

Pattern-based roles take priority over bot-composition slots: when a pattern
assigns a role to a player who previously held that slot from composition, the
slot is released and redistributed by ``GameAnalyzer._enforce_composition``.

Pure module: no I/O, no async, no imports from ``infra/`` or ``app/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from domain.club_patterns import ClubPattern, ClubPatterns
from domain.events import (
    MessageEvent,
    MessagePayload,
    RoleRevealEvent,
    RoleRevealPayload,
    VoteEvent,
)
from domain.game_state import GameState
from domain.roles import RoleRegistry
from infra.llm.schemas import PlayerPrediction

#: Regex for direct role claims: "я - Role", "я Role", "он - Role", etc.
_CLAIM_ROLE_RE: re.Pattern[str] = re.compile(
    r"(?:я|он|она|этот)\s*[-–—]\s*([А-Яа-яЁё\s]+?)(?:\.|,|!|\?|$|\s+(?:и|но|а|или))",
    re.IGNORECASE,
)

#: Regex for investigation claims: "я проверил X", "проверка X показала".
_INVESTIGATION_RE: re.Pattern[str] = re.compile(
    r"(?:я|мы)\s+(?:проверил|проверили|посмотрел|посмотрели)\s+(\S+?)(?:\.|,|!|\?|$|\s)",
    re.IGNORECASE,
)

#: Role-name keywords used to recognise claimed roles in free text.
_ROLE_KEYWORDS: tuple[str, ...] = (
    "Мафия",
    "Дон",
    "Комиссар",
    "Каттани",
    "Сержант",
    "Доктор",
    "Мирный",
    "Мирный житель",
    "Маньяк",
    "Убийца",
    "Адвокат",
    "Любовница",
    "Журналист",
    "Оборотень",
    "Камикадзе",
    "Бомж",
    "Слуга дьявола",
    "Знахарь",
)


@dataclass(frozen=True)
class PatternMatch:
    """One detected pattern application for a player."""

    player_id: int
    pattern_id: str
    role: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class SlotRelease:
    """A player whose composition-assigned slot is released by a pattern override."""

    player_id: int
    pattern_id: str


class PatternDetector:
    """Detects fired club patterns and computes role overrides."""

    def __init__(self, *, patterns: ClubPatterns, registry: RoleRegistry) -> None:
        """Retain the pattern catalogue and role registry.

        Args:
            patterns: validated club patterns from ``configs/club_patterns.yaml``.
            registry: role registry used to canonicalise role names.
        """
        self._patterns = patterns
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        *,
        game_state: GameState,
        previous_predictions_role_by_id: dict[int, str | None],
    ) -> tuple[list[PatternMatch], list[SlotRelease]]:
        """Detect fired patterns and compute slot releases.

        Scans all game events for pattern triggers. Patterns with
        ``requires_confirmation=true`` only fire after a confirming vote.

        Args:
            game_state: current game state with events.
            previous_predictions_role_by_id: mapping of player id to their
                previously assigned role (from composition or earlier cycle),
                used to compute slot releases.

        Returns:
            ``(matches, releases)``. ``matches`` are pattern overrides to
            apply; ``releases`` are players who lose their composition slot
            because a pattern took their role.
        """
        events = game_state.events()
        votes = [e for e in events if isinstance(e, VoteEvent)]
        messages = [e for e in events if isinstance(e, MessageEvent)]
        reveals = [e for e in events if isinstance(e, RoleRevealEvent)]

        confirmed_commissioners = self._confirmed_commissioners(reveals, votes)
        voter_ids = {v.payload.voter_id for v in votes}
        vote_targets = {v.payload.target_id for v in votes if v.payload.target_id is not None}

        matches: list[PatternMatch] = []
        seen_players: set[int] = set()

        # Direct role claims (#2 → Мафия, #7 → Дон): highest tier, no confirmation.
        for message in messages:
            match = self._detect_direct_claim(message.payload)
            if match is not None and match.player_id not in seen_players:
                matches.append(match)
                seen_players.add(match.player_id)

        # Investigation claims (#1): requires the claim author to have voted.
        for message in messages:
            match = self._detect_investigation_claim(message.payload, voter_ids)
            if match is not None and match.player_id not in seen_players:
                matches.append(match)
                seen_players.add(match.player_id)

        # Bot investigation results (#5): requires the revealed player to have
        # been voted for.
        for reveal in reveals:
            match = self._detect_bot_investigation(reveal.payload, vote_targets)
            if match is not None and match.player_id not in seen_players:
                matches.append(match)
                seen_players.add(match.player_id)

        # Vote against confirmed commissioner (#3).
        for vote in votes:
            if vote.payload.target_id in confirmed_commissioners:
                pattern = self._patterns.get("#3")
                matches.append(
                    PatternMatch(
                        player_id=vote.payload.voter_id,
                        pattern_id=pattern.id,
                        role=pattern.conclusion_role,
                        confidence=pattern.confidence_tier,
                        reason=(
                            f"Голосовал против подтверждённого Комиссара "
                            f"(паттерн #{pattern.id[1:]})"
                        ),
                    )
                )
                seen_players.add(vote.payload.voter_id)

        # Lucky survival (#6): survived a night kill.
        # Requires night-action targeting data not yet available; skipped here.

        # Compute slot releases: players who previously had a role that a
        # pattern now overrides with a different role.
        releases: list[SlotRelease] = []
        for match in matches:
            old_role = previous_predictions_role_by_id.get(match.player_id)
            if old_role is not None and old_role != match.role:
                releases.append(SlotRelease(player_id=match.player_id, pattern_id=match.pattern_id))

        return matches, releases

    def apply_overrides(
        self,
        *,
        predictions: list[PlayerPrediction],
        matches: list[PatternMatch],
        releases: list[SlotRelease],
        name_by_id: dict[int, str],
    ) -> list[PlayerPrediction]:
        """Apply pattern matches and slot releases to predictions.

        Pattern matches override the player's role with the pattern role,
        confidence tier, and pattern metadata. Slot releases set the released
        player to ``suspected_role = null`` with reason
        ``"слот переопределён паттерном #N"``.

        Args:
            predictions: current predictions (from LLM or composition).
            matches: pattern overrides from :meth:`detect`.
            releases: slot releases from :meth:`detect`.
            name_by_id: mapping of player id to display name for new
                predictions created by slot releases.

        Returns:
            Updated predictions with pattern overrides applied.
        """
        match_by_id = {m.player_id: m for m in matches}
        release_by_id = {r.player_id: r.pattern_id for r in releases}
        predicted_ids = {p.player_id for p in predictions}

        result: list[PlayerPrediction] = []
        for prediction in predictions:
            if prediction.player_id in match_by_id:
                # Pattern override takes priority: the player gets the pattern
                # role. Any slot release for this player is implicit — their old
                # role is freed by the new assignment.
                match = match_by_id[prediction.player_id]
                pattern = self._patterns.get(match.pattern_id)
                result.append(
                    prediction.model_copy(
                        update={
                            "suspected_role": match.role,
                            "confidence": match.confidence,
                            "reason": match.reason,
                            "reasons": [match.reason],
                            "pattern_ids": [match.pattern_id],
                            "pattern_descriptions": [
                                pattern.description if pattern else match.reason
                            ],
                        }
                    )
                )
            elif prediction.player_id in release_by_id:
                # Slot released with no replacement pattern: null role.
                pattern_id = release_by_id[prediction.player_id]
                pattern = self._patterns.get(pattern_id)
                result.append(
                    PlayerPrediction(
                        player_id=prediction.player_id,
                        player_name=name_by_id.get(
                            prediction.player_id, f"Игрок {prediction.player_id}"
                        ),
                        suspected_role=None,
                        confidence=0.0,
                        reason=f"слот переопределён паттерном {pattern_id}",
                        reasons=[f"слот переопределён паттерном {pattern_id}"],
                        pattern_ids=[pattern_id],
                        pattern_descriptions=[pattern.description if pattern else pattern_id],
                    )
                )
            else:
                result.append(prediction)

        # Add new predictions for matched players who had no prediction yet
        # (pattern override creates a prediction where the LLM was silent).
        for match in matches:
            if match.player_id not in predicted_ids and match.player_id not in release_by_id:
                pattern = self._patterns.get(match.pattern_id)
                result.append(
                    PlayerPrediction(
                        player_id=match.player_id,
                        player_name=name_by_id.get(match.player_id, f"Игрок {match.player_id}"),
                        suspected_role=match.role,
                        confidence=match.confidence,
                        reason=match.reason,
                        reasons=[match.reason],
                        pattern_ids=[match.pattern_id],
                        pattern_descriptions=[pattern.description if pattern else match.reason],
                    )
                )

        # Add new predictions for released players who had no prediction yet
        # and no matching override.
        for release in releases:
            if release.player_id not in predicted_ids and release.player_id not in match_by_id:
                pattern = self._patterns.get(release.pattern_id)
                result.append(
                    PlayerPrediction(
                        player_id=release.player_id,
                        player_name=name_by_id.get(release.player_id, f"Игрок {release.player_id}"),
                        suspected_role=None,
                        confidence=0.0,
                        reason=f"слот переопределён паттерном {release.pattern_id}",
                        reasons=[f"слот переопределён паттерном {release.pattern_id}"],
                        pattern_ids=[release.pattern_id],
                        pattern_descriptions=[
                            pattern.description if pattern else release.pattern_id
                        ],
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def _canonical_role(self, role_name: str) -> str | None:
        """Return canonical ``name_ru`` for a role name, or None."""
        role = self._registry.resolve_alias(role_name)
        return role.name_ru if role is not None else None

    def _extract_claimed_role(self, text: str) -> str | None:
        """Extract a claimed role name from free text, or None."""
        for keyword in _ROLE_KEYWORDS:
            if keyword.casefold() in text.casefold():
                canonical = self._canonical_role(keyword)
                if canonical is not None:
                    return canonical
        return None

    def _find_pattern_by_conclusion(self, role_name: str, trigger: str) -> ClubPattern | None:
        """Find a pattern matching the given conclusion role and trigger."""
        for pattern in self._patterns.patterns:
            if pattern.trigger == trigger and pattern.conclusion_role == role_name:
                return pattern
        return None

    def _detect_direct_claim(self, payload: MessagePayload) -> PatternMatch | None:
        """Detect a direct role claim ('я - Role', 'он - Role', …).

        Only the explicit dash-separated claim syntax is recognised; bare
        keyword mentions (e.g. "он Комиссар" in an investigation story) are
        intentionally ignored to avoid false positives.
        """
        text = payload.text
        claim_match = _CLAIM_ROLE_RE.search(text)
        if claim_match is None:
            return None

        claimed_raw = claim_match.group(1).strip()
        claimed_role = self._canonical_role(claimed_raw)
        if claimed_role is None:
            return None

        # Match against pattern conclusion_role; fall back to #2 for any
        # unmatched direct claim.
        pattern = self._find_pattern_by_conclusion(claimed_role, "claim_role_direct")
        if pattern is None:
            pattern = self._patterns.get("#2")

        return PatternMatch(
            player_id=payload.author_id,
            pattern_id=pattern.id,
            role=claimed_role,
            confidence=pattern.confidence_tier,
            reason=(f"Прямое заявление роли «{claimed_role}» (паттерн #{pattern.id[1:]})"),
        )

    def _detect_investigation_claim(
        self,
        payload: MessagePayload,
        voter_ids: set[int],
    ) -> PatternMatch | None:
        """Detect a claim of investigation (#1). Requires vote confirmation."""
        text = payload.text
        if _INVESTIGATION_RE.search(text) is None:
            return None

        pattern = self._patterns.get("#1")
        if pattern.requires_confirmation and payload.author_id not in voter_ids:
            # No confirming vote yet.
            return None

        return PatternMatch(
            player_id=payload.author_id,
            pattern_id=pattern.id,
            role=pattern.conclusion_role,
            confidence=pattern.confidence_tier,
            reason=(
                f"Заявление о проверке роли, подтверждённое голосованием "
                f"(паттерн #{pattern.id[1:]})"
            ),
        )

    def _detect_bot_investigation(
        self,
        payload: RoleRevealPayload,
        vote_targets: set[int],
    ) -> PatternMatch | None:
        """Detect a bot investigation result (#5). Requires vote confirmation."""
        pattern = self._patterns.get("#5")
        if pattern.requires_confirmation and payload.player_id not in vote_targets:
            return None

        return PatternMatch(
            player_id=payload.player_id,
            pattern_id=pattern.id,
            role=pattern.conclusion_role,
            confidence=pattern.confidence_tier,
            reason=(
                f"Результат проверки из сообщения бота, подтверждён голосованием "
                f"(паттерн #{pattern.id[1:]})"
            ),
        )

    def _confirmed_commissioners(
        self,
        reveals: list[RoleRevealEvent],
        votes: list[VoteEvent],
    ) -> set[int]:
        """Return player ids revealed as Commissioner and confirmed by a vote."""
        confirmed: set[int] = set()
        for reveal in reveals:
            role = self._registry.resolve_alias(reveal.payload.role_name_ru)
            if role is not None and role.name_ru == "Комиссар Каттани":
                # Confirmed if there was a vote targeting this player.
                for vote in votes:
                    if vote.payload.target_id == reveal.payload.player_id:
                        confirmed.add(reveal.payload.player_id)
                        break
        return confirmed
