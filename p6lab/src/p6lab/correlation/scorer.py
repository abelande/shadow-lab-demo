"""
p6lab.correlation.scorer
========================
Ensemble scoring + confidence tier assignment — §7.2 of the P6 Lab Spec.

Confidence tiers (OB-reference.md §L500-L507)
-----------------------------------------------
    Tier A : ensemble_score ≥ 0.85 → auto-alert + position sizing
    Tier B : 0.72 ≤ score < 0.85  → alert only
    Tier C : 0.60 ≤ score < 0.72  → log only
    < 0.60 : discard (not emitted to consumers)

The scorer converts raw ``MatchResult`` objects from the TemplateMatcher
into ``ScoredMatch`` objects with tier assignments and action labels.

Tier actions and UI integration
--------------------------------
- Tier A → ``correlation_feed.js`` (§10.4) renders green row + auto-alert
- Tier A → ``signal_bar.js`` (existing) lowers detection threshold if FI > 0.6
- Tier B → yellow row, alert-only
- Tier C → grey row, log-only (hidden in default view; visible with filter)

Per-pattern precision targets (notebook 06 §05)
-------------------------------------------------
Each pattern's per-tier precision must exceed >65% on the CPCV validation
set (OB-reference.md §L485).  If a pattern's precision drops below 65%
at tier B, it gets demoted to tier-C-only until retrained.

References
----------
- Spec §7.2 — tier cutoffs, action labels
- OB-reference.md §L500-L507 — tier definitions
- OB-reference.md §L485 — per-pattern precision target >65%
- Spec §10.4 — correlation_feed.js color coding
- Spec §10.5 — FI threshold interaction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from p6lab.patterns.template_matcher import MatchResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Tier cutoff thresholds.
TIER_A_CUTOFF: float = 0.85
TIER_B_CUTOFF: float = 0.72
TIER_C_CUTOFF: float = 0.60

#: Minimum per-pattern precision to allow tier B scoring.
MIN_PRECISION_FOR_TIER_B: float = 0.65

#: Actions per tier (for UI / alerting system).
TIER_ACTIONS: dict[str, str] = {
    "A": "auto_alert_and_size",
    "B": "alert_only",
    "C": "log_only",
    "D": "discard",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ScoredMatch:
    """MatchResult enriched with tier assignment and action label.

    Attributes
    ----------
    pattern_id:
        Library pattern slug.
    ensemble_score:
        Combined score ∈ [0, 1].
    confidence_tier:
        'A', 'B', 'C' (D is discarded before reaching here).
    action:
        Tier-derived action string.
    template_similarity:
        Cosine similarity component.
    mahalanobis_score:
        Feature distance component.
    contextual_score:
        Context component.
    used_euclidean_fallback:
        True if Mahalanobis fell back to Euclidean.
    precision_validated:
        True if this pattern has >65% precision at this tier in CPCV.
    demoted:
        True if the pattern was demoted from a higher tier due to
        insufficient precision.
    """

    pattern_id: str
    ensemble_score: float
    confidence_tier: Literal["A", "B", "C"]
    action: str
    template_similarity: float
    mahalanobis_score: float
    contextual_score: float
    used_euclidean_fallback: bool = False
    precision_validated: bool = True
    demoted: bool = False


# ---------------------------------------------------------------------------
# EnsembleScorer
# ---------------------------------------------------------------------------


class EnsembleScorer:
    """Score and tier-assign pattern matches.

    Parameters
    ----------
    precision_by_pattern:
        Dict[pattern_id → Dict[tier → float]] mapping each pattern's
        per-tier precision from CPCV validation.
        Example: {'iceberg_accumulation': {'A': 0.91, 'B': 0.73, 'C': 0.62}}
        If not provided, all patterns are assumed precision-validated.
    tier_cutoffs:
        Custom tier cutoffs (default: module-level constants).
    """

    def __init__(
        self,
        precision_by_pattern: dict[str, dict[str, float]] | None = None,
        tier_cutoffs: dict[str, float] | None = None,
    ) -> None:
        self.precision_by_pattern = precision_by_pattern or {}
        self.tier_cutoffs = tier_cutoffs or {
            "A": TIER_A_CUTOFF,
            "B": TIER_B_CUTOFF,
            "C": TIER_C_CUTOFF,
        }

    def score(self, match: MatchResult) -> ScoredMatch | None:
        """Convert a raw MatchResult to a ScoredMatch with tier + action.

        Returns None if the match is below Tier C (discarded), OR if the
        pattern has insufficient precision at every tier ≤ natural tier.
        """
        natural = self._assign_tier(match.ensemble_score)
        if natural == "D":
            return None
        # Precision-based demotion
        tier = natural
        demoted = False
        order = ["A", "B", "C"]
        idx = order.index(tier)
        while idx < len(order):
            if self._check_precision(match.pattern_id, order[idx]):
                tier = order[idx]
                break
            idx += 1
            demoted = True
        else:
            return None
        # If we exited the loop because every lower tier failed precision
        if idx >= len(order):
            return None
        return ScoredMatch(
            pattern_id=match.pattern_id,
            ensemble_score=match.ensemble_score,
            confidence_tier=tier,  # type: ignore[arg-type]
            action=TIER_ACTIONS[tier],
            template_similarity=match.template_cosine_similarity,
            mahalanobis_score=match.mahalanobis_distance,
            contextual_score=match.contextual_score,
            used_euclidean_fallback=match.used_euclidean_fallback,
            precision_validated=not demoted,
            demoted=demoted,
        )

    def score_batch(self, matches: list[MatchResult]) -> list[ScoredMatch]:
        """Score a list of matches; filter discards; sort desc."""
        scored = [self.score(m) for m in matches]
        kept = [s for s in scored if s is not None]
        kept.sort(key=lambda s: s.ensemble_score, reverse=True)
        return kept

    def _assign_tier(self, score: float) -> Literal["A", "B", "C", "D"]:
        """Assign tier from score using self.tier_cutoffs."""
        if score >= self.tier_cutoffs["A"]:
            return "A"
        if score >= self.tier_cutoffs["B"]:
            return "B"
        if score >= self.tier_cutoffs["C"]:
            return "C"
        return "D"

    def _check_precision(
        self,
        pattern_id: str,
        tier: str,
    ) -> bool:
        """Return True if pattern has sufficient precision at the given tier.

        Returns True if no precision data is available (assume validated).
        """
        if pattern_id not in self.precision_by_pattern:
            return True
        tier_precision = self.precision_by_pattern[pattern_id].get(tier, 1.0)
        return tier_precision >= MIN_PRECISION_FOR_TIER_B

    def update_precision(
        self,
        pattern_id: str,
        tier_precision: dict[str, float],
    ) -> None:
        """Update per-tier precision for a pattern.

        Called after notebook 06 CPCV validation exports new metrics.
        """
        self.precision_by_pattern[pattern_id] = tier_precision
