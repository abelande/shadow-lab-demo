"""
p6lab.correlation.regime_conditioner
====================================
Regime-conditioned template selection — §7.3 of the P6 Lab Spec.

Purpose
-------
Select the subset of patterns/templates valid for the current instrument and
VIX regime before running expensive similarity matching.

VIX buckets (from §3.3 instrument_normalizer)
---------------------------------------------
- low      : VIX < 15
- normal   : 15 <= VIX < 25
- elevated : 25 <= VIX < 35
- high     : VIX >= 35

Selection rules
---------------
1. Pattern must be active: status in {active, mined_approved}
2. Instrument must match if pattern.instruments is non-empty
3. If pattern.regime_specific == True, pattern must have stats/templates for
   the current regime
4. Optional minimum sample size per regime (default 200)

This module is used by:
- CorrelationEngine.match() to pre-filter pattern IDs
- Notebook 06 §09 for regime conditioning validation

References
----------
- Spec §7.3 — per-instrument, per-VIX-bucket selection
- Spec §3.3 — VIX regime buckets
- Spec §5.1 — pattern status + min sample size
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from p6lab.patterns.library import PatternLibrary, PatternDefinition


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_REGIME_SAMPLE_SIZE: int = 200

VIX_LOW_MAX: float = 15.0
VIX_NORMAL_MAX: float = 25.0
VIX_ELEVATED_MAX: float = 35.0

RegimeName = Literal["low", "normal", "elevated", "high"]

# FI bucket cutoffs (Fragility Index, 0..1 scale). Critique §1.4: FI should
# gate patterns at the engine, not live in the feature matrix as a raw input.
FI_CALM_MAX: float = 0.3
FI_ELEVATED_MAX: float = 0.6

FIBucket = Literal["calm", "elevated", "fragile"]


@dataclass
class RegimeSelection:
    """Result of a regime conditioning step.

    Attributes
    ----------
    instrument:
        Target instrument symbol.
    regime:
        Current regime bucket.
    selected_pattern_ids:
        Pattern IDs that passed all filters.
    rejected_pattern_ids:
        Pattern IDs that failed one or more filters.
    """

    instrument: str
    regime: RegimeName
    selected_pattern_ids: list[str]
    rejected_pattern_ids: list[str]


class RegimeConditioner:
    """Filter pattern library by instrument and regime.

    Parameters
    ----------
    min_regime_sample_size:
        Minimum sample size required for a regime-specific pattern.
    """

    def __init__(self, min_regime_sample_size: int = MIN_REGIME_SAMPLE_SIZE) -> None:
        self.min_regime_sample_size = min_regime_sample_size

    def classify_regime(self, vix: float) -> RegimeName:
        """Map raw VIX value to regime bucket."""
        if vix < VIX_LOW_MAX:
            return "low"
        if vix < VIX_NORMAL_MAX:
            return "normal"
        if vix < VIX_ELEVATED_MAX:
            return "elevated"
        return "high"

    def select_patterns(
        self,
        library: PatternLibrary,
        instrument: str,
        regime: RegimeName,
    ) -> RegimeSelection:
        """Return pattern IDs valid for instrument + regime."""
        active = library.get_active_patterns()
        selected: list[str] = []
        rejected: list[str] = []
        for pid, pat in active.items():
            # Instrument filter
            if pat.instruments and instrument not in pat.instruments:
                rejected.append(pid)
                continue
            # Regime support
            if pat.regime_specific and not self.supports_regime(pat, regime):
                rejected.append(pid)
                continue
            selected.append(pid)
        return RegimeSelection(
            instrument=instrument, regime=regime,
            selected_pattern_ids=selected,
            rejected_pattern_ids=rejected,
        )

    def supports_regime(
        self,
        pattern: PatternDefinition,
        regime: RegimeName,
    ) -> bool:
        """A pattern supports a regime when:
          - it has outcome_distribution data for that regime key, with n >= threshold; OR
          - it has no per-regime data at all (treat as regime-agnostic data).
        """
        if not pattern.regime_specific:
            return True
        # Look for regime-keyed outcome data
        key_match = None
        for k in pattern.outcome_distribution.keys():
            if regime in k.lower():
                key_match = k
                break
        if key_match is None:
            # No regime-tagged data — accept if no regime-tagged data at all (data not yet split)
            return all("low" not in k.lower() and "normal" not in k.lower()
                       and "elevated" not in k.lower() and "high" not in k.lower()
                       for k in pattern.outcome_distribution.keys())
        return pattern.outcome_distribution[key_match].n >= self.min_regime_sample_size


class FIConditioner:
    """Gate patterns by Fragility Index bucket.

    Buckets: 'calm' (<0.3) / 'elevated' (0.3-0.6) / 'fragile' (>=0.6).

    Patterns may opt into FI gating by setting ``fi_bucket`` on their
    outcome_distribution sub-keys (e.g. ``'1m_fragile': {...}``) or via
    a top-level ``fi_bucket`` metadata tag. If a pattern has no FI tag,
    it matches any bucket (regime-agnostic).

    This replaces the old approach of passing ``fi_fast``/``fi_full`` as
    raw feature-matrix columns — the engine gate is statistically cleaner
    because FI's own sub-indices overlap with L1/L2 features, so including
    it in X double-counts information.
    """

    def classify_fi(self, fi_fast: float) -> FIBucket:
        """Map FI_fast (0..1) to bucket."""
        if fi_fast < FI_CALM_MAX:
            return "calm"
        if fi_fast < FI_ELEVATED_MAX:
            return "elevated"
        return "fragile"

    def select_patterns(
        self,
        library: PatternLibrary,
        instrument: str,
        fi_bucket: FIBucket,
    ) -> list[str]:
        """Return pattern IDs that fire in this FI bucket."""
        active = library.get_active_patterns()
        out: list[str] = []
        for pid, pat in active.items():
            if pat.instruments and instrument not in pat.instruments:
                continue
            tag = self._pattern_fi_bucket(pat)
            if tag is None or tag == fi_bucket:
                out.append(pid)
        return out

    @staticmethod
    def _pattern_fi_bucket(pattern: PatternDefinition) -> FIBucket | None:
        """Extract the pattern's FI-bucket tag, if any.

        Two encodings are supported:
          - ``pattern.fi_bucket`` attribute on the dataclass (optional)
          - a key suffix on ``outcome_distribution`` like ``'1m_fragile'``
        """
        explicit = getattr(pattern, "fi_bucket", None)
        if explicit in ("calm", "elevated", "fragile"):
            return explicit
        for key in pattern.outcome_distribution.keys():
            k = key.lower()
            if "fragile" in k:
                return "fragile"
            if "calm" in k:
                return "calm"
            # 'elevated' may collide with VIX regime 'elevated'; require exact suffix
            if k.endswith("_elevated_fi") or k.startswith("elevated_fi"):
                return "elevated"
        return None
