"""Abstain policy for low-confidence or unstable/volatile conditions."""
from __future__ import annotations
from ..models import RegimeWeights


class AbstainPolicy:
    def should_abstain(
        self,
        regime_weights: RegimeWeights,
        confidence: float,
        authenticity_score: float,
        pressure_abs: float,
    ) -> bool:
        if regime_weights.abstain:
            return True
        if confidence < 0.25:
            return True
        if authenticity_score < 0.3:
            return True
        if pressure_abs < 0.05:
            return True
        return False
