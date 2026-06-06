"""Per-regime layer weights."""
from __future__ import annotations
from ..models import RegimeType, RegimeWeights


def get_regime_weights(regime: RegimeType) -> RegimeWeights:
    if regime == RegimeType.TRENDING:
        return RegimeWeights(regime=regime, l1_weight=0.2, l2_weight=0.4, l3_weight=0.3, l4_weight=0.1, abstain=False)
    if regime == RegimeType.RANGING:
        return RegimeWeights(regime=regime, l1_weight=0.4, l2_weight=0.2, l3_weight=0.1, l4_weight=0.3, abstain=False)
    if regime == RegimeType.VOLATILE:
        # all reduced + abstain bias
        return RegimeWeights(regime=regime, l1_weight=0.1, l2_weight=0.1, l3_weight=0.1, l4_weight=0.1, abstain=True)
    return RegimeWeights(regime=regime, l1_weight=0.25, l2_weight=0.25, l3_weight=0.25, l4_weight=0.25, abstain=False)
