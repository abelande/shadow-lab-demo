"""Regime classifier adapter consuming Combined Regime Engine output."""
from __future__ import annotations
from ..models import RegimeType


class RegimeClassifier:
    def classify(self, combined_regime_output: dict | None) -> RegimeType:
        if not combined_regime_output:
            return RegimeType.UNKNOWN
        label = str(combined_regime_output.get('regime', 'UNKNOWN')).upper()
        if label in RegimeType.__members__:
            return RegimeType[label]
        # heuristic fallback
        trend = float(combined_regime_output.get('trend_strength', 0.0))
        vol = float(combined_regime_output.get('volatility', 0.0))
        if vol > 0.8:
            return RegimeType.VOLATILE
        if trend > 0.6:
            return RegimeType.TRENDING
        if trend < 0.35:
            return RegimeType.RANGING
        return RegimeType.UNKNOWN
