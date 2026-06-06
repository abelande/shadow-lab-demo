"""Weighted authenticity score with event-count normalization.

The raw spoof event count shouldn't directly tank the score. Instead:
- Take the max confidence per spoof type (unchanged)
- Weight by type importance (unchanged)
- Apply a decay curve so authenticity degrades smoothly, not cliff-style
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from ..models import AuthenticityProfile, SpoofEvent, SpoofType


@dataclass
class AuthenticityConfig:
    """Tunable weights and thresholds.

    Demo note: the values below are illustrative defaults. Production-tuned
    weights/thresholds are withheld in this public build (see DEMO-SCOPE.md).
    """
    pull_weight: float = 0.35
    layer_weight: float = 0.30
    phantom_weight: float = 0.25
    iceberg_weight: float = 0.10
    # Decay curve: authenticity = 1 - spoof_risk^exponent
    # Higher exponent = more lenient (requires stronger evidence to tank score)
    risk_exponent: float = 1.5
    # Floor: authenticity never drops below this
    floor: float = 0.15


class AuthenticityScorer:
    def __init__(self, config: Optional[AuthenticityConfig] = None):
        self.config = config or AuthenticityConfig()

    def score(self, events: List[SpoofEvent], timestamp_ms: int) -> AuthenticityProfile:
        cfg = self.config

        pull = layer = phantom = stuffing = iceberg = 0.0
        for e in events:
            if e.spoof_type == SpoofType.PULL_BEFORE_TOUCH:
                pull = max(pull, e.confidence)
            elif e.spoof_type == SpoofType.LAYERING:
                layer = max(layer, e.confidence)
            elif e.spoof_type == SpoofType.PHANTOM_WALL:
                phantom = max(phantom, e.confidence)
            elif e.spoof_type == SpoofType.STUFFING:
                stuffing = max(stuffing, e.confidence)
            elif e.spoof_type == SpoofType.ICEBERG:
                iceberg = max(iceberg, e.confidence)

        # Weighted risk — iceberg is informational, not deceptive like spoofing
        spoof_risk = (
            cfg.pull_weight * pull +
            cfg.layer_weight * layer +
            cfg.phantom_weight * phantom +
            cfg.iceberg_weight * iceberg
        )

        # Apply decay curve for smoother degradation
        spoof_risk_curved = min(1.0, spoof_risk ** (1.0 / max(cfg.risk_exponent, 0.1)))
        authenticity = max(cfg.floor, 1.0 - spoof_risk_curved)

        return AuthenticityProfile(
            authenticity_score=authenticity,
            spoof_events=events,
            pull_score=pull,
            layering_score=layer,
            phantom_score=phantom,
            stuffing_score=stuffing,
            timestamp_ms=timestamp_ms,
        )
