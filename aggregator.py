"""Cross-layer weighted aggregation => direction/confidence/urgency/size_multiplier."""
from __future__ import annotations
from dataclasses import dataclass
from .models import StaircaseProfile, GameState, ForceVector, AuthenticityProfile, RegimeWeights, AggregatedSignal, RegimeType, CupFlipState


@dataclass
class AggregatorConfig:
    streak_floor: float = 0.4
    force_squash_denom: float = 1.0
    auth_center: float = 0.5
    auth_scale: float = 2.0
    confidence_base: float = 0.6
    confidence_auth_weight: float = 0.4
    urgency_pressure_weight: float = 0.5
    urgency_force_weight: float = 0.5
    urgency_force_cap: float = 10.0
    size_base: float = 0.5
    size_scale: float = 1.5
    min_confidence: float = 0.2
    min_authenticity: float = 0.3


class SignalAggregator:
    def __init__(self, config: AggregatorConfig | None = None) -> None:
        self.config = config or AggregatorConfig()

    def aggregate(
        self,
        staircase: StaircaseProfile,
        game_state: GameState,
        force_vector: ForceVector,
        authenticity: AuthenticityProfile,
        regime_weights: RegimeWeights,
        timestamp_ms: int,
    ) -> AggregatedSignal:
        cfg = self.config

        # L1 direction from staircase imbalance
        l1 = staircase.imbalance_ratio

        # L2 direction from pressure/state
        l2 = game_state.pressure
        if game_state.state == CupFlipState.BULL_STREAK:
            l2 = max(l2, cfg.streak_floor)
        elif game_state.state == CupFlipState.BEAR_STREAK:
            l2 = min(l2, -cfg.streak_floor)

        # L3 direction from spectral force (squash)
        l3 = force_vector.total_force
        l3 = l3 / (abs(l3) + cfg.force_squash_denom) if l3 != 0 else 0.0

        # L4 authenticity penalty contributes directional neutrality; low authenticity reduces conviction
        l4 = (authenticity.authenticity_score - cfg.auth_center) * cfg.auth_scale  # [-1,1]

        direction = (
            regime_weights.l1_weight * l1 +
            regime_weights.l2_weight * l2 +
            regime_weights.l3_weight * l3 +
            regime_weights.l4_weight * l4
        )
        direction = max(-1.0, min(1.0, direction))

        raw_conviction = abs(direction)
        confidence = max(0.0, min(1.0, raw_conviction * (cfg.confidence_base + cfg.confidence_auth_weight * authenticity.authenticity_score)))
        urgency = max(0.0, min(1.0, cfg.urgency_pressure_weight * abs(game_state.pressure) + cfg.urgency_force_weight * min(1.0, abs(force_vector.total_force) / cfg.urgency_force_cap)))

        size_multiplier = 0.0 if regime_weights.abstain else (cfg.size_base + cfg.size_scale * confidence * authenticity.authenticity_score)

        abstain = regime_weights.abstain or confidence < cfg.min_confidence 

        return AggregatedSignal(
            direction=direction,
            confidence=confidence,
            urgency=urgency,
            size_multiplier=0.0 if abstain else size_multiplier,
            regime=regime_weights.regime,
            abstain=abstain,
            components={"L1": l1, "L2": l2, "L3": l3, "L4": l4},
            timestamp_ms=timestamp_ms,
        )
