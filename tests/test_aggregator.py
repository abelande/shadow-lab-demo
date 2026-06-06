"""Tests for SignalAggregator."""
from __future__ import annotations
import pytest
from p6.aggregator import SignalAggregator, AggregatorConfig
from p6.models import (
    StaircaseProfile, GameState, ForceVector, AuthenticityProfile,
    RegimeWeights, RegimeType, CupFlipState, AggregatedSignal,
)


def _make_staircase(imbalance: float = 0.0, ts: int = 1000) -> StaircaseProfile:
    return StaircaseProfile(
        timestamp_ms=ts,
        imbalance_ratio=imbalance,
    )


def _make_game_state(pressure: float = 0.0, state: CupFlipState = CupFlipState.BALANCED) -> GameState:
    return GameState(state=state, pressure=pressure, timestamp_ms=1000)


def _make_force(total: float = 0.0) -> ForceVector:
    return ForceVector(total_force=total, timestamp_ms=1000)


def _make_auth(score: float = 0.8) -> AuthenticityProfile:
    return AuthenticityProfile(authenticity_score=score, timestamp_ms=1000)


def _make_weights(regime: RegimeType = RegimeType.TRENDING, abstain: bool = False) -> RegimeWeights:
    return RegimeWeights(
        regime=regime, l1_weight=0.25, l2_weight=0.25,
        l3_weight=0.25, l4_weight=0.25, abstain=abstain,
    )


def test_aggregator_returns_signal():
    agg = SignalAggregator()
    sig = agg.aggregate(
        staircase=_make_staircase(),
        game_state=_make_game_state(),
        force_vector=_make_force(),
        authenticity=_make_auth(),
        regime_weights=_make_weights(),
        timestamp_ms=1000,
    )
    assert isinstance(sig, AggregatedSignal)


def test_direction_clamped_to_minus_one_one():
    agg = SignalAggregator()
    sig = agg.aggregate(
        staircase=_make_staircase(imbalance=5.0),
        game_state=_make_game_state(pressure=5.0),
        force_vector=_make_force(total=100.0),
        authenticity=_make_auth(score=1.0),
        regime_weights=_make_weights(),
        timestamp_ms=1000,
    )
    assert -1.0 <= sig.direction <= 1.0


def test_abstain_when_regime_abstain():
    agg = SignalAggregator()
    sig = agg.aggregate(
        staircase=_make_staircase(imbalance=0.8),
        game_state=_make_game_state(pressure=0.8),
        force_vector=_make_force(total=5.0),
        authenticity=_make_auth(score=0.9),
        regime_weights=_make_weights(abstain=True),
        timestamp_ms=1000,
    )
    assert sig.abstain is True
    assert sig.size_multiplier == 0.0


def test_abstain_when_low_confidence():
    agg = SignalAggregator()
    # Near-zero inputs should produce low confidence and trigger abstain
    sig = agg.aggregate(
        staircase=_make_staircase(imbalance=0.0),
        game_state=_make_game_state(pressure=0.0),
        force_vector=_make_force(total=0.0),
        authenticity=_make_auth(score=0.5),
        regime_weights=_make_weights(),
        timestamp_ms=1000,
    )
    assert sig.abstain is True


def test_low_authenticity_reduces_size_but_does_not_abstain():
    """Low authenticity reduces size_multiplier via the auth factor in the
    size formula, but no longer triggers abstain. The auth-gated abstain
    was removed because spoof-fade strategies require trading into
    manipulated books — abstaining contradicts the signal."""
    agg = SignalAggregator()
    sig = agg.aggregate(
        staircase=_make_staircase(imbalance=0.9),
        game_state=_make_game_state(pressure=0.9),
        force_vector=_make_force(total=8.0),
        authenticity=_make_auth(score=0.1),
        regime_weights=_make_weights(),
        timestamp_ms=1000,
    )
    assert sig.abstain is False
    # size_multiplier should be small due to low auth in the formula:
    # size = size_base + size_scale * confidence * auth_score
    # With auth=0.1, the auth factor severely limits size.
    assert sig.size_multiplier < 1.0


def test_bull_streak_floor_applied():
    agg = SignalAggregator()
    sig = agg.aggregate(
        staircase=_make_staircase(imbalance=0.0),
        game_state=_make_game_state(pressure=0.1, state=CupFlipState.BULL_STREAK),
        force_vector=_make_force(total=0.0),
        authenticity=_make_auth(score=0.8),
        regime_weights=_make_weights(),
        timestamp_ms=1000,
    )
    # direction should be nudged positive by the streak floor
    assert sig.direction > 0.0


def test_bear_streak_floor_applied():
    agg = SignalAggregator()
    sig = agg.aggregate(
        staircase=_make_staircase(imbalance=0.0),
        game_state=_make_game_state(pressure=-0.1, state=CupFlipState.BEAR_STREAK),
        force_vector=_make_force(total=0.0),
        authenticity=_make_auth(score=0.5),  # neutral L4 so bear streak dominates
        regime_weights=_make_weights(),
        timestamp_ms=1000,
    )
    assert sig.direction < 0.0


def test_config_override_changes_thresholds():
    cfg = AggregatorConfig(min_confidence=0.0, min_authenticity=0.0)
    agg = SignalAggregator(config=cfg)
    sig = agg.aggregate(
        staircase=_make_staircase(imbalance=0.6),
        game_state=_make_game_state(pressure=0.6),
        force_vector=_make_force(total=2.0),
        authenticity=_make_auth(score=0.5),
        regime_weights=_make_weights(),
        timestamp_ms=1000,
    )
    # With thresholds at zero, should NOT abstain for normal inputs
    assert sig.abstain is False


def test_components_dict_has_four_keys():
    agg = SignalAggregator()
    sig = agg.aggregate(
        staircase=_make_staircase(),
        game_state=_make_game_state(),
        force_vector=_make_force(),
        authenticity=_make_auth(),
        regime_weights=_make_weights(),
        timestamp_ms=1000,
    )
    assert set(sig.components.keys()) == {"L1", "L2", "L3", "L4"}
