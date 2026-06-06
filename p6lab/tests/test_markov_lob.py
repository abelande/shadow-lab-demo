"""Tests for p6lab.features.markov_lob (Wave 7 Phase 7E)."""
from __future__ import annotations

import pytest

from p6lab.features.markov_lob import (
    MARKOV_FEATURE_NAMES,
    MarkovLOBState,
    classify_lob_state,
    snapshot_markov_features,
    update_markov_lob,
)


def test_classify_bid_heavy_tight() -> None:
    s = classify_lob_state(bid_depth=80.0, ask_depth=20.0, spread=0.01)
    # bid_ratio = 0.8 → depth_bin=0, spread < tight → spread_bin=0 → 0*2+0=0
    assert s == 0


def test_classify_ask_heavy_wide() -> None:
    s = classify_lob_state(bid_depth=20.0, ask_depth=80.0, spread=0.10)
    # bid_ratio = 0.2 → depth_bin=2, spread wide → spread_bin=1 → 2*2+1=5
    assert s == 5


def test_warmup_returns_neutral_features() -> None:
    state = MarkovLOBState()
    snap = snapshot_markov_features(state)
    assert snap["markov_p_price_up_given_state"] == 0.5
    assert snap["markov_state_entropy"] == 1.0


def test_bid_heavy_states_yield_higher_p_up() -> None:
    """Feed a stream where bid-heavy states precede up-moves, ask-heavy
    states precede down-moves. The classifier should learn P(up|bid-heavy)
    > P(up|ask-heavy)."""
    state = MarkovLOBState()
    mid = 100.0
    for i in range(100):
        if i % 2 == 0:
            # bid-heavy → price rises next
            update_markov_lob(
                state, ts_ms=i * 100, bid_depth=80.0, ask_depth=20.0,
                spread=0.01, mid=mid,
            )
            mid += 0.25
        else:
            # ask-heavy → price falls next
            update_markov_lob(
                state, ts_ms=i * 100, bid_depth=20.0, ask_depth=80.0,
                spread=0.01, mid=mid,
            )
            mid -= 0.25

    # Query bid-heavy state P(up)
    state_bid = MarkovLOBState()
    state_bid.states = state.states
    state_bid.mids = state.mids
    state_bid.transition_counts = state.transition_counts
    state_bid.price_up_counts = state.price_up_counts
    state_bid.state_visits = state.state_visits
    # Set current state = 0 (bid-heavy tight)
    state_bid.states[-1] = 0
    p_up_bid = snapshot_markov_features(state_bid)["markov_p_price_up_given_state"]
    # Set current state = 4 (ask-heavy tight)
    state_bid.states[-1] = 4
    p_up_ask = snapshot_markov_features(state_bid)["markov_p_price_up_given_state"]
    assert p_up_bid > p_up_ask


def test_snapshot_returns_all_named_features() -> None:
    state = MarkovLOBState()
    for i in range(30):
        update_markov_lob(
            state, ts_ms=i * 100, bid_depth=50.0, ask_depth=50.0,
            spread=0.01, mid=100.0 + i * 0.01,
        )
    snap = snapshot_markov_features(state)
    for k in MARKOV_FEATURE_NAMES:
        assert k in snap


def test_entropy_range() -> None:
    state = MarkovLOBState()
    for i in range(50):
        update_markov_lob(
            state, ts_ms=i * 100, bid_depth=50.0, ask_depth=50.0,
            spread=0.02, mid=100.0 + i * 0.01,
        )
    snap = snapshot_markov_features(state)
    assert 0.0 <= snap["markov_state_entropy"] <= 1.0
