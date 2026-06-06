"""Tests for p6lab.features.vpin — VPIN with Lee-Ready and BVC."""
from __future__ import annotations

import pytest

from p6lab.features.vpin import (
    ClassificationMethod, VPINConfig, VPINState,
    classify_trade_bvc, classify_trade_lee_ready,
    compute_vpin, update_vpin_state,
)


class TestLeeReady:
    def test_above_mid_is_buy(self):
        assert classify_trade_lee_ready(100.5, 100.0, 100.0, 100.5) == "buy"

    def test_below_mid_is_sell(self):
        # mid = (100 + 100.5)/2 = 100.25; trade at 100 is below
        assert classify_trade_lee_ready(100.0, 100.0, 100.0, 100.5) == "sell"

    def test_tick_rule_fallback_up(self):
        # Trade exactly at mid → use tick rule. Price increased vs prev.
        assert classify_trade_lee_ready(100.25, 100.0, 100.0, 100.5) == "buy"

    def test_tick_rule_fallback_down(self):
        assert classify_trade_lee_ready(100.25, 100.5, 100.0, 100.5) == "sell"


class TestBVC:
    def test_zero_change_split_evenly(self):
        b, s = classify_trade_bvc(0.0, 100.0, 1.0)
        assert b == pytest.approx(50.0)
        assert s == pytest.approx(50.0)

    def test_positive_change_more_buys(self):
        b, s = classify_trade_bvc(1.0, 100.0, 1.0)
        assert b > s

    def test_zero_volatility_returns_split(self):
        b, s = classify_trade_bvc(2.0, 100.0, 0.0)
        assert b == pytest.approx(50.0)
        assert s == pytest.approx(50.0)


class TestStateMachine:
    def test_bucket_finalizes_at_target(self):
        cfg = VPINConfig(bucket_size_fraction=1.0, avg_daily_volume=100.0, window_size=5)
        state = VPINState()
        # bucket target = 100. Add 50 buy then 50 sell → bucket finalizes
        b1 = update_vpin_state(state, cfg, 1, 100.0, 50.0, "buy")
        assert b1 is None
        b2 = update_vpin_state(state, cfg, 2, 100.0, 50.0, "sell")
        assert b2 is not None
        assert b2.buy_volume == 50.0
        assert b2.sell_volume == 50.0
        assert b2.vpin_contribution == pytest.approx(0.0)

    def test_oversized_trade_splits_across_buckets(self):
        cfg = VPINConfig(bucket_size_fraction=1.0, avg_daily_volume=100.0, window_size=5)
        state = VPINState()
        # 250 buy → 2 full buckets + 50 leftover
        update_vpin_state(state, cfg, 1, 100.0, 250.0, "buy")
        assert len(state.buckets) == 2
        assert state.current_buy_volume == 50.0
        assert all(b.vpin_contribution == 1.0 for b in state.buckets)

    def test_compute_vpin_none_below_window(self):
        cfg = VPINConfig(bucket_size_fraction=1.0, avg_daily_volume=10.0, window_size=10)
        state = VPINState()
        for i in range(5):
            update_vpin_state(state, cfg, i, 100.0, 10.0, "buy")
        assert compute_vpin(state, cfg) is None

    def test_compute_vpin_full_window(self):
        cfg = VPINConfig(bucket_size_fraction=1.0, avg_daily_volume=10.0, window_size=3)
        state = VPINState()
        # All buy → vpin contribution = 1.0 per bucket
        for i in range(3):
            update_vpin_state(state, cfg, i, 100.0, 10.0, "buy")
        v = compute_vpin(state, cfg)
        assert v == pytest.approx(1.0)
