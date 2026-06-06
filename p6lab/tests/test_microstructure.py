"""Wave 4 Phase 2 — unit tests for microstructure.py features."""
from __future__ import annotations

import math

import numpy as np
import pytest

from p6lab.features.microstructure import (
    KyleLambdaState,
    MICROSTRUCTURE_FEATURE_NAMES,
    MicrostructureState,
    OFIState,
    RealizedVarianceState,
    RollSpreadState,
    TickRulePINState,
    snapshot_features,
    update_microstructure,
)


# ---------------------------------------------------------------------------
# OFI
# ---------------------------------------------------------------------------


class TestOFI:
    def test_empty_returns_zero(self):
        s = OFIState()
        assert s.ofi(now_ms=1000, window_ms=1000) == 0.0

    def test_signed_sum_within_window(self):
        s = OFIState()
        s.update(100, "buy", 5.0)
        s.update(500, "sell", 2.0)
        s.update(800, "buy", 3.0)
        # Window covers all three
        assert s.ofi(1000, 1000) == 6.0   # 5 - 2 + 3

    def test_trim_excludes_old(self):
        s = OFIState()
        s.update(100, "buy", 10.0)   # 900ms old at now=1000
        s.update(950, "sell", 3.0)
        assert s.ofi(1000, 100) == -3.0   # only the recent sell

    def test_all_sell_gives_negative(self):
        s = OFIState()
        for i in range(5):
            s.update(100 + i * 10, "sell", 1.0)
        assert s.ofi(1000, 1000) == -5.0


# ---------------------------------------------------------------------------
# Realized variance
# ---------------------------------------------------------------------------


class TestRealizedVariance:
    def test_flat_series_gives_zero(self):
        s = RealizedVarianceState()
        for i in range(10):
            s.update(i * 100, 100.0)
        assert s.value() == 0.0

    def test_positive_variance_on_nonflat(self):
        s = RealizedVarianceState()
        # 2% move
        s.update(0, 100.0)
        s.update(100, 102.0)
        v = s.value()
        expected = math.log(1.02) ** 2
        assert abs(v - expected) < 1e-9

    def test_warmup_returns_zero(self):
        s = RealizedVarianceState()
        s.update(0, 100.0)
        assert s.value() == 0.0    # need 2+ mids


# ---------------------------------------------------------------------------
# Roll spread
# ---------------------------------------------------------------------------


class TestRollSpread:
    def test_warmup_returns_zero(self):
        s = RollSpreadState()
        s.update(0, 100.0)
        s.update(100, 100.5)
        assert s.value() == 0.0   # need 3+ Δmids

    def test_positive_autocorrelation_returns_zero(self):
        """Monotonic up-trend has positive cov of Δ with lag(Δ) → Roll returns 0."""
        s = RollSpreadState()
        for i in range(20):
            s.update(i * 100, 100.0 + i * 0.1)
        # Roll returns 0 when cov >= 0; floating-point noise may produce tiny
        # magnitudes, so accept anything below ~1e-6
        assert s.value() < 1e-6

    def test_alternating_ups_and_downs_gives_spread(self):
        """Alternating ±0.5 moves → strong negative autocorrelation → non-zero spread."""
        s = RollSpreadState()
        prices = [100.0, 100.5, 100.0, 100.5, 100.0, 100.5, 100.0, 100.5, 100.0, 100.5]
        for i, p in enumerate(prices):
            s.update(i * 100, p)
        assert s.value() > 0


# ---------------------------------------------------------------------------
# Kyle's λ
# ---------------------------------------------------------------------------


class TestKyleLambda:
    def test_warmup_returns_zero(self):
        s = KyleLambdaState()
        s.update(0, 100.0, 1.0)
        assert s.value() == 0.0

    def test_positive_lambda_on_price_impact(self):
        """Δmid correlates with signed volume → λ > 0."""
        s = KyleLambdaState()
        for i in range(20):
            # Signed vol +1/-1 alternating; mid follows proportionally
            sv = 1.0 if i % 2 else -1.0
            mid = 100.0 + 0.1 * sv * (i + 1)   # impact grows with time
            s.update(i * 100, mid, sv)
        lam = s.value()
        assert lam > 0

    def test_zero_variance_volume_returns_zero(self):
        s = KyleLambdaState()
        for i in range(15):
            s.update(i * 100, 100.0 + i * 0.1, 5.0)   # all same flow
        assert s.value() == 0.0


# ---------------------------------------------------------------------------
# Tick-rule PIN
# ---------------------------------------------------------------------------


class TestTickRulePIN:
    def test_no_trades_returns_zero(self):
        s = TickRulePINState()
        assert s.value() == 0.0

    def test_all_up_ticks_gives_one(self):
        s = TickRulePINState()
        for i in range(10):
            s.update(i * 100, 100.0 + i * 0.1)
        assert s.value() == 1.0

    def test_balanced_ticks_gives_zero(self):
        s = TickRulePINState()
        prices = [100.0, 100.5, 100.0, 100.5, 100.0, 100.5]   # alternating
        for i, p in enumerate(prices):
            s.update(i * 100, p)
        # 5 moves: +, -, +, -, + → signed_sum = +1, total = 5 → 0.2
        assert 0.0 <= s.value() <= 0.4


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class TestAggregator:
    def test_update_and_snapshot(self):
        s = MicrostructureState()
        update_microstructure(
            s, ts_ms=100, mid=100.0,
            trades=[{"price": 100.0, "volume": 5.0, "side": "buy"}],
        )
        update_microstructure(
            s, ts_ms=500, mid=100.5,
            trades=[{"price": 100.5, "volume": 3.0, "side": "sell"}],
        )
        out = snapshot_features(s, now_ms=600, mid=100.5)
        assert set(out.keys()) == set(MICROSTRUCTURE_FEATURE_NAMES)
        assert isinstance(out["ofi_1s"], float)
        assert out["ofi_1s"] == 2.0   # 5 buy - 3 sell

    def test_empty_trades_ok(self):
        s = MicrostructureState()
        update_microstructure(s, ts_ms=100, mid=100.0, trades=None)
        update_microstructure(s, ts_ms=200, mid=100.1, trades=[])
        out = snapshot_features(s, now_ms=300, mid=100.1)
        assert out["ofi_1s"] == 0.0   # no trades
        assert out["realized_variance_30s"] > 0   # mid moved
