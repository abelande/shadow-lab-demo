"""Tests for p6lab.features.l1_features — 16 features per SPEC.md §4.1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p6lab.features.l1_features import (
    L1Snapshot,
    L1History,
    L1FeatureNames,
    L1_FEATURE_DIM,
    ROLL_100MS,
    ROLL_250MS,
    ROLL_500MS,
    ROLL_1S,
    bid_ask_imbalance_baseline,
    compute_l1_features,
    compute_l1_series,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

def _snap(
    ts: int,
    bid: float = 100.0,
    ask: float = 100.25,
    bid_sz: float = 50.0,
    ask_sz: float = 50.0,
    tick_size: float = 0.25,
    trade_price: float | None = None,
    trade_size: float | None = None,
    trade_side: str | None = None,
) -> L1Snapshot:
    return L1Snapshot(
        timestamp_ms=ts,
        best_bid=bid, best_ask=ask,
        best_bid_size=bid_sz, best_ask_size=ask_sz,
        last_trade_price=trade_price,
        last_trade_size=trade_size,
        last_trade_side=trade_side,
        tick_size=tick_size,
    )


@pytest.fixture
def history() -> L1History:
    return L1History()


# ═══════════════════════════════════════════════════════════════════
# Snapshot properties
# ═══════════════════════════════════════════════════════════════════

class TestL1Snapshot:
    def test_mid_arithmetic(self):
        s = _snap(1000, bid=100.0, ask=100.50)
        assert s.mid == 100.25

    def test_spread(self):
        s = _snap(1000, bid=100.0, ask=100.25)
        assert s.spread == pytest.approx(0.25)

    def test_microprice_equal_size_equals_mid(self):
        s = _snap(1000, bid=100.0, ask=100.50, bid_sz=10, ask_sz=10)
        assert s.microprice == pytest.approx(100.25)

    def test_microprice_bid_heavy(self):
        """Heavier bid side pulls the microprice toward the ask.

        microprice = (bid × ask_sz + ask × bid_sz) / total
                   = (100 × 10 + 100.50 × 90) / 100
                   = 10045 / 100 = 100.45
        """
        s = _snap(1000, bid=100.0, ask=100.50, bid_sz=90, ask_sz=10)
        # Large bid_sz → numerator weighted toward ask term → microprice > mid
        assert s.microprice > s.mid
        assert s.microprice == pytest.approx(100.45)

    def test_microprice_zero_size_falls_back_to_mid(self):
        s = _snap(1000, bid=100.0, ask=100.50, bid_sz=0, ask_sz=0)
        assert s.microprice == s.mid


# ═══════════════════════════════════════════════════════════════════
# History accumulators
# ═══════════════════════════════════════════════════════════════════

class TestL1History:
    def test_append_snapshot_records_tick_event_on_mid_change(self, history):
        history.append_snapshot(_snap(1000, bid=100.0, ask=100.25))
        history.append_snapshot(_snap(1001, bid=100.0, ask=100.25))
        # No mid change → no tick event recorded
        assert len(history.tick_event_timestamps_ms) == 0

        history.append_snapshot(_snap(1002, bid=100.25, ask=100.50))
        # Mid changed → tick event recorded
        assert history.tick_event_timestamps_ms == [1002]

    def test_trim_drops_old_events(self, history):
        history.append_trade(1000, "bid", 10.0)
        history.append_trade(1500, "ask", 20.0)
        history.append_trade(2500, "bid", 5.0)
        # Now @ ts=3000 with 1s horizon → drop everything before 2000
        history.trim(now_ms=3000, horizon_ms=1000)
        assert history.trade_timestamps_ms == [2500]
        assert history.trade_sides == ["bid"]
        assert history.trade_sizes == [5.0]

    def test_trim_preserves_snapshot_recent_enough(self, history):
        # Snapshot window is horizon + 500ms extra for velocity rollback
        history.append_snapshot(_snap(1000))
        history.append_snapshot(_snap(2000))
        history.append_snapshot(_snap(2800))
        history.trim(now_ms=3000, horizon_ms=1000)
        # Cutoff for snapshots = 3000 - max(1000, 500) - 50 = 1950
        assert [s.timestamp_ms for s in history.snapshots] == [2000, 2800]


# ═══════════════════════════════════════════════════════════════════
# Per-feature tests — 16 features
# ═══════════════════════════════════════════════════════════════════

class TestFeature00SpreadTicks:
    def test_basic(self, history):
        snap = _snap(1000, bid=100.0, ask=100.25, tick_size=0.25)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[0] == pytest.approx(1.0)

    def test_wider_spread(self, history):
        snap = _snap(1000, bid=100.0, ask=101.00, tick_size=0.25)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[0] == pytest.approx(4.0)

    def test_zero_tick_size_safe(self, history):
        snap = _snap(1000, bid=100.0, ask=100.25, tick_size=0.0)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[0] == 0.0


class TestFeature01SpreadBpsL1:
    def test_basic(self, history):
        snap = _snap(1000, bid=100.0, ask=100.25)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        # spread_bps = (0.25 / 100.125) * 10_000 ≈ 24.97
        assert out[1] == pytest.approx(24.9688, rel=1e-3)

    def test_zero_mid_safe(self, history):
        snap = _snap(1000, bid=0.0, ask=0.0)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[1] == 0.0


class TestFeature02_03Sizes:
    def test_raw_sizes(self, history):
        snap = _snap(1000, bid_sz=50, ask_sz=30)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[2] == 50.0
        assert out[3] == 30.0


class TestFeature04TopImbalance:
    def test_symmetric(self, history):
        snap = _snap(1000, bid_sz=50, ask_sz=50)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[4] == 0.0

    def test_bid_heavy_positive(self, history):
        snap = _snap(1000, bid_sz=75, ask_sz=25)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[4] == pytest.approx(0.5)   # (75-25)/100

    def test_ask_heavy_negative(self, history):
        snap = _snap(1000, bid_sz=25, ask_sz=75)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[4] == pytest.approx(-0.5)

    def test_zero_size_safe(self, history):
        snap = _snap(1000, bid_sz=0, ask_sz=0)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[4] == 0.0

    def test_range_bounded(self, history):
        """Imbalance must stay in [-1, +1]."""
        for bid, ask in [(1, 100), (100, 1), (0.001, 1000)]:
            snap = _snap(1000, bid_sz=bid, ask_sz=ask)
            history.append_snapshot(snap)
            out = compute_l1_features(snap, history)
            assert -1.0 <= out[4] <= 1.0
            history.snapshots.clear()


class TestFeature05_06RefreshRates:
    def test_zero_when_no_adds(self, history):
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[5] == 0.0
        assert out[6] == 0.0

    def test_bid_refresh_rate_converts_to_per_second(self, history):
        """3 bid-adds in 100ms = 30 adds/sec."""
        for ts in (910, 950, 990):
            history.append_bid_add(ts)
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[5] == pytest.approx(30.0)

    def test_ask_refresh_rate(self, history):
        for ts in (925, 975):
            history.append_ask_add(ts)
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        # 2 adds / 100ms = 20/sec
        assert out[6] == pytest.approx(20.0)

    def test_events_outside_100ms_excluded(self, history):
        history.append_bid_add(800)  # too old
        history.append_bid_add(950)  # in window
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        # Only 1 event in [900, 1000]
        assert out[5] == pytest.approx(10.0)


class TestFeature07_09Velocities:
    def test_bid_retreat_positive_when_bid_falls(self, history):
        # Bid drops from 100.00 to 99.75 over 250ms
        history.append_snapshot(_snap(750, bid=100.00, ask=100.25))
        history.append_snapshot(_snap(875, bid=99.875, ask=100.125))
        snap = _snap(1000, bid=99.75, ask=100.00)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        # Rate = (99.75 - 100.00) / 0.25s = -1.0 → retreat = +1.0
        assert out[7] > 0.0

    def test_ask_advance_positive_when_ask_rises(self, history):
        history.append_snapshot(_snap(750, bid=100.00, ask=100.00))
        history.append_snapshot(_snap(875, bid=100.00, ask=100.125))
        snap = _snap(1000, bid=100.00, ask=100.25)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        # Rate = (100.25 - 100.00) / 0.25s = +1.0 → advance = +1.0
        assert out[8] > 0.0

    def test_spread_compression_rate_negative_when_tightening(self, history):
        # Spread goes from 0.50 → 0.25
        history.append_snapshot(_snap(750, bid=100.00, ask=100.50))
        history.append_snapshot(_snap(875, bid=100.00, ask=100.375))
        snap = _snap(1000, bid=100.00, ask=100.25)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        # Δ spread = -0.25 over 0.25s → -1.0/sec (tightening = negative)
        assert out[9] < 0.0

    def test_velocities_zero_during_warmup(self, history):
        # Only one snapshot → no window for 250ms rate
        snap = _snap(1000, bid=100.0, ask=100.25)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[7] == 0.0
        assert out[8] == 0.0
        assert out[9] == 0.0


class TestFeature10TickDirectionStreak:
    def test_zero_with_insufficient_history(self, history):
        snap = _snap(1000, bid=100.0, ask=100.25)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[10] == 0.0

    def test_positive_up_streak(self, history):
        """5 consecutive upticks should register +5."""
        prices = [(100.00, 100.25), (100.25, 100.50), (100.50, 100.75),
                  (100.75, 101.00), (101.00, 101.25), (101.25, 101.50)]
        for i, (b, a) in enumerate(prices):
            history.append_snapshot(_snap(1000 + i * 10, bid=b, ask=a))
        snap = history.snapshots[-1]
        out = compute_l1_features(snap, history)
        assert out[10] > 0.0

    def test_negative_down_streak(self, history):
        prices = [(101.50, 101.75), (101.25, 101.50), (101.00, 101.25),
                  (100.75, 101.00), (100.50, 100.75)]
        for i, (b, a) in enumerate(prices):
            history.append_snapshot(_snap(1000 + i * 10, bid=b, ask=a))
        snap = history.snapshots[-1]
        out = compute_l1_features(snap, history)
        assert out[10] < 0.0

    def test_streak_resets_on_reversal(self, history):
        prices = [(100.0, 100.25), (100.25, 100.50), (100.25, 100.25)]
        for i, (b, a) in enumerate(prices):
            history.append_snapshot(_snap(1000 + i * 10, bid=b, ask=a))
        snap = history.snapshots[-1]
        out = compute_l1_features(snap, history)
        # Last move was 100.375 → 100.125 = downtick = -1 streak
        assert out[10] == pytest.approx(-1.0)


class TestFeature11TickAcceleration:
    def test_zero_when_no_ticks(self, history):
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[11] == 0.0

    def test_positive_when_ticks_accelerating(self, history):
        """Few ticks in old half, many in new half → positive accel."""
        # Window is 500ms ending at 1000; halves are [500,750) and [750,1000]
        for ts in (510, 540, 570):
            history.tick_event_timestamps_ms.append(ts)
        for ts in (760, 780, 800, 820, 850, 880, 910, 940, 970, 990):
            history.tick_event_timestamps_ms.append(ts)
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[11] > 0.0

    def test_negative_when_ticks_decelerating(self, history):
        # Many ticks in old half, few in new half → negative accel
        for ts in (510, 540, 570, 600, 630, 660, 690, 720):
            history.tick_event_timestamps_ms.append(ts)
        for ts in (800, 950):
            history.tick_event_timestamps_ms.append(ts)
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[11] < 0.0


class TestFeature12TradeAtBidRatio:
    def test_neutral_when_no_trades(self, history):
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[12] == 0.5

    def test_all_bid_trades(self, history):
        for ts in (100, 200, 500, 800):
            history.append_trade(ts, "bid", 10.0)
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[12] == 1.0

    def test_all_ask_trades(self, history):
        for ts in (100, 200, 500, 800):
            history.append_trade(ts, "ask", 10.0)
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[12] == 0.0

    def test_mixed_half(self, history):
        history.append_trade(100, "bid", 5)
        history.append_trade(500, "ask", 5)
        history.append_trade(800, "bid", 5)
        history.append_trade(900, "ask", 5)
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[12] == 0.5

    def test_excludes_trades_outside_1s_window(self, history):
        history.append_trade(-100, "bid", 5)  # outside
        history.append_trade(500, "ask", 5)   # inside
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[12] == 0.0


class TestFeature13SizeSpikeRatio:
    def test_unity_when_no_trades(self, history):
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[13] == 1.0

    def test_flat_sizes_produce_unity(self, history):
        for ts in (100, 500, 900):
            history.append_trade(ts, "bid", 10.0)
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[13] == pytest.approx(1.0)

    def test_spike_detected(self, history):
        for ts in (100, 300, 500, 700):
            history.append_trade(ts, "bid", 1.0)
        history.append_trade(900, "bid", 50.0)  # spike
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        # max / median = 50 / 1 = 50
        assert out[13] == pytest.approx(50.0)


class TestFeature14MicropriceVelocity:
    def test_zero_during_warmup(self, history):
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out[14] == 0.0

    def test_positive_when_microprice_rising(self, history):
        """Growing ask-size weighted microprice."""
        history.append_snapshot(_snap(750, bid=100.0, ask=100.5, bid_sz=10, ask_sz=10))
        history.append_snapshot(_snap(875, bid=100.0, ask=100.5, bid_sz=50, ask_sz=10))
        snap = _snap(1000, bid=100.0, ask=100.5, bid_sz=90, ask_sz=10)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        # As bid_sz grows, microprice → ask → rising
        assert out[14] > 0.0


class TestFeature15L1ShapeVector:
    def test_produces_scalar_in_range(self, history):
        snap = _snap(1000, bid=100.0, ask=100.25, bid_sz=50, ask_sz=30)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        # L2-normalized + weighted dot product → bounded roughly [-1, 1]
        assert -1.0 <= out[15] <= 1.0

    def test_zero_when_all_components_zero(self, history):
        """Zero book, zero spread, zero imbalance → zero composite."""
        snap = _snap(1000, bid=0, ask=0, bid_sz=0, ask_sz=0)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        # spread_bps = 0, bid/ask_sz = 0, imbalance = 0 → norm = 0 → zero
        assert out[15] == 0.0


# ═══════════════════════════════════════════════════════════════════
# Output shape + types
# ═══════════════════════════════════════════════════════════════════

class TestOutputShape:
    def test_returns_19d_float64(self, history):
        snap = _snap(1000)
        history.append_snapshot(snap)
        out = compute_l1_features(snap, history)
        assert out.shape == (L1_FEATURE_DIM,)   # 19 post-Phase-5A
        assert out.dtype == np.float64

    def test_feature_names_count(self):
        assert len(L1FeatureNames.ALL) == L1_FEATURE_DIM

    def test_feature_names_unique(self):
        assert len(set(L1FeatureNames.ALL)) == L1_FEATURE_DIM


# ═══════════════════════════════════════════════════════════════════
# Bulk series
# ═══════════════════════════════════════════════════════════════════

class TestComputeL1Series:
    def test_empty_returns_empty_dataframe(self):
        df = compute_l1_series([])
        assert df.empty
        assert list(df.columns) == L1FeatureNames.ALL

    def test_shape_matches_16_columns(self):
        snaps = [_snap(1000 + i * 10, bid=100.0 + i * 0.01, ask=100.25 + i * 0.01)
                 for i in range(20)]
        df = compute_l1_series(snaps)
        assert len(df) == 20
        assert list(df.columns) == L1FeatureNames.ALL

    def test_series_matches_per_snapshot(self):
        """Bulk path produces same results as per-snapshot loop."""
        snaps = [_snap(1000 + i * 10, bid=100.0 + i * 0.01, ask=100.25 + i * 0.01)
                 for i in range(50)]
        df = compute_l1_series(snaps)

        history = L1History()
        expected = []
        for snap in snaps:
            history.append_snapshot(snap)
            expected.append(compute_l1_features(snap, history))
        expected_arr = np.array(expected)
        np.testing.assert_allclose(df.values, expected_arr, rtol=1e-10)

    def test_index_is_utc_datetime(self):
        snaps = [_snap(1000 + i, bid=100.0, ask=100.25) for i in range(5)]
        df = compute_l1_series(snaps)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None

    def test_no_nan_values(self):
        """After warmup, features should all be finite."""
        snaps = [_snap(1000 + i * 10, bid=100.0 + i * 0.01, ask=100.25 + i * 0.01,
                       bid_sz=50, ask_sz=50) for i in range(100)]
        df = compute_l1_series(snaps)
        assert df.isna().sum().sum() == 0
        assert np.isfinite(df.values).all()


# ═══════════════════════════════════════════════════════════════════
# Baseline feature
# ═══════════════════════════════════════════════════════════════════

class TestBaseline:
    def test_empty(self):
        result = bid_ask_imbalance_baseline([])
        assert result.empty

    def test_matches_feature_04(self):
        """The baseline Series should equal feature[4] column in compute_l1_series."""
        snaps = [_snap(1000 + i * 10, bid=100.0, ask=100.25,
                       bid_sz=50 + i, ask_sz=50 - i * 0.1) for i in range(30)]
        baseline = bid_ask_imbalance_baseline(snaps)
        df = compute_l1_series(snaps)
        np.testing.assert_allclose(
            baseline.values, df[L1FeatureNames.TOP_IMBALANCE].values, rtol=1e-10)

    def test_range_bounded(self):
        snaps = [_snap(1000, bid_sz=100, ask_sz=1)]
        baseline = bid_ask_imbalance_baseline(snaps)
        assert -1.0 <= baseline.iloc[0] <= 1.0

    def test_series_name(self):
        snaps = [_snap(1000)]
        baseline = bid_ask_imbalance_baseline(snaps)
        assert baseline.name == "bid_ask_imbalance_baseline"


# ═══════════════════════════════════════════════════════════════════
# Integration test — end-to-end through history
# ═══════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_long_run_stays_finite(self):
        """500-snapshot run with all bells/whistles produces finite values."""
        history = L1History()
        # Simulate a realistic stream: periodic trades, add events
        for i in range(500):
            ts = 1000 + i * 10  # 10ms cadence
            # Drift bid/ask slightly
            drift = 0.0001 * i
            snap = _snap(
                ts,
                bid=100.0 + drift,
                ask=100.25 + drift,
                bid_sz=50 + (i % 5),
                ask_sz=45 + (i % 7),
            )
            history.append_snapshot(snap)

            # Every 5th iteration: add bid/ask events
            if i % 5 == 0:
                history.append_bid_add(ts - 5)
            if i % 7 == 0:
                history.append_ask_add(ts - 3)

            # Every 10th: trade
            if i % 10 == 0:
                history.append_trade(ts, "bid" if i % 20 == 0 else "ask", 10.0 + i)

            # Periodically trim
            if i % 100 == 0:
                history.trim(ts)

            out = compute_l1_features(snap, history)
            assert np.isfinite(out).all(), f"Non-finite at i={i}: {out}"
            assert out.shape == (L1_FEATURE_DIM,)
