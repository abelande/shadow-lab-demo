"""Tests for p6lab.features.l2_features."""
from __future__ import annotations

import numpy as np
import pytest

from p6lab.features.l2_features import (
    BOOK_SHAPE_VECTOR_DIM, L2_FEATURE_DIM,
    L2FeatureNames, L2History, L2Snapshot,
    compute_book_shape_vector, compute_l2_features, compute_l2_series,
)


def _snap(ts: int = 0, mid: float = 100.0,
          bids: list[tuple[float, float]] | None = None,
          asks: list[tuple[float, float]] | None = None) -> L2Snapshot:
    bids = bids or [(99.75, 10), (99.50, 20)]
    asks = asks or [(100.25, 10), (100.50, 20)]
    levels: list[tuple[float, float, float]] = []
    for p, b in bids:
        levels.append((p, float(b), 0.0))
    for p, a in asks:
        levels.append((p, 0.0, float(a)))
    return L2Snapshot(timestamp_ms=ts, symbol="NQ", mid_price=mid, book_levels=levels)


class TestL2Features:
    def test_returns_correct_dim(self):
        snap = _snap()
        feats = compute_l2_features(snap, L2History())
        assert feats.shape == (L2_FEATURE_DIM,)
        assert np.all(np.isfinite(feats))

    def test_imbalance_sign(self):
        snap = _snap(bids=[(99.75, 100), (99.5, 100)],
                     asks=[(100.25, 1), (100.5, 1)])
        feats = compute_l2_features(snap, L2History())
        assert feats[0] > 0  # bid_ask_imbalance positive when bids dominate

    def test_spread_bps(self):
        snap = _snap(mid=100.0, bids=[(99.75, 10)], asks=[(100.25, 10)])
        feats = compute_l2_features(snap, L2History())
        # 0.50 / 100 * 10000 = 50 bps
        assert feats[3] == pytest.approx(50.0, abs=0.01)


class TestBookShapeVector:
    def test_dim(self):
        bsv = compute_book_shape_vector(_snap())
        assert bsv.shape == (BOOK_SHAPE_VECTOR_DIM,)

    def test_self_normalizing(self):
        bsv = compute_book_shape_vector(_snap())
        assert bsv[:20].sum() == pytest.approx(1.0, abs=1e-6)
        assert bsv[20:].sum() == pytest.approx(1.0, abs=1e-6)


class TestSeries:
    def test_bulk_matches_per_snapshot(self):
        snaps = [_snap(ts=i * 100) for i in range(5)]
        df = compute_l2_series(snaps)
        assert len(df) == 5
        assert df.shape[1] == L2_FEATURE_DIM


# ---------------------------------------------------------------------------
# Wave 4 Phase 1A: recent_events → refresh_rate wiring tests
# ---------------------------------------------------------------------------

class TestRecentEventsWiring:
    """Feed synthetic ADD events through L2Snapshot.recent_events; expect
    refresh_rate (L2 feature [9]) to be non-zero."""

    def test_no_events_gives_zero_refresh_rate(self):
        snap = _snap(ts=1000)
        feats = compute_l2_features(snap, L2History())
        assert feats[9] == 0.0

    def test_add_events_populate_refresh_rate(self):
        """10 ADD events in the last 100ms → refresh_rate > 0."""
        class _E:
            def __init__(self, action_name, ts):
                class _A:
                    name = action_name
                self.action = _A()
                self.timestamp_ms = ts

        events = [_E("ADD", 950 + i * 10) for i in range(10)]
        snap = _snap(ts=1000)
        snap.recent_events = events
        hist = L2History()
        feats = compute_l2_features(snap, hist)
        # refresh_window_ms defaults to 1000; 10 events in 1000ms → 10 events/sec
        assert feats[9] > 0
        assert len(hist.refresh_event_timestamps) == 10

    def test_dict_events_also_work(self):
        """Dict-shaped events (some ingestion paths use dicts) should work."""
        events = [{"action": "ADD", "timestamp_ms": 900 + i}
                  for i in range(5)]
        snap = _snap(ts=1000)
        snap.recent_events = events
        feats = compute_l2_features(snap, L2History())
        assert feats[9] > 0

    def test_non_add_events_ignored(self):
        class _E:
            def __init__(self, action_name):
                class _A:
                    name = action_name
                self.action = _A()
                self.timestamp_ms = 900

        events = [_E("CANCEL"), _E("MODIFY")]
        snap = _snap(ts=1000)
        snap.recent_events = events
        hist = L2History()
        feats = compute_l2_features(snap, hist)
        assert feats[9] == 0.0
        assert len(hist.refresh_event_timestamps) == 0


# ---------------------------------------------------------------------------
# Wave 9 A2a: momentum features (signed_flow, imbalance velocity, streak,
# liquidity withdrawal asymmetry).
# ---------------------------------------------------------------------------


class _Fill:
    """Minimal stub of a FILL event for the L2 feature pipeline."""

    def __init__(
        self, ts_ms: int, side: str, size: float, price: float = 100.0,
    ):
        class _Action:
            name = "FILL"
        self.action = _Action()
        self.timestamp_ms = ts_ms
        self.side = side  # 'B' or 'S'
        self.size = float(size)
        self.price = float(price)


class TestMomentumFeatures:
    """Wave 9 A2a — verify the four new features."""

    def _idx(self, name: str) -> int:
        return L2FeatureNames.ALL.index(name)

    def test_schema_size_and_names(self):
        assert L2_FEATURE_DIM == 18
        assert len(L2FeatureNames.ALL) == 18
        assert L2FeatureNames.ALL[12] == L2FeatureNames.SIGNED_FLOW_60S
        assert L2FeatureNames.ALL[13] == L2FeatureNames.IMBALANCE_VELOCITY_5S
        assert L2FeatureNames.ALL[14] == L2FeatureNames.CURRENT_STREAK_LENGTH
        assert L2FeatureNames.ALL[15] == L2FeatureNames.LIQUIDITY_WITHDRAWAL_ASYM
        assert L2FeatureNames.ALL[16] == L2FeatureNames.CURRENT_STREAK_VELOCITY
        assert L2FeatureNames.ALL[17] == L2FeatureNames.CURRENT_STREAK_VW_STRENGTH

    def test_zero_when_no_fill_events(self):
        """Defensive: snap with no FILL events → all streak/flow features 0."""
        snap = _snap(ts=1000)
        feats = compute_l2_features(snap, L2History())
        assert feats[self._idx(L2FeatureNames.SIGNED_FLOW_60S)] == 0.0
        assert feats[self._idx(L2FeatureNames.CURRENT_STREAK_LENGTH)] == 0.0
        assert feats[self._idx(L2FeatureNames.CURRENT_STREAK_VELOCITY)] == 0.0
        assert feats[self._idx(L2FeatureNames.CURRENT_STREAK_VW_STRENGTH)] == 0.0

    def test_signed_flow_pure_buys_is_positive_one(self):
        snap = _snap(ts=1000)
        snap.recent_events = [
            _Fill(900, "B", 5.0),
            _Fill(950, "B", 3.0),
            _Fill(990, "B", 2.0),
        ]
        feats = compute_l2_features(snap, L2History())
        assert feats[self._idx(L2FeatureNames.SIGNED_FLOW_60S)] == pytest.approx(1.0)

    def test_signed_flow_pure_sells_is_negative_one(self):
        snap = _snap(ts=1000)
        snap.recent_events = [_Fill(950, "S", 4.0), _Fill(990, "S", 1.0)]
        feats = compute_l2_features(snap, L2History())
        assert feats[self._idx(L2FeatureNames.SIGNED_FLOW_60S)] == pytest.approx(-1.0)

    def test_signed_flow_mixed_returns_volume_weighted_ratio(self):
        snap = _snap(ts=1000)
        # Buys=10, sells=4; ratio = (10 - 4)/(10 + 4) = 6/14 ≈ 0.4286
        snap.recent_events = [_Fill(950, "B", 10.0), _Fill(990, "S", 4.0)]
        feats = compute_l2_features(snap, L2History())
        assert feats[self._idx(L2FeatureNames.SIGNED_FLOW_60S)] == pytest.approx(6 / 14)

    def test_signed_flow_drops_old_events_outside_window(self):
        """Events older than 60s are excluded from the rolling sum."""
        hist = L2History()
        # First snap at t=0 with old buy
        snap0 = _snap(ts=0)
        snap0.recent_events = [_Fill(0, "B", 100.0)]
        compute_l2_features(snap0, hist)
        # Second snap at t=70_000 (70s later) with small sell
        snap1 = _snap(ts=70_000)
        snap1.recent_events = [_Fill(70_000, "S", 1.0)]
        feats = compute_l2_features(snap1, hist)
        # The buy at t=0 is now > 60s old → out of window. Only the sell
        # remains → signed_flow = -1.0.
        assert feats[self._idx(L2FeatureNames.SIGNED_FLOW_60S)] == pytest.approx(-1.0)

    def test_streak_length_extends_on_same_side(self):
        """Three same-side fills produce a signed streak length of +3
        (cup_flip's StreakDetector starts a streak on first fill, then
        appends each same-side fill)."""
        snap = _snap(ts=1000)
        snap.recent_events = [
            _Fill(900, "B", 1.0, price=100.00),
            _Fill(950, "B", 1.0, price=100.25),
            _Fill(990, "B", 1.0, price=100.50),
        ]
        feats = compute_l2_features(snap, L2History())
        assert feats[self._idx(L2FeatureNames.CURRENT_STREAK_LENGTH)] == 3.0

    def test_streak_absorbs_one_opposing_within_gap_tolerance(self):
        """With default gap_tolerance=1, a single opposing fill is
        absorbed into the current streak (length keeps growing, side
        unchanged)."""
        hist = L2History()
        s0 = _snap(ts=900)
        s0.recent_events = [
            _Fill(900, "B", 1.0), _Fill(950, "B", 1.0),
        ]
        compute_l2_features(s0, hist)
        # One opposing fill — absorbed; streak stays on the buy side.
        s1 = _snap(ts=1000)
        s1.recent_events = [_Fill(1000, "S", 1.0)]
        feats = compute_l2_features(s1, hist)
        # length=3 (2 buys + 1 absorbed sell), side still ASK → +3.0
        assert feats[self._idx(L2FeatureNames.CURRENT_STREAK_LENGTH)] == 3.0

    def test_streak_flips_after_gap_tolerance_exceeded(self):
        """A second opposing fill exceeds gap_tolerance=1 and closes
        the current streak, starting a new opposite-side streak."""
        hist = L2History()
        s0 = _snap(ts=900)
        s0.recent_events = [
            _Fill(900, "B", 1.0), _Fill(950, "B", 1.0),
        ]
        compute_l2_features(s0, hist)
        # Two opposing fills — second one flips the streak.
        s1 = _snap(ts=1000)
        s1.recent_events = [_Fill(1000, "S", 1.0), _Fill(1050, "S", 1.0)]
        feats = compute_l2_features(s1, hist)
        # First sell absorbed; second sell triggers flip → new BID
        # streak with length=1 → -1.0.
        assert feats[self._idx(L2FeatureNames.CURRENT_STREAK_LENGTH)] == -1.0

    def test_streak_velocity_and_strength_populated_on_multi_level_run(self):
        """A streak that consumes multiple price levels produces non-zero
        velocity (levels/sec) and vw_strength (decayed volume sum)."""
        from p6v2.cup_flip.streak_detector import StreakDetector
        hist = L2History(
            streak_detector=StreakDetector(min_streak_length=1, gap_tolerance=0),
        )
        snap = _snap(ts=5_000)
        # 5 buys, each at a distinct price (5 levels), spread over 4 seconds.
        snap.recent_events = [
            _Fill(1_000, "B", 2.0, price=100.00),
            _Fill(2_000, "B", 3.0, price=100.25),
            _Fill(3_000, "B", 4.0, price=100.50),
            _Fill(4_000, "B", 5.0, price=100.75),
            _Fill(5_000, "B", 6.0, price=101.00),
        ]
        feats = compute_l2_features(snap, hist)
        # Length: 5 buys; signed +5
        assert feats[self._idx(L2FeatureNames.CURRENT_STREAK_LENGTH)] == 5.0
        # Velocity = depth/duration_s = 5 levels / 4 seconds = 1.25
        # Signed by ASK → +1.25
        assert feats[self._idx(L2FeatureNames.CURRENT_STREAK_VELOCITY)] == pytest.approx(1.25)
        # vw_strength positive (bull streak) and bounded
        vw = feats[self._idx(L2FeatureNames.CURRENT_STREAK_VW_STRENGTH)]
        assert vw > 0.0
        assert vw < 100.0   # cup_flip 0.9-decay caps asymptotic sum near 10*max_size

    def test_imbalance_velocity_zero_without_history(self):
        """First snapshot has no history → velocity falls back to 0."""
        snap = _snap(ts=0)
        feats = compute_l2_features(snap, L2History())
        assert feats[self._idx(L2FeatureNames.IMBALANCE_VELOCITY_5S)] == 0.0

    def test_imbalance_velocity_positive_when_imbalance_rises(self):
        hist = L2History()
        # t=0: sell-heavy book (imbalance < 0)
        s0 = _snap(ts=0,
                   bids=[(99.75, 10)],
                   asks=[(100.25, 100)])
        compute_l2_features(s0, hist)
        # t=5000ms (5s later): bid-heavy book (imbalance > 0)
        s1 = _snap(ts=5_000,
                   bids=[(99.75, 100)],
                   asks=[(100.25, 10)])
        feats = compute_l2_features(s1, hist)
        assert feats[self._idx(L2FeatureNames.IMBALANCE_VELOCITY_5S)] > 0

    def test_liquidity_withdrawal_asym_zero_without_history(self):
        snap = _snap(ts=0)
        feats = compute_l2_features(snap, L2History())
        assert feats[self._idx(L2FeatureNames.LIQUIDITY_WITHDRAWAL_ASYM)] == 0.0

    def test_liquidity_withdrawal_asym_positive_when_bid_side_drops(self):
        """Bid side withdraws more than ask side → positive asymmetry."""
        hist = L2History()
        s0 = _snap(ts=0,
                   bids=[(99.75, 100)], asks=[(100.25, 100)])
        compute_l2_features(s0, hist)
        # 5s later: bids halved, asks unchanged
        s1 = _snap(ts=5_000,
                   bids=[(99.75, 50)], asks=[(100.25, 100)])
        feats = compute_l2_features(s1, hist)
        # bid_withdraw=50, ask_withdraw=0 → asymmetry = +1
        assert feats[self._idx(L2FeatureNames.LIQUIDITY_WITHDRAWAL_ASYM)] == pytest.approx(1.0)

    def test_liquidity_withdrawal_asym_negative_when_ask_side_drops(self):
        hist = L2History()
        s0 = _snap(ts=0,
                   bids=[(99.75, 100)], asks=[(100.25, 100)])
        compute_l2_features(s0, hist)
        s1 = _snap(ts=5_000,
                   bids=[(99.75, 100)], asks=[(100.25, 30)])
        feats = compute_l2_features(s1, hist)
        # bid_withdraw=0, ask_withdraw=70 → asymmetry = -1
        assert feats[self._idx(L2FeatureNames.LIQUIDITY_WITHDRAWAL_ASYM)] == pytest.approx(-1.0)

    def test_liquidity_withdrawal_asym_zero_when_both_sides_grow(self):
        """If both sides are adding depth, withdrawal asymmetry = 0."""
        hist = L2History()
        s0 = _snap(ts=0,
                   bids=[(99.75, 100)], asks=[(100.25, 100)])
        compute_l2_features(s0, hist)
        s1 = _snap(ts=5_000,
                   bids=[(99.75, 200)], asks=[(100.25, 200)])
        feats = compute_l2_features(s1, hist)
        assert feats[self._idx(L2FeatureNames.LIQUIDITY_WITHDRAWAL_ASYM)] == 0.0

    def test_features_finite_on_realistic_series(self):
        """Bulk computation: no NaN / inf across a multi-snap series."""
        hist = L2History()
        for i in range(10):
            snap = _snap(ts=i * 100)
            snap.recent_events = [_Fill(i * 100, "B" if i % 2 == 0 else "S", 1.0)]
            feats = compute_l2_features(snap, hist)
            assert np.all(np.isfinite(feats))
