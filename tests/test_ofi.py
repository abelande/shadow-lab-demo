"""Tests for OFITracker — price-matched OFI + Chakrabarty hybrid VPIN."""
from __future__ import annotations

import pytest
from p6.models import OrderBookSnapshot, OrderBookLevel, Order, Side, OrderAction
from p6.ofi import OFITracker, OFIConfig


def _level(price: float, volume: float, side: Side = Side.BID) -> OrderBookLevel:
    return OrderBookLevel(price=price, side=side, volume=volume, order_count=1)


def _snap(
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
    trades: list[Order] | None = None,
    ts: int = 1000,
) -> OrderBookSnapshot:
    bid_levels = [_level(p, v, Side.BID) for p, v in (bids or [(100.0, 50)])]
    ask_levels = [_level(p, v, Side.ASK) for p, v in (asks or [(100.25, 50)])]
    return OrderBookSnapshot(
        timestamp_ms=ts,
        symbol="TEST",
        bids=bid_levels,
        asks=ask_levels,
        recent_trades=trades or [],
    )


def _trade(price: float, size: float, side: Side = Side.ASK, ts: int = 1000) -> Order:
    return Order(
        order_id="t1", side=side, price=price, size=size,
        timestamp_ms=ts, action=OrderAction.FILL,
    )


class TestOFIFirstCall:
    def test_returns_zero_on_first_call(self):
        tracker = OFITracker()
        ofi, vpin = tracker.update(_snap())
        assert ofi == 0.0
        assert vpin == 0.0


class TestOFIPriceMatching:
    def test_identical_snapshots_give_zero(self):
        tracker = OFITracker()
        snap = _snap(bids=[(100.0, 50)], asks=[(100.25, 50)])
        tracker.update(snap)
        ofi, _ = tracker.update(snap)
        assert abs(ofi) < 1e-6

    def test_bid_volume_increase_positive_ofi(self):
        tracker = OFITracker()
        tracker.update(_snap(bids=[(100.0, 50)], asks=[(100.25, 50)]))
        ofi, _ = tracker.update(_snap(bids=[(100.0, 80)], asks=[(100.25, 50)]))
        assert ofi > 0  # bid pressure increasing

    def test_ask_volume_increase_negative_ofi(self):
        tracker = OFITracker()
        tracker.update(_snap(bids=[(100.0, 50)], asks=[(100.25, 50)]))
        ofi, _ = tracker.update(_snap(bids=[(100.0, 50)], asks=[(100.25, 80)]))
        assert ofi < 0  # sell pressure increasing

    def test_level_disappears_counted_as_negative_delta(self):
        tracker = OFITracker()
        tracker.update(_snap(
            bids=[(100.0, 50), (99.75, 30)],
            asks=[(100.25, 50)],
        ))
        # Level at 99.75 disappears
        ofi, _ = tracker.update(_snap(
            bids=[(100.0, 50)],
            asks=[(100.25, 50)],
        ))
        assert ofi < 0  # lost bid volume

    def test_new_level_appears_counted_as_positive_delta(self):
        tracker = OFITracker()
        tracker.update(_snap(bids=[(100.0, 50)], asks=[(100.25, 50)]))
        # New level at 99.75 appears
        ofi, _ = tracker.update(_snap(
            bids=[(100.0, 50), (99.75, 40)],
            asks=[(100.25, 50)],
        ))
        assert ofi > 0  # gained bid volume

    def test_price_matched_not_index_matched(self):
        """Regression test: if best bid drops from 100.0 to 99.75 but
        volume stays the same, OFI should reflect the price level
        change correctly, not produce a large spurious delta."""
        tracker = OFITracker()
        tracker.update(_snap(bids=[(100.0, 100)], asks=[(100.25, 50)]))
        # Best bid drops — same total volume but at a lower price
        ofi, _ = tracker.update(_snap(bids=[(99.75, 100)], asks=[(100.25, 50)]))
        # Volume at 100.0 disappeared (-100), volume at 99.75 appeared (+100)
        # But 99.75 is rank 0 (weight 1.0) and 100.0 is also rank 0 in prev
        # The net OFI should NOT be wildly negative like a buggy index-based impl
        # would produce (which would see -100 at "level 0")
        assert abs(ofi) < 50  # bounded, not a -100 spike


class TestVPIN:
    def test_buy_trade_above_mid(self):
        tracker = OFITracker(OFIConfig(vpin_bucket_size=10.0))
        snap = _snap(
            bids=[(100.0, 50)],
            asks=[(100.50, 50)],
            trades=[_trade(price=100.40, size=15.0, side=Side.ASK)],
        )
        tracker.update(snap)
        _, vpin = tracker.update(snap)
        # Trade above mid (100.25) → classified as buy → VPIN = |buy - sell| / total
        assert vpin > 0

    def test_sell_trade_below_mid(self):
        tracker = OFITracker(OFIConfig(vpin_bucket_size=10.0))
        snap = _snap(
            bids=[(100.0, 50)],
            asks=[(100.50, 50)],
            trades=[_trade(price=100.10, size=15.0, side=Side.BID)],
        )
        tracker.update(snap)
        _, vpin = tracker.update(snap)
        assert vpin > 0  # imbalance detected

    def test_vpin_bucket_boundary(self):
        tracker = OFITracker(OFIConfig(vpin_bucket_size=20.0))
        # First snap: 10 contracts (half a bucket)
        snap1 = _snap(trades=[_trade(price=100.40, size=10.0)])
        tracker.update(snap1)
        # No bucket complete yet
        _, vpin1 = tracker.update(_snap())
        # Second: 15 more contracts (completes bucket)
        snap2 = _snap(trades=[_trade(price=100.40, size=15.0)])
        _, vpin2 = tracker.update(snap2)
        assert vpin2 > 0  # bucket completed, VPIN now meaningful


class TestEMAConvergence:
    def test_ema_converges_toward_recent(self):
        tracker = OFITracker(OFIConfig(ema_span=5))
        # Initialize
        tracker.update(_snap(bids=[(100.0, 50)], asks=[(100.25, 50)]))
        # Push positive OFI for several calls
        values = []
        for _ in range(10):
            ofi, _ = tracker.update(_snap(bids=[(100.0, 80)], asks=[(100.25, 50)]))
            values.append(ofi)
        # EMA should be positive and increasing toward steady state
        assert all(v > 0 for v in values[1:])


class TestHybridClassification:
    """Test the Chakrabarty et al. (2007) 5-zone hybrid algorithm.

    Uses a wide spread (bid=100.0, ask=101.0, spread=1.0) so the
    decile boundaries are easy to reason about:
      Zone 3 (quote → buy):  price >= 100.70
      Zone 4 (quote → sell): price <= 100.30
      Zone 5 (tick rule):    100.30 < price < 100.70
    """

    def _wide_snap(self, trades, prev_trades=None):
        """Snapshot with bid=100, ask=101, mid=100.5, spread=1.0."""
        return _snap(
            bids=[(100.0, 50)],
            asks=[(101.0, 50)],
            trades=trades,
        )

    def test_zone1_at_ask_tick_rule_uptick(self):
        """Trade at ask after an uptick → buy."""
        tracker = OFITracker(OFIConfig(vpin_bucket_size=100.0))
        # First snapshot with a trade at mid to set prev_trade_price
        snap1 = self._wide_snap([_trade(price=100.50, size=5.0)])
        tracker.update(snap1)
        # Second snapshot: trade at ask (101.0) — uptick from 100.5 → buy
        snap2 = self._wide_snap([_trade(price=101.0, size=10.0)])
        tracker.update(snap2)
        assert tracker._vpin_buy_vol > 0

    def test_zone2_at_bid_tick_rule_downtick(self):
        """Trade at bid after a downtick → sell."""
        tracker = OFITracker(OFIConfig(vpin_bucket_size=100.0))
        snap1 = self._wide_snap([_trade(price=100.50, size=5.0)])
        tracker.update(snap1)
        # Trade at bid (100.0) — downtick from 100.5 → sell
        snap2 = self._wide_snap([_trade(price=100.0, size=10.0)])
        tracker.update(snap2)
        assert tracker._vpin_sell_vol > 0

    def test_zone3_upper30_quote_rule_buy(self):
        """Trade at 100.80 (relative_pos=0.80) → quote rule → buy."""
        tracker = OFITracker(OFIConfig(vpin_bucket_size=100.0))
        snap = self._wide_snap([_trade(price=100.80, size=10.0)])
        tracker.update(snap)
        assert tracker._vpin_buy_vol == 10.0
        assert tracker._vpin_sell_vol == 0.0

    def test_zone4_lower30_quote_rule_sell(self):
        """Trade at 100.15 (relative_pos=0.15) → quote rule → sell."""
        tracker = OFITracker(OFIConfig(vpin_bucket_size=100.0))
        snap = self._wide_snap([_trade(price=100.15, size=10.0)])
        tracker.update(snap)
        assert tracker._vpin_sell_vol == 10.0
        assert tracker._vpin_buy_vol == 0.0

    def test_zone5_middle_tick_rule_uptick(self):
        """Trade at 100.45 (middle 40%) with prev=100.30 → uptick → buy."""
        tracker = OFITracker(OFIConfig(vpin_bucket_size=100.0))
        # Set prev_trade_price to 100.30
        snap1 = self._wide_snap([_trade(price=100.30, size=5.0)])
        tracker.update(snap1)
        buy_before = tracker._vpin_buy_vol
        # Trade at 100.45 — uptick from 100.30 → buy
        snap2 = self._wide_snap([_trade(price=100.45, size=10.0)])
        tracker.update(snap2)
        assert tracker._vpin_buy_vol > buy_before

    def test_zone5_middle_tick_rule_downtick(self):
        """Trade at 100.45 (middle 40%) with prev=100.60 → downtick → sell."""
        tracker = OFITracker(OFIConfig(vpin_bucket_size=100.0))
        # Set prev_trade_price to 100.60
        snap1 = self._wide_snap([_trade(price=100.60, size=5.0)])
        tracker.update(snap1)
        sell_before = tracker._vpin_sell_vol
        # Trade at 100.45 — downtick from 100.60 → sell
        snap2 = self._wide_snap([_trade(price=100.45, size=10.0)])
        tracker.update(snap2)
        assert tracker._vpin_sell_vol > sell_before

    def test_zone5_first_trade_no_prev_uses_side_fallback(self):
        """First trade in middle zone with no prev price → uses trade.side."""
        tracker = OFITracker(OFIConfig(vpin_bucket_size=100.0))
        # ASK-side fill → fallback_buy=True → classified as buy
        snap = self._wide_snap([_trade(price=100.50, size=10.0, side=Side.ASK)])
        tracker.update(snap)
        assert tracker._vpin_buy_vol == 10.0

    def test_zone_boundary_070_is_quote_buy(self):
        """Trade at exactly relative_pos=0.70 → Zone 3 → buy."""
        tracker = OFITracker(OFIConfig(vpin_bucket_size=100.0))
        # bid=100, ask=101, so 0.70 of spread = 100.70
        snap = self._wide_snap([_trade(price=100.70, size=10.0)])
        tracker.update(snap)
        assert tracker._vpin_buy_vol == 10.0

    def test_zone_boundary_030_is_quote_sell(self):
        """Trade at exactly relative_pos=0.30 → Zone 4 → sell."""
        tracker = OFITracker(OFIConfig(vpin_bucket_size=100.0))
        snap = self._wide_snap([_trade(price=100.30, size=10.0)])
        tracker.update(snap)
        assert tracker._vpin_sell_vol == 10.0

    def test_zero_spread_falls_back_to_side(self):
        """When spread is 0, use trade.side as tiebreaker."""
        tracker = OFITracker(OFIConfig(vpin_bucket_size=100.0))
        snap = _snap(
            bids=[(100.0, 50)],
            asks=[(100.0, 50)],  # zero spread
            trades=[_trade(price=100.0, size=10.0, side=Side.ASK)],
        )
        tracker.update(snap)
        assert tracker._vpin_buy_vol == 10.0  # ASK side → buyer lifted ask

    def test_prev_trade_price_updates_sequentially(self):
        """Verify _prev_trade_price tracks the last trade processed."""
        tracker = OFITracker(OFIConfig(vpin_bucket_size=1000.0))
        snap = self._wide_snap([
            _trade(price=100.40, size=5.0, ts=1000),
            _trade(price=100.60, size=5.0, ts=1001),
            _trade(price=100.50, size=5.0, ts=1002),
        ])
        tracker.update(snap)
        assert tracker._prev_trade_price == 100.50  # last trade


class TestReset:
    def test_reset_clears_state(self):
        tracker = OFITracker()
        tracker.update(_snap(bids=[(100.0, 50)], asks=[(100.25, 50)]))
        tracker.update(_snap(bids=[(100.0, 80)], asks=[(100.25, 50)]))
        tracker.reset()
        ofi, vpin = tracker.update(_snap(bids=[(100.0, 50)], asks=[(100.25, 50)]))
        assert ofi == 0.0  # first call after reset
        assert vpin == 0.0

    def test_reset_clears_prev_trade_price(self):
        tracker = OFITracker()
        snap = _snap(trades=[_trade(price=100.10, size=5.0)])
        tracker.update(snap)
        assert tracker._prev_trade_price is not None
        tracker.reset()
        assert tracker._prev_trade_price is None
