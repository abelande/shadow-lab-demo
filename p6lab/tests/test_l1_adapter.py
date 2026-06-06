"""Integration tests for p6lab.features._l1_adapter.

The adapter bridges p6-v2 OrderBookSnapshot → L1Snapshot + L1History.
These tests build synthetic objects conforming to the p6-v2 Protocol
interface and run the adapter end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pytest

from p6lab.features._l1_adapter import L1Adapter, L1AdapterConfig
from p6lab.features.l1_features import L1FeatureNames, compute_l1_features


# ═══════════════════════════════════════════════════════════════════
# Synthetic p6-v2 types conforming to the Protocol interface
# ═══════════════════════════════════════════════════════════════════

class _Side(Enum):
    BID = "BID"
    ASK = "ASK"


class _OrderAction(Enum):
    ADD = "ADD"
    CANCEL = "CANCEL"
    MODIFY = "MODIFY"
    FILL = "FILL"


@dataclass
class _Order:
    order_id: str
    side: _Side
    price: float
    size: float
    timestamp_ms: int
    action: _OrderAction = _OrderAction.ADD
    is_aggressive: bool = False


@dataclass
class _OrderBookLevel:
    price: float
    volume: float
    order_count: int = 1


@dataclass
class _OrderBookSnapshot:
    timestamp_ms: int
    symbol: str
    bids: list = field(default_factory=list)
    asks: list = field(default_factory=list)
    recent_trades: list = field(default_factory=list)
    recent_events: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _snap(ts, bid=100.0, ask=100.25, bid_sz=50.0, ask_sz=50.0,
          trades=None, events=None, symbol="NQ"):
    return _OrderBookSnapshot(
        timestamp_ms=ts,
        symbol=symbol,
        bids=[_OrderBookLevel(bid, bid_sz)],
        asks=[_OrderBookLevel(ask, ask_sz)],
        recent_trades=trades or [],
        recent_events=events or [],
    )


def _trade(order_id, ts, price, size, side=_Side.ASK):
    return _Order(
        order_id=order_id, side=side, price=price, size=size,
        timestamp_ms=ts, action=_OrderAction.FILL,
    )


def _add_event(order_id, ts, price, size, side, is_aggressive=False):
    return _Order(
        order_id=order_id, side=side, price=price, size=size,
        timestamp_ms=ts, action=_OrderAction.ADD, is_aggressive=is_aggressive,
    )


# ═══════════════════════════════════════════════════════════════════
# Adapter — basic conversion
# ═══════════════════════════════════════════════════════════════════

class TestAdapterBasicConversion:
    def test_single_snapshot_converts_correctly(self):
        adapter = L1Adapter(L1AdapterConfig(tick_size=0.25))
        snap = _snap(1000, bid=100.00, ask=100.25, bid_sz=50, ask_sz=30)
        l1_snap = adapter.ingest(snap)

        assert l1_snap.timestamp_ms == 1000
        assert l1_snap.best_bid == 100.00
        assert l1_snap.best_ask == 100.25
        assert l1_snap.best_bid_size == 50.0
        assert l1_snap.best_ask_size == 30.0
        assert l1_snap.tick_size == 0.25

    def test_empty_book_handled(self):
        adapter = L1Adapter()
        snap = _snap(1000)
        snap.bids = []
        snap.asks = []
        l1_snap = adapter.ingest(snap)
        assert l1_snap.best_bid == 0.0
        assert l1_snap.best_ask == 0.0

    def test_last_trade_populated(self):
        adapter = L1Adapter()
        trade = _trade("t1", 999, price=100.15, size=5.0)
        snap = _snap(1000, trades=[trade])
        l1_snap = adapter.ingest(snap)
        assert l1_snap.last_trade_price == 100.15
        assert l1_snap.last_trade_size == 5.0
        # 100.15 < mid (100.125)... wait: mid = (100.0 + 100.25)/2 = 100.125
        # 100.15 > 100.125 → buyer-initiated → "ask"
        assert l1_snap.last_trade_side == "ask"

    def test_history_grows_across_ingests(self):
        adapter = L1Adapter()
        for i in range(5):
            adapter.ingest(_snap(1000 + i * 10))
        assert len(adapter.history.snapshots) == 5


# ═══════════════════════════════════════════════════════════════════
# Trade classification
# ═══════════════════════════════════════════════════════════════════

class TestTradeClassification:
    def test_trade_above_mid_classified_as_ask(self):
        """Trade executed above mid = buyer lifted ask → 'ask'."""
        adapter = L1Adapter()
        trade = _trade("t1", 1000, price=100.20, size=5.0)  # above mid 100.125
        snap = _snap(1000, bid=100.0, ask=100.25, trades=[trade])
        adapter.ingest(snap)
        assert adapter.history.trade_sides == ["ask"]

    def test_trade_below_mid_classified_as_bid(self):
        """Trade below mid = seller hit bid → 'bid'."""
        adapter = L1Adapter()
        trade = _trade("t1", 1000, price=100.05, size=5.0)  # below mid
        snap = _snap(1000, bid=100.0, ask=100.25, trades=[trade])
        adapter.ingest(snap)
        assert adapter.history.trade_sides == ["bid"]

    def test_trade_at_mid_uses_side_tiebreaker(self):
        """At exactly mid: use trade.side. Resting BID side = 'bid'."""
        adapter = L1Adapter()
        trade = _trade("t1", 1000, price=100.125, size=5.0, side=_Side.BID)
        snap = _snap(1000, bid=100.0, ask=100.25, trades=[trade])
        adapter.ingest(snap)
        assert adapter.history.trade_sides == ["bid"]

    def test_duplicate_trades_deduplicated(self):
        """Same trade across 2 snapshots (overlapping recent_trades) counted once."""
        adapter = L1Adapter()
        trade = _trade("t1", 999, price=100.20, size=5.0)
        snap1 = _snap(1000, trades=[trade])
        snap2 = _snap(1100, trades=[trade])  # same trade, next snap
        adapter.ingest(snap1)
        adapter.ingest(snap2)
        assert len(adapter.history.trade_sides) == 1

    def test_sizes_stored_as_absolute_values(self):
        adapter = L1Adapter()
        trade = _trade("t1", 1000, price=100.20, size=-5.0)  # negative
        snap = _snap(1000, trades=[trade])
        adapter.ingest(snap)
        assert adapter.history.trade_sizes == [5.0]


# ═══════════════════════════════════════════════════════════════════
# Passive-add event classification
# ═══════════════════════════════════════════════════════════════════

class TestEventClassification:
    def test_bid_add_at_best_recorded(self):
        adapter = L1Adapter()
        # Warm up with a prior snapshot establishing the best
        adapter.ingest(_snap(900, bid=100.0, ask=100.25))
        event = _add_event("o1", 950, price=100.0, size=10.0, side=_Side.BID)
        snap = _snap(1000, bid=100.0, ask=100.25, events=[event])
        adapter.ingest(snap)
        assert adapter.history.bid_add_timestamps_ms == [950]

    def test_ask_add_at_best_recorded(self):
        adapter = L1Adapter()
        adapter.ingest(_snap(900, bid=100.0, ask=100.25))
        event = _add_event("o2", 950, price=100.25, size=10.0, side=_Side.ASK)
        snap = _snap(1000, bid=100.0, ask=100.25, events=[event])
        adapter.ingest(snap)
        assert adapter.history.ask_add_timestamps_ms == [950]

    def test_add_away_from_best_ignored(self):
        """Adds at non-best prices don't count toward refresh rate."""
        adapter = L1Adapter()
        adapter.ingest(_snap(900, bid=100.0, ask=100.25))
        # Add at 99.75 (one tick below best bid) — should not be recorded
        event = _add_event("o3", 950, price=99.75, size=10.0, side=_Side.BID)
        snap = _snap(1000, bid=100.0, ask=100.25, events=[event])
        adapter.ingest(snap)
        assert adapter.history.bid_add_timestamps_ms == []

    def test_aggressive_add_ignored(self):
        """Aggressive adds (crossing the spread) aren't passive refreshes."""
        adapter = L1Adapter()
        adapter.ingest(_snap(900, bid=100.0, ask=100.25))
        event = _add_event("o4", 950, price=100.25, size=10.0,
                           side=_Side.BID, is_aggressive=True)
        snap = _snap(1000, bid=100.0, ask=100.25, events=[event])
        adapter.ingest(snap)
        # Aggressive bid at ask price → skipped
        assert adapter.history.bid_add_timestamps_ms == []

    def test_cancel_event_ignored(self):
        adapter = L1Adapter()
        adapter.ingest(_snap(900, bid=100.0, ask=100.25))
        event = _Order(
            order_id="c1", side=_Side.BID, price=100.0, size=5.0,
            timestamp_ms=950, action=_OrderAction.CANCEL,
        )
        snap = _snap(1000, bid=100.0, ask=100.25, events=[event])
        adapter.ingest(snap)
        assert adapter.history.bid_add_timestamps_ms == []


# ═══════════════════════════════════════════════════════════════════
# End-to-end with compute_l1_features
# ═══════════════════════════════════════════════════════════════════

class TestEndToEndFeatures:
    def test_pipeline_produces_19d_output(self):
        from p6lab.features.l1_features import L1_FEATURE_DIM
        adapter = L1Adapter()
        for i in range(5):
            adapter.ingest(_snap(1000 + i * 50))
        last_snap = adapter.ingest(_snap(1250))
        features = compute_l1_features(last_snap, adapter.history)
        assert features.shape == (L1_FEATURE_DIM,)   # 19 post-Phase-5A
        assert np.isfinite(features).all()

    def test_refresh_rate_feature_populated_from_events(self):
        adapter = L1Adapter()
        adapter.ingest(_snap(900, bid=100.0, ask=100.25))

        # 3 passive bid adds in the last 100ms
        events = [
            _add_event("o1", 920, price=100.0, size=10.0, side=_Side.BID),
            _add_event("o2", 960, price=100.0, size=10.0, side=_Side.BID),
            _add_event("o3", 990, price=100.0, size=10.0, side=_Side.BID),
        ]
        snap = _snap(1000, bid=100.0, ask=100.25, events=events)
        l1_snap = adapter.ingest(snap)
        features = compute_l1_features(l1_snap, adapter.history)
        # feature[5] = bid_refresh_rate: 3 events / 0.1s = 30/sec
        assert features[5] == pytest.approx(30.0)

    def test_trade_ratio_populated_from_trades(self):
        adapter = L1Adapter()
        adapter.ingest(_snap(500, bid=100.0, ask=100.25))

        trades = [
            _trade("t1", 600, price=100.20, size=5.0),   # ask
            _trade("t2", 700, price=100.05, size=5.0),   # bid
            _trade("t3", 800, price=100.05, size=5.0),   # bid
        ]
        snap = _snap(1000, bid=100.0, ask=100.25, trades=trades)
        l1_snap = adapter.ingest(snap)
        features = compute_l1_features(l1_snap, adapter.history)
        # feature[12] = trade_at_bid_ratio: 2/3
        assert features[12] == pytest.approx(2.0 / 3.0)

    def test_velocity_features_populate_after_multiple_snapshots(self):
        adapter = L1Adapter()
        # Simulate ask rising over 3 snapshots spanning 250ms
        adapter.ingest(_snap(750, bid=100.0, ask=100.0, bid_sz=10, ask_sz=10))
        adapter.ingest(_snap(875, bid=100.0, ask=100.125, bid_sz=10, ask_sz=10))
        last_snap = adapter.ingest(_snap(1000, bid=100.0, ask=100.25, bid_sz=10, ask_sz=10))
        features = compute_l1_features(last_snap, adapter.history)
        # feature[8] = ask_advance_velocity should be positive
        assert features[8] > 0.0


# ═══════════════════════════════════════════════════════════════════
# Trimming / memory bounds
# ═══════════════════════════════════════════════════════════════════

class TestTrimmingBehavior:
    def test_trim_triggered_periodically(self):
        """trim_every_n=5 means trim fires every 5 ingests."""
        adapter = L1Adapter(L1AdapterConfig(tick_size=0.25, trim_every_n=5))

        # Feed 20 old trades first
        for ts in range(0, 1000, 50):
            adapter.history.append_trade(ts, "bid", 1.0)

        # Now ingest 5 snapshots at ts=5000+ — trim should drop all old trades
        for i in range(5):
            adapter.ingest(_snap(5000 + i * 10))

        # After 5 ingests, trim fired at i=5 (i.e., on the 5th ingest)
        # All pre-4000 trades should be gone (1000ms horizon, so anything
        # older than 5040-1000=4040 gets dropped)
        assert all(t >= 4040 for t in adapter.history.trade_timestamps_ms)


# ═══════════════════════════════════════════════════════════════════
# Reset behavior
# ═══════════════════════════════════════════════════════════════════

class TestReset:
    def test_reset_clears_all_state(self):
        adapter = L1Adapter()
        adapter.ingest(_snap(1000, trades=[_trade("t1", 1000, 100.20, 5.0)]))
        adapter.ingest(_snap(2000, events=[
            _add_event("e1", 1500, 100.0, 10.0, _Side.BID)
        ]))
        assert len(adapter.history.snapshots) == 2
        adapter.reset()
        assert adapter.history.snapshots == []
        assert adapter.history.trade_timestamps_ms == []
        assert adapter.history.bid_add_timestamps_ms == []
