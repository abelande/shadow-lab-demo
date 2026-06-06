"""Tests for p6lab.execution.fill_simulator — bulk + interactive fill model."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from p6lab.execution.fill_simulator import (
    FillOutcome,
    FillSimulator,
    OrderSpec,
)
from p6lab.execution.queue_tracker import Side


# ═══════════════════════════════════════════════════════════════════
# Event fixtures
# ═══════════════════════════════════════════════════════════════════

class _Side(Enum):
    BID = "BID"
    ASK = "ASK"


class _Action(Enum):
    ADD = "ADD"
    CANCEL = "CANCEL"
    MODIFY = "MODIFY"
    FILL = "FILL"
    TRADE = "TRADE"


@dataclass
class _Event:
    order_id: str
    side: _Side
    action: _Action
    price: float
    size: float
    timestamp_ms: int


def add(oid: str, side: _Side, price: float, size: float, ts: int):
    return _Event(oid, side, _Action.ADD, price, size, ts)


def cancel(oid: str, side: _Side, price: float, ts: int):
    return _Event(oid, side, _Action.CANCEL, price, 0, ts)


def fill(side: _Side, price: float, size: float, ts: int):
    return _Event("anon", side, _Action.FILL, price, size, ts)


# ═══════════════════════════════════════════════════════════════════
# Interactive path — full trajectory
# ═══════════════════════════════════════════════════════════════════

class TestInteractiveFill:
    def test_immediate_fill_when_alone_at_level(self):
        """Place order alone at price, a fill at that price hits us."""
        sim = FillSimulator()
        # Prime both sides so _estimate_mid() sees a real book
        events = [
            add("rA", _Side.ASK, 100.25, 10, ts=900),
            # After order is placed at 1000 on bid side, a fill arrives
            fill(_Side.BID, 100.00, 5, ts=1100),
        ]
        order = OrderSpec(
            timestamp_ms=1000, side=Side.BUY,
            price=100.00, size=5, max_horizon_ms=60_000,
        )
        outcome = sim.simulate_interactive(order, events)
        assert outcome.filled is True
        assert outcome.filled_size == 5.0
        assert outcome.fill_reason == "full"
        assert outcome.fill_timestamp_ms == 1100

    def test_trajectory_populated(self):
        sim = FillSimulator()
        events = [
            add("r1", _Side.BID, 100.00, 10, ts=900),
            add("r2", _Side.BID, 100.00, 20, ts=950),
            cancel("r1", _Side.BID, 100.00, ts=1100),
        ]
        order = OrderSpec(
            timestamp_ms=1000, side=Side.BUY,
            price=100.00, size=5, max_horizon_ms=2000,
        )
        outcome = sim.simulate_interactive(order, events)
        # Should have at least "placed" + the cancel event snapshot
        assert len(outcome.trajectory) >= 2
        types = [s.event_type for s in outcome.trajectory]
        assert types[0] == "placed"

    def test_timeout_no_fill(self):
        """No fill event before max_horizon → timeout."""
        sim = FillSimulator()
        events = [
            add("r1", _Side.BID, 100.00, 10, ts=900),
            add("r2", _Side.BID, 100.00, 20, ts=1500),
            add("r3", _Side.BID, 100.00, 30, ts=60_000),  # way past horizon
        ]
        order = OrderSpec(
            timestamp_ms=1000, side=Side.BUY,
            price=100.00, size=5, max_horizon_ms=5000,
        )
        outcome = sim.simulate_interactive(order, events)
        assert outcome.filled is False
        assert outcome.filled_size == 0.0
        assert outcome.fill_reason == "timeout"

    def test_adverse_exit(self):
        """Mid moves against us by adverse_exit_ticks → exit.

        Our virtual BUY @ 100.00 makes us the best bid. For adverse
        to exceed 4 ticks (1.00) we need mid ≤ 99.00, which means the
        ask must drop below 98.00 to pull the mid low enough.
        """
        sim = FillSimulator(tick_size=0.25)
        events = [
            # Initial book: ask 100.25, we add ourselves at bid 100.00 at ts=1000.
            add("aA", _Side.ASK, 100.25, 10, ts=900),
            # After placement, ask crashes to 97.00 → mid = (100.00 + 97.00)/2 = 98.50
            cancel("aA", _Side.ASK, 100.25, ts=1100),
            add("aB", _Side.ASK, 97.00, 10, ts=1100),
        ]
        order = OrderSpec(
            timestamp_ms=1000, side=Side.BUY, price=100.00, size=5,
            adverse_exit_ticks=4, max_horizon_ms=60_000,
        )
        outcome = sim.simulate_interactive(order, events)
        assert outcome.fill_reason == "adverse_exit"
        # Mid at 98.50 → 100.00 - 98.50 = 1.50 = 6 ticks ≥ 4
        assert outcome.adverse_ticks_at_fill >= 4


# ═══════════════════════════════════════════════════════════════════
# Bulk path — multiple orders sharing a stream
# ═══════════════════════════════════════════════════════════════════

class TestBulkFill:
    def test_bulk_returns_in_input_order(self):
        sim = FillSimulator()
        orders = [
            OrderSpec(timestamp_ms=1000, side=Side.BUY, price=100.0, size=5),
            OrderSpec(timestamp_ms=1200, side=Side.BUY, price=99.75, size=3),
            OrderSpec(timestamp_ms=1400, side=Side.SELL, price=100.5, size=2),
        ]
        events = [add("r1", _Side.BID, 100.0, 10, ts=500)]  # minimal stream
        results = sim.simulate_bulk(orders, events)
        assert len(results) == 3
        assert all(isinstance(r, FillOutcome) for r in results)

    def test_bulk_shared_state_affects_multiple_orders(self):
        """Two orders at the same level share the queue state."""
        sim = FillSimulator()
        orders = [
            OrderSpec(timestamp_ms=1000, side=Side.BUY, price=100.0, size=5),
            OrderSpec(timestamp_ms=1100, side=Side.BUY, price=100.0, size=3),
        ]
        events = [
            add("rA", _Side.ASK, 100.25, 1, ts=900),  # prime other side
            fill(_Side.BID, 100.0, 5, ts=1200),  # hits order1 first
            fill(_Side.BID, 100.0, 3, ts=1300),  # hits order2
        ]
        results = sim.simulate_bulk(orders, events)
        assert results[0].filled is True
        assert results[1].filled is True

    def test_bulk_order_activated_only_when_ts_reached(self):
        """An order with timestamp_ms=2000 should not activate before t=2000."""
        sim = FillSimulator()
        orders = [
            OrderSpec(timestamp_ms=2000, side=Side.BUY, price=100.0, size=5),
        ]
        events = [
            add("rA", _Side.ASK, 100.25, 10, ts=900),
            # Fill before activation → should not be consumed by our order
            fill(_Side.BID, 100.0, 5, ts=1000),
            # Later fill should still be present for our late-activated order
            fill(_Side.BID, 100.0, 10, ts=2500),
        ]
        results = sim.simulate_bulk(orders, events)
        # The 1000 fill should not affect us (not yet placed).
        # The 2500 fill should consume us.
        assert results[0].filled is True


# ═══════════════════════════════════════════════════════════════════
# Partial fill handling
# ═══════════════════════════════════════════════════════════════════

class TestPartialFill:
    def test_partial_then_timeout(self):
        """Part of order fills, rest times out → partial reason."""
        sim = FillSimulator()
        # Our order: buy 10 @ 100. Stream has a fill of 3, then nothing.
        events = [
            add("rA", _Side.ASK, 100.25, 10, ts=900),
            fill(_Side.BID, 100.0, 3, ts=1100),
            # Simulate a quiet period to exhaust the horizon
            add("later", _Side.BID, 99.75, 1, ts=60_000),
        ]
        order = OrderSpec(
            timestamp_ms=1000, side=Side.BUY, price=100.0,
            size=10, max_horizon_ms=5000,
        )
        outcome = sim.simulate_interactive(order, events)
        # Only 3 of 10 consumed
        assert outcome.filled_size == 3.0
        assert outcome.filled is False
        assert outcome.fill_reason == "partial"


# ═══════════════════════════════════════════════════════════════════
# Adverse tick measurement (standalone)
# ═══════════════════════════════════════════════════════════════════

class TestAdverseTicks:
    def test_adverse_for_buy_on_price_drop(self):
        sim = FillSimulator(tick_size=0.25)
        events = [
            add("a", _Side.ASK, 100.25, 10, ts=100),
            add("b", _Side.BID, 99.75, 10, ts=100),  # mid = 100.0
        ]
        # BUY filled at 100.0, post-fill mid = 100.0 → adverse = 0
        result = sim._compute_adverse_ticks(
            fill_price=100.0, side=Side.BUY,
            post_fill_events=events, horizons_ms=[1000],
        )
        assert result[1000] == 0

    def test_adverse_for_buy_grows_as_price_falls(self):
        sim = FillSimulator(tick_size=0.25)
        # Mid at event time is near fill price then crashes down
        events = [
            add("aA", _Side.ASK, 100.25, 1, ts=100),
            add("bA", _Side.BID, 100.00, 1, ts=100),
            cancel("aA", _Side.ASK, 100.25, ts=500),
            add("aB", _Side.ASK, 99.00, 1, ts=500),
            cancel("bA", _Side.BID, 100.00, ts=500),
            add("bB", _Side.BID, 98.75, 1, ts=500),  # mid = 98.875
        ]
        result = sim._compute_adverse_ticks(
            fill_price=100.0, side=Side.BUY,
            post_fill_events=events, horizons_ms=[1000],
        )
        # 100.0 - 98.875 = 1.125 → 4.5 ticks → round to 4 or 5
        assert result[1000] >= 4

    def test_adverse_for_sell_on_price_rise(self):
        sim = FillSimulator(tick_size=0.25)
        events = [
            add("aA", _Side.ASK, 100.25, 1, ts=100),
            add("bA", _Side.BID, 100.00, 1, ts=100),
            cancel("aA", _Side.ASK, 100.25, ts=500),
            add("aB", _Side.ASK, 101.50, 1, ts=500),
            cancel("bA", _Side.BID, 100.00, ts=500),
            add("bB", _Side.BID, 101.25, 1, ts=500),  # mid = 101.375
        ]
        result = sim._compute_adverse_ticks(
            fill_price=100.0, side=Side.SELL,
            post_fill_events=events, horizons_ms=[1000],
        )
        # 101.375 - 100.0 = 1.375 → 5.5 ticks → ~5 or 6
        assert result[1000] >= 5


# ═══════════════════════════════════════════════════════════════════
# Output schema
# ═══════════════════════════════════════════════════════════════════

class TestOutputSchema:
    def test_fill_outcome_fields(self):
        sim = FillSimulator()
        events = [add("rA", _Side.ASK, 100.25, 10, ts=100)]
        order = OrderSpec(timestamp_ms=1000, side=Side.BUY, price=100.0, size=5)
        outcome = sim.simulate_interactive(order, events)
        # All documented fields present
        assert hasattr(outcome, "filled")
        assert hasattr(outcome, "filled_size")
        assert hasattr(outcome, "fill_timestamp_ms")
        assert hasattr(outcome, "queue_position_at_entry")
        assert hasattr(outcome, "adverse_ticks_at_fill")
        assert hasattr(outcome, "realized_pnl")
        assert hasattr(outcome, "fill_reason")
        assert hasattr(outcome, "trajectory")

    def test_bulk_omits_trajectory(self):
        sim = FillSimulator()
        orders = [OrderSpec(timestamp_ms=1000, side=Side.BUY, price=100.0, size=5)]
        events = [add("rA", _Side.ASK, 100.25, 10, ts=100)]
        results = sim.simulate_bulk(orders, events)
        assert results[0].trajectory == []
