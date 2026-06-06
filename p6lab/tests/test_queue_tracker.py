"""Tests for p6lab.execution.queue_tracker — MBO-driven FIFO queue tracker."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from p6lab.execution.queue_tracker import (
    MatchingAlgorithm,
    OrderHandle,
    QueuePosition,
    QueueTracker,
    Side,
)


# ═══════════════════════════════════════════════════════════════════
# Synthetic MBO event fixtures
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


def add(oid: str, side: _Side, price: float, size: float, ts: int = 1000):
    return _Event(oid, side, _Action.ADD, price, size, ts)


def cancel(oid: str, side: _Side, price: float, ts: int = 1000):
    return _Event(oid, side, _Action.CANCEL, price, 0, ts)


def modify(oid: str, side: _Side, price: float, new_size: float, ts: int = 1000):
    return _Event(oid, side, _Action.MODIFY, price, new_size, ts)


def fill(side: _Side, price: float, size: float, ts: int = 1000):
    return _Event("anon", side, _Action.FILL, price, size, ts)


# ═══════════════════════════════════════════════════════════════════
# Virtual order placement and position
# ═══════════════════════════════════════════════════════════════════

class TestPlaceLimitOrder:
    def test_returns_handle(self):
        t = QueueTracker()
        h = t.place_limit_order(1000, Side.BUY, 100.0, 5.0)
        assert isinstance(h, OrderHandle)
        assert h.side == Side.BUY
        assert h.price == 100.0
        assert h.size == 5.0

    def test_position_on_empty_level(self):
        t = QueueTracker()
        h = t.place_limit_order(1000, Side.BUY, 100.0, 5.0)
        pos = t.get_position(h)
        assert pos.position_from_front == 0.0
        assert pos.total_at_level == 5.0

    def test_accepts_string_side(self):
        t = QueueTracker()
        h = t.place_limit_order(1000, "buy", 100.0, 5.0)
        assert h.side == Side.BUY


class TestQueuePositionBasic:
    def test_after_resting_orders(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        t.on_event(add("r2", _Side.BID, 100.0, 20, ts=600))
        h = t.place_limit_order(700, Side.BUY, 100.0, 5.0)
        pos = t.get_position(h)
        assert pos.position_from_front == 30.0   # 10 + 20
        assert pos.total_at_level == 35.0        # + our 5

    def test_orders_after_us_dont_affect_position(self):
        t = QueueTracker()
        h = t.place_limit_order(500, Side.BUY, 100.0, 5.0)
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=600))
        t.on_event(add("r2", _Side.BID, 100.0, 20, ts=700))
        pos = t.get_position(h)
        assert pos.position_from_front == 0.0
        assert pos.total_at_level == 35.0

    def test_position_at_different_price_is_independent(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 99.75, 100, ts=500))
        h = t.place_limit_order(600, Side.BUY, 100.0, 5.0)
        pos = t.get_position(h)
        # 99.75 level has 100 contracts but our order is at 100.0 → unaffected
        assert pos.position_from_front == 0.0
        assert pos.total_at_level == 5.0


# ═══════════════════════════════════════════════════════════════════
# ADD events
# ═══════════════════════════════════════════════════════════════════

class TestAdd:
    def test_add_appends_to_back(self):
        t = QueueTracker()
        h = t.place_limit_order(500, Side.BUY, 100.0, 5.0)
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=600))
        pos = t.get_position(h)
        # Our order was placed first → we're ahead of r1
        assert pos.position_from_front == 0.0
        assert pos.total_at_level == 15.0

    def test_add_zero_size_ignored(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 0, ts=500))
        h = t.place_limit_order(600, Side.BUY, 100.0, 5.0)
        pos = t.get_position(h)
        assert pos.total_at_level == 5.0

    def test_level_appears_in_inspector(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        sizes = t.level_sizes(Side.BUY)
        assert sizes == {100.0: 10.0}


# ═══════════════════════════════════════════════════════════════════
# CANCEL events
# ═══════════════════════════════════════════════════════════════════

class TestCancel:
    def test_cancel_ahead_advances_us(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        t.on_event(add("r2", _Side.BID, 100.0, 20, ts=600))
        h = t.place_limit_order(700, Side.BUY, 100.0, 5.0)
        assert t.get_position(h).position_from_front == 30.0
        # r1 cancels → we advance by 10
        t.on_event(cancel("r1", _Side.BID, 100.0, ts=800))
        assert t.get_position(h).position_from_front == 20.0

    def test_cancel_behind_no_change(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        h = t.place_limit_order(600, Side.BUY, 100.0, 5.0)
        t.on_event(add("r2", _Side.BID, 100.0, 20, ts=700))
        pos_before = t.get_position(h)
        t.on_event(cancel("r2", _Side.BID, 100.0, ts=800))
        pos_after = t.get_position(h)
        assert pos_after.position_from_front == pos_before.position_from_front
        # Total shrank though
        assert pos_after.total_at_level == pos_before.total_at_level - 20

    def test_cancel_nonexistent_order_is_safe(self):
        t = QueueTracker()
        # Should not raise
        t.on_event(cancel("nosuch", _Side.BID, 100.0, ts=500))
        assert t.level_sizes(Side.BUY) == {}

    def test_level_disappears_when_empty(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        t.on_event(cancel("r1", _Side.BID, 100.0, ts=600))
        assert t.level_sizes(Side.BUY) == {}


# ═══════════════════════════════════════════════════════════════════
# MODIFY events
# ═══════════════════════════════════════════════════════════════════

class TestModify:
    def test_size_decrease_in_place(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        t.on_event(add("r2", _Side.BID, 100.0, 20, ts=600))
        h = t.place_limit_order(700, Side.BUY, 100.0, 5.0)
        # r1 reduces from 10 to 3 — still ahead of us, priority preserved
        t.on_event(modify("r1", _Side.BID, 100.0, 3, ts=800))
        pos = t.get_position(h)
        assert pos.position_from_front == 23  # 3 + 20

    def test_size_increase_loses_priority_by_default(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        h = t.place_limit_order(600, Side.BUY, 100.0, 5.0)
        # r1 grows from 10 to 15 → CANCEL + ADD at back
        t.on_event(modify("r1", _Side.BID, 100.0, 15, ts=700))
        pos = t.get_position(h)
        # We were second; now we're first, r1 sits behind us
        assert pos.position_from_front == 0.0
        assert pos.total_at_level == 20.0

    def test_size_increase_preserves_priority_when_configured(self):
        t = QueueTracker(modify_loses_priority=False)
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        h = t.place_limit_order(600, Side.BUY, 100.0, 5.0)
        t.on_event(modify("r1", _Side.BID, 100.0, 15, ts=700))
        pos = t.get_position(h)
        assert pos.position_from_front == 15  # grew in place

    def test_price_change_treated_as_cancel_plus_add(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 99.75, 10, ts=500))
        t.on_event(add("r2", _Side.BID, 100.0, 20, ts=600))
        h = t.place_limit_order(700, Side.BUY, 100.0, 5.0)
        assert t.get_position(h).position_from_front == 20.0
        # r1 moves from 99.75 to 100.0 — behind us now
        t.on_event(modify("r1", _Side.BID, 100.0, 10, ts=800))
        pos = t.get_position(h)
        assert pos.position_from_front == 20.0  # r2 still ahead
        assert pos.total_at_level == 35.0       # + r1 + our 5 + r2
        assert t.level_sizes(Side.BUY).get(99.75) is None

    def test_size_to_zero_cancels(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        t.on_event(modify("r1", _Side.BID, 100.0, 0, ts=600))
        assert t.level_sizes(Side.BUY) == {}


# ═══════════════════════════════════════════════════════════════════
# FILL / TRADE events
# ═══════════════════════════════════════════════════════════════════

class TestFill:
    def test_fill_consumes_from_front_fifo(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        t.on_event(add("r2", _Side.BID, 100.0, 20, ts=600))
        t.on_event(fill(_Side.BID, 100.0, 5, ts=700))
        # 5 eaten from front: r1 now 5, r2 still 20
        assert t.level_sizes(Side.BUY) == {100.0: 25.0}
        counts = t.level_order_count(Side.BUY)
        assert counts[100.0] == 2

    def test_fill_consumes_whole_entries(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        t.on_event(add("r2", _Side.BID, 100.0, 20, ts=600))
        t.on_event(fill(_Side.BID, 100.0, 25, ts=700))
        # r1 wiped out (10), r2 partially consumed by 15 → 5 left
        assert t.level_sizes(Side.BUY) == {100.0: 5.0}
        counts = t.level_order_count(Side.BUY)
        assert counts[100.0] == 1

    def test_fill_advances_virtual_order_when_ahead(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        h = t.place_limit_order(600, Side.BUY, 100.0, 5.0)
        pos_before = t.get_position(h).position_from_front
        assert pos_before == 10.0
        t.on_event(fill(_Side.BID, 100.0, 7, ts=700))
        pos_after = t.get_position(h).position_from_front
        assert pos_after == 3.0  # r1 now at 3, we're still 2nd

    def test_fill_consumes_virtual_order(self):
        t = QueueTracker()
        h = t.place_limit_order(500, Side.BUY, 100.0, 5.0)
        t.on_event(fill(_Side.BID, 100.0, 5, ts=600))
        # Our virtual order eaten → removed from tracker
        assert t.level_sizes(Side.BUY) == {}

    def test_trade_event_treated_as_fill(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        t.on_event(_Event("t1", _Side.BID, _Action.TRADE, 100.0, 5, 600))
        assert t.level_sizes(Side.BUY) == {100.0: 5.0}

    def test_fill_at_empty_level_is_safe(self):
        t = QueueTracker()
        # No adds, just a fill — should not raise
        t.on_event(fill(_Side.BID, 100.0, 5, ts=500))
        assert t.level_sizes(Side.BUY) == {}


# ═══════════════════════════════════════════════════════════════════
# Pro-rata matching
# ═══════════════════════════════════════════════════════════════════

class TestProRata:
    def test_proportional_consumption(self):
        t = QueueTracker(matching_algorithm=MatchingAlgorithm.PRO_RATA)
        t.on_event(add("r1", _Side.BID, 100.0, 40, ts=500))
        t.on_event(add("r2", _Side.BID, 100.0, 60, ts=600))
        # Fill 10 → r1 gets 4, r2 gets 6
        t.on_event(fill(_Side.BID, 100.0, 10, ts=700))
        sizes = t.level_sizes(Side.BUY)
        assert sizes[100.0] == pytest.approx(90.0)


# ═══════════════════════════════════════════════════════════════════
# Fill probability estimate
# ═══════════════════════════════════════════════════════════════════

class TestFillProbability:
    def test_front_of_queue_high_probability(self):
        t = QueueTracker()
        h = t.place_limit_order(500, Side.BUY, 100.0, 5.0)
        pos = t.get_position(h)
        # Alone at level: position 0 of total 5 → p_fill = 1.0
        assert pos.fill_probability_estimate == 1.0

    def test_back_of_queue_low_probability(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 95, ts=500))
        h = t.place_limit_order(600, Side.BUY, 100.0, 5.0)
        pos = t.get_position(h)
        # Position 95 of 100 → p_fill = 0.05
        assert pos.fill_probability_estimate == pytest.approx(0.05)

    def test_empty_level_zero_probability(self):
        pos = QueueTracker._fill_probability(position_from_front=0, total_at_level=0)
        assert pos == 0.0


# ═══════════════════════════════════════════════════════════════════
# Cancellation and cleanup
# ═══════════════════════════════════════════════════════════════════

class TestCancelOrder:
    def test_cancel_removes_virtual_order(self):
        t = QueueTracker()
        h = t.place_limit_order(500, Side.BUY, 100.0, 5.0)
        assert t.level_sizes(Side.BUY) == {100.0: 5.0}
        t.cancel_order(h)
        assert t.level_sizes(Side.BUY) == {}

    def test_cancel_unknown_handle_safe(self):
        t = QueueTracker()
        # Fabricate a handle that was never placed
        fake = OrderHandle(handle_id=99, side=Side.BUY, price=100.0,
                           size=5, timestamp_ms=500, _internal_id="virt-x")
        t.cancel_order(fake)   # no exception

    def test_reset_clears_all_state(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.BID, 100.0, 10, ts=500))
        t.place_limit_order(500, Side.BUY, 100.0, 5.0)
        t.reset()
        assert t.level_sizes(Side.BUY) == {}
        assert t.get_all_positions() == []


# ═══════════════════════════════════════════════════════════════════
# Side parsing & edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEventParsing:
    def test_accepts_ask_side_from_mbo(self):
        t = QueueTracker()
        t.on_event(add("r1", _Side.ASK, 100.25, 10, ts=500))
        sizes = t.level_sizes(Side.SELL)
        assert sizes == {100.25: 10.0}

    def test_missing_order_id_ignored(self):
        t = QueueTracker()
        ev = _Event("", _Side.BID, _Action.ADD, 100.0, 10, 500)
        t.on_event(ev)
        assert t.level_sizes(Side.BUY) == {}

    def test_unknown_action_ignored(self):
        @dataclass
        class _Weird:
            order_id: str
            side: _Side
            action: Enum   # intentionally weird
            price: float
            size: float
            timestamp_ms: int

        class _UnknownAct(Enum):
            WEIRD = "WEIRD"

        t = QueueTracker()
        ev = _Weird("r1", _Side.BID, _UnknownAct.WEIRD, 100.0, 10, 500)
        t.on_event(ev)
        assert t.level_sizes(Side.BUY) == {}


# ═══════════════════════════════════════════════════════════════════
# Multiple virtual orders
# ═══════════════════════════════════════════════════════════════════

class TestMultipleVirtual:
    def test_two_virtual_orders_at_same_level(self):
        t = QueueTracker()
        h1 = t.place_limit_order(500, Side.BUY, 100.0, 5.0)
        h2 = t.place_limit_order(600, Side.BUY, 100.0, 3.0)
        assert t.get_position(h1).position_from_front == 0.0
        assert t.get_position(h2).position_from_front == 5.0

    def test_virtual_orders_at_different_levels(self):
        t = QueueTracker()
        h1 = t.place_limit_order(500, Side.BUY, 100.0, 5.0)
        h2 = t.place_limit_order(600, Side.BUY, 99.75, 3.0)
        assert t.get_position(h1).total_at_level == 5.0
        assert t.get_position(h2).total_at_level == 3.0

    def test_get_all_positions(self):
        t = QueueTracker()
        t.place_limit_order(500, Side.BUY, 100.0, 5.0)
        t.place_limit_order(600, Side.SELL, 100.5, 3.0)
        positions = t.get_all_positions()
        assert len(positions) == 2
