"""Shared fixtures for P6 test suite."""
from __future__ import annotations
import time
import pytest
from p6.models import (
    Order, OrderAction, Side, OrderBookLevel, OrderBookSnapshot,
)


def _mk_order(oid, side, price, size, ts, action=OrderAction.ADD, aggressive=False):
    return Order(
        order_id=oid, side=side, price=price, size=size,
        timestamp_ms=ts, action=action, is_aggressive=aggressive,
    )


@pytest.fixture
def t0() -> int:
    return 1_700_000_000_000


@pytest.fixture
def sample_orders(t0) -> list[Order]:
    return [
        _mk_order("o1", Side.BID, 100.0, 5.0, t0),
        _mk_order("o2", Side.BID, 99.5, 3.0, t0),
        _mk_order("o3", Side.ASK, 100.5, 4.0, t0),
        _mk_order("o4", Side.ASK, 101.0, 6.0, t0),
    ]


@pytest.fixture
def sample_trades(t0) -> list[Order]:
    return [
        _mk_order("t1", Side.ASK, 100.5, 10.0, t0 + 100, OrderAction.FILL, aggressive=True),
        _mk_order("t2", Side.ASK, 101.0, 15.0, t0 + 200, OrderAction.FILL, aggressive=True),
        _mk_order("t3", Side.ASK, 101.5, 20.0, t0 + 300, OrderAction.FILL, aggressive=True),
        _mk_order("t4", Side.ASK, 102.0, 10.0, t0 + 400, OrderAction.FILL, aggressive=True),
    ]


@pytest.fixture
def sample_snapshot(t0, sample_trades) -> OrderBookSnapshot:
    wall_orders = [
        _mk_order("w1", Side.ASK, 101.0, 220.0, t0 + 5),
        _mk_order("w2", Side.ASK, 101.0, 180.0, t0 + 6),
    ]
    asks = [
        OrderBookLevel(
            price=100.5, side=Side.ASK, volume=80.0, order_count=20,
            orders=[_mk_order("a1", Side.ASK, 100.5, 4.0, t0)],
        ),
        OrderBookLevel(
            price=101.0, side=Side.ASK, volume=400.0, order_count=2,
            orders=wall_orders,
        ),
        OrderBookLevel(
            price=101.5, side=Side.ASK, volume=90.0, order_count=18,
            orders=[_mk_order("a3", Side.ASK, 101.5, 5.0, t0)],
        ),
    ]
    bids = [
        OrderBookLevel(
            price=100.0, side=Side.BID, volume=120.0, order_count=30,
            orders=[_mk_order("b1", Side.BID, 100.0, 4.0, t0)],
        ),
        OrderBookLevel(
            price=99.5, side=Side.BID, volume=100.0, order_count=25,
            orders=[_mk_order("b2", Side.BID, 99.5, 4.0, t0)],
        ),
        OrderBookLevel(
            price=99.0, side=Side.BID, volume=90.0, order_count=22,
            orders=[_mk_order("b3", Side.BID, 99.0, 4.0, t0)],
        ),
    ]
    spoof_events = [
        _mk_order("s1", Side.ASK, 100.5, 150.0, t0 + 100, OrderAction.ADD),
        _mk_order("s2", Side.ASK, 100.6, 150.0, t0 + 110, OrderAction.ADD),
        _mk_order("s1", Side.ASK, 100.5, 150.0, t0 + 250, OrderAction.CANCEL),
    ]
    recent_events = spoof_events + sample_trades
    return OrderBookSnapshot(
        timestamp_ms=t0 + 600,
        symbol="SYNTH",
        bids=bids,
        asks=asks,
        recent_trades=sample_trades,
        recent_events=recent_events,
    )
