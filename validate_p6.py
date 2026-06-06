"""Synthetic validation for P6 pipeline.
Patterns:
1) institutional wall at one level
2) spoof sequence
3) momentum streak
"""
from __future__ import annotations
import time
from typing import List

try:
    from .models import (
        Order, OrderAction, Side, OrderBookLevel, OrderBookSnapshot
    )
    from .pipeline import OrderBookMetaPipeline
except ImportError:
    # Standalone execution: python validate_p6.py
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from p6.models import (  # type: ignore[no-redef]
        Order, OrderAction, Side, OrderBookLevel, OrderBookSnapshot
    )
    from p6.pipeline import OrderBookMetaPipeline  # type: ignore[no-redef]


def _mk_order(oid, side, price, size, ts, action=OrderAction.ADD, aggressive=False):
    return Order(order_id=oid, side=side, price=price, size=size, timestamp_ms=ts, action=action, is_aggressive=aggressive)


def make_synthetic_snapshot() -> OrderBookSnapshot:
    t0 = int(time.time() * 1000)

    # pattern (1): institutional wall — huge volume in few orders at ask 101.0
    wall_orders = [
        _mk_order("w1", Side.ASK, 101.0, 220.0, t0 + 5),
        _mk_order("w2", Side.ASK, 101.0, 180.0, t0 + 6),
    ]

    asks = [
        OrderBookLevel(price=100.5, side=Side.ASK, volume=80.0, order_count=20, orders=[_mk_order("a1", Side.ASK, 100.5, 4.0, t0)]),
        OrderBookLevel(price=101.0, side=Side.ASK, volume=400.0, order_count=2, orders=wall_orders),
        OrderBookLevel(price=101.5, side=Side.ASK, volume=90.0, order_count=18, orders=[_mk_order("a3", Side.ASK, 101.5, 5.0, t0)]),
    ]

    bids = [
        OrderBookLevel(price=100.0, side=Side.BID, volume=120.0, order_count=30, orders=[_mk_order("b1", Side.BID, 100.0, 4.0, t0)]),
        OrderBookLevel(price=99.5, side=Side.BID, volume=100.0, order_count=25, orders=[_mk_order("b2", Side.BID, 99.5, 4.0, t0)]),
        OrderBookLevel(price=99.0, side=Side.BID, volume=90.0, order_count=22, orders=[_mk_order("b3", Side.BID, 99.0, 4.0, t0)]),
    ]

    # pattern (2): spoof sequence pull-before-touch near best ask
    spoof_events = [
        _mk_order("s1", Side.ASK, 100.5, 150.0, t0 + 100, OrderAction.ADD),
        _mk_order("s2", Side.ASK, 100.6, 150.0, t0 + 110, OrderAction.ADD),
        _mk_order("s3", Side.ASK, 100.7, 150.0, t0 + 120, OrderAction.ADD),
        _mk_order("s1", Side.ASK, 100.5, 150.0, t0 + 250, OrderAction.CANCEL),
        _mk_order("s2", Side.ASK, 100.6, 150.0, t0 + 260, OrderAction.CANCEL),
    ]

    # pattern (3): momentum streak — consecutive fills clearing asks
    momentum_trades = [
        _mk_order("t1", Side.ASK, 100.5, 20.0, t0 + 300, OrderAction.FILL, aggressive=True),
        _mk_order("t2", Side.ASK, 101.0, 30.0, t0 + 380, OrderAction.FILL, aggressive=True),
        _mk_order("t3", Side.ASK, 101.5, 25.0, t0 + 460, OrderAction.FILL, aggressive=True),
        _mk_order("t4", Side.ASK, 102.0, 20.0, t0 + 540, OrderAction.FILL, aggressive=True),
    ]

    recent_events = spoof_events + momentum_trades

    return OrderBookSnapshot(
        timestamp_ms=t0 + 600,
        symbol="SYNTH",
        bids=bids,
        asks=asks,
        recent_trades=momentum_trades,
        recent_events=recent_events,
    )


def main():
    snap = make_synthetic_snapshot()
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(snap, combined_regime_output={"regime": "TRENDING", "trend_strength": 0.8, "volatility": 0.4})

    # Verifications
    # 1 institutional wall detected via L1 fragility + high avg order size at 101.0 ask
    wall_level = next((l for l in frame.staircase.ask_levels if abs(l.price - 101.0) < 1e-9), None)
    v1 = bool(wall_level and wall_level.order_count <= 3 and wall_level.avg_order_size >= 100 and wall_level.fragility_score > 0.45)

    # 2 spoof detected
    v2 = any(e.spoof_type.value in ("PULL_BEFORE_TOUCH", "LAYERING", "PHANTOM_WALL") for e in frame.authenticity.spoof_events)

    # 3 momentum streak detected in cup flip
    v3 = frame.game_state.state.value in ("BULL_STREAK", "STOP_RUN") and frame.game_state.streak_length >= 3

    print("=== P6 VALIDATION ===")
    print(f"institutional_wall_detected={v1}")
    print(f"spoof_detected={v2}")
    print(f"momentum_streak_detected={v3}")
    print(f"direction={frame.direction:.3f} confidence={frame.confidence:.3f} urgency={frame.urgency:.3f} size_mult={frame.size_multiplier:.3f}")
    print(f"cup_state={frame.game_state.state.value} pressure={frame.game_state.pressure:.3f} streak_len={frame.game_state.streak_length}")
    print(f"authenticity={frame.authenticity.authenticity_score:.3f} spoof_events={len(frame.authenticity.spoof_events)}")

    ok = v1 and v2 and v3
    print(f"ALL_CHECKS_PASS={ok}")


if __name__ == "__main__":
    main()
