"""C++ order book construction — rendering-only pipeline.

Architecture:
  Databento feed → Python OrderBook (source of truth) → C++ Book (rendering) → bid/ask/dom/tape output

The C++ module is used ONLY for order book construction and rendering output
(bid_bars, ask_bars, dom_rows, tape, stats). All 5-layer analysis (staircase,
cup flip, spectral force, spoof detection, regime, aggregator) runs in the
Python OrderBookMetaPipeline.

DEPRECATED LAYERS (retained in core/src but no longer called from Python):
  - C++ staircase (fragility scoring)
  - C++ cup flip (tape dynamics)
  - C++ spectral force (FFT decomposition)
  - C++ spoof detection (pull-before-touch, layering, iceberg, phantom wall)
  - C++ regime context (regime classification + weights)
  - C++ aggregator (signal aggregation)
  These had parity bugs with Python (different weights in L1, different sign
  detection in L3) and produced degraded output due to synthetic top-10
  book reconstruction. They will be rebuilt as event-driven C++ layers once
  the Python implementations are calibrated and validated.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# Load C++ module
_cpp_available = False
_p6 = None


def _load_cpp():
    global _cpp_available, _p6
    core_build = os.path.join(os.path.dirname(__file__), 'core', 'build')
    if core_build not in sys.path:
        sys.path.insert(0, core_build)
    try:
        import _p6_core
        _p6 = _p6_core
        _cpp_available = True
        logger.info("C++ core engine loaded (rendering-only mode)")
    except ImportError as e:
        logger.warning("C++ core not available: %s", e)


_load_cpp()


def is_cpp_available() -> bool:
    """Check if the C++ module is loaded and usable."""
    return _cpp_available


class CppBookRenderer:
    """C++ order book construction for rendering output only.

    Builds bid_bars, ask_bars, dom_rows, tape, and stats from the authoritative
    Python OrderBookSnapshot. Does NOT run any analysis layers — those are
    handled by OrderBookMetaPipeline in Python.

    Usage:
        renderer = CppBookRenderer()
        output = renderer.render(snapshot)
        # output has: bid_bars, ask_bars, dom_rows, tape, stats, timestamp_ms, symbol
    """

    def __init__(self, book_levels: int = 10):
        if not _cpp_available:
            raise RuntimeError("C++ core not available")

        self._book = _p6.OrderBook(book_levels)
        self._book_levels = book_levels
        self._side_map = {
            'BID': _p6.Side.BID,
            'ASK': _p6.Side.ASK,
        }
        logger.info("CppBookRenderer initialized (rendering-only, %d levels)", book_levels)

    def render(self, snapshot) -> dict:
        """Build rendering output from a Python OrderBookSnapshot.

        Returns a dict with: bid_bars, ask_bars, dom_rows, tape, stats,
        timestamp_ms, symbol. No analysis fields.
        """
        ts = snapshot.timestamp_ms or int(time.time() * 1000)

        # Rebuild C++ book from authoritative Python book state
        self._book.clear()
        for lvl in snapshot.bids[:self._book_levels]:
            order = _p6.Order()
            order.order_id = 0
            order.side = _p6.Side.BID
            order.price = float(lvl.price)
            order.size = float(lvl.volume)
            order.timestamp_ms = ts
            order.action = _p6.OrderAction.ADD
            order.is_aggressive = False
            self._book.apply(order)

        for lvl in snapshot.asks[:self._book_levels]:
            order = _p6.Order()
            order.order_id = 0
            order.side = _p6.Side.ASK
            order.price = float(lvl.price)
            order.size = float(lvl.volume)
            order.timestamp_ms = ts
            order.action = _p6.OrderAction.ADD
            order.is_aggressive = False
            self._book.apply(order)

        cpp_snap = self._book.build_snapshot(ts, [], [])

        # Build bid/ask bars from C++ snapshot
        bid_bars = [
            {'price': lvl.price, 'side': 'BID', 'volume': lvl.volume, 'order_count': lvl.order_count}
            for lvl in cpp_snap.bids()
        ]
        ask_bars = [
            {'price': lvl.price, 'side': 'ASK', 'volume': lvl.volume, 'order_count': lvl.order_count}
            for lvl in cpp_snap.asks()
        ]

        # DOM rows (interleaved)
        dom_rows = [
            {**b, 'cumulative_volume': b['volume']} for b in bid_bars
        ] + [
            {**a, 'cumulative_volume': a['volume']} for a in ask_bars
        ]

        # Tape from recent trades
        tape = [
            {
                'timestamp_ms': t.timestamp_ms,
                'side': t.side.value,
                'price': float(t.price),
                'size': float(t.size),
            }
            for t in (snapshot.recent_trades or [])[-30:]
        ]

        # Stats from events
        events = snapshot.recent_events or []
        trades = snapshot.recent_trades or []
        buy_vol = sum(float(t.size) for t in trades if t.side.value == 'BID')
        sell_vol = sum(float(t.size) for t in trades if t.side.value == 'ASK')

        stats = {
            'cvd': buy_vol - sell_vol,
            'trades_per_sec': len(trades) * (1000.0 / max(1, len(events))),
            'add_count': sum(1 for e in events if e.action.value == 'ADD'),
            'cancel_count': sum(1 for e in events if e.action.value == 'CANCEL'),
            'modify_count': sum(1 for e in events if e.action.value == 'MODIFY'),
            'fill_count': sum(1 for e in events if e.action.value == 'FILL'),
            'live_orders': int(self._book.total_orders()),
        }

        return {
            'timestamp_ms': ts,
            'symbol': getattr(snapshot, 'symbol', None) or 'UNKNOWN',
            'bid_bars': bid_bars,
            'ask_bars': ask_bars,
            'dom_rows': dom_rows,
            'tape': tape,
            'stats': stats,
        }


# ── Backward compatibility ───────────────────────────────────────────
# CppAcceleratedPipeline is retained as a thin wrapper that delegates
# to CppBookRenderer for rendering and OrderBookMetaPipeline for analysis.
# engine_runner.py should prefer using OrderBookMetaPipeline directly.

class CppAcceleratedPipeline:
    """DEPRECATED — use OrderBookMetaPipeline for analysis.

    Retained for backward compatibility. Delegates to CppBookRenderer
    for rendering and OrderBookMetaPipeline for analysis.
    """

    def __init__(self):
        if not _cpp_available:
            raise RuntimeError("C++ core not available")

        self._renderer = CppBookRenderer()

        try:
            from .pipeline import OrderBookMetaPipeline
        except ImportError:
            from pipeline import OrderBookMetaPipeline
        self._pipeline = OrderBookMetaPipeline()

        try:
            from .level_tracker import LevelTracker
        except ImportError:
            from level_tracker import LevelTracker
        self._level_tracker = LevelTracker()

        logger.info("CppAcceleratedPipeline initialized (analysis via Python, rendering via C++)")

    @property
    def is_cpp(self) -> bool:
        return True

    def run(self, snapshot, combined_regime_output=None):
        """Process snapshot: Python analysis + C++ rendering."""
        # Run Python analysis pipeline (all 5 layers + aggregator)
        frame = self._pipeline.run(snapshot, combined_regime_output)

        # Run C++ book rendering for bid/ask/dom/tape/stats
        rendered = self._renderer.render(snapshot)

        # Run LevelTracker
        spoof_events = frame.authenticity.spoof_events if frame.authenticity else []
        auth_score = frame.authenticity.authenticity_score if frame.authenticity else 1.0
        level_states = self._level_tracker.update(
            snapshot, spoof_events=spoof_events, authenticity_score=auth_score
        )

        # Merge: analysis from Python frame + rendering from C++
        output = _CppFrame(
            timestamp_ms=snapshot.timestamp_ms or int(time.time() * 1000),
            symbol=getattr(snapshot, 'symbol', None) or 'UNKNOWN',
            bid_bars=rendered['bid_bars'],
            ask_bars=rendered['ask_bars'],
            dom_rows=rendered['dom_rows'],
            tape=rendered['tape'],
            stats=rendered['stats'],
            staircase={
                'imbalance_ratio': frame.staircase.imbalance_ratio if frame.staircase else 0.0,
                'bid_total_volume': frame.staircase.bid_total_volume if frame.staircase else 0.0,
                'ask_total_volume': frame.staircase.ask_total_volume if frame.staircase else 0.0,
            },
            game_state={
                'state': frame.game_state.state.value if frame.game_state else 'BALANCED',
                'streak_length': frame.game_state.streak_length if frame.game_state else 0,
                'streak_velocity': frame.game_state.streak_velocity if frame.game_state else 0.0,
                'pressure': frame.game_state.pressure if frame.game_state else 0.0,
                'stall_count': frame.game_state.stall_count if frame.game_state else 0,
            },
            force_vector={
                'total_force': frame.force_vector.total_force if frame.force_vector else 0.0,
                'institutional_score': frame.force_vector.institutional_score if frame.force_vector else 0.0,
            },
            authenticity={
                'authenticity_score': auth_score,
                'spoof_events': [],
            },
            regime_weights={
                'regime': 'UNKNOWN',
                'abstain': False,
                'l1_weight': 0.25, 'l2_weight': 0.25, 'l3_weight': 0.25, 'l4_weight': 0.25,
            },
            direction=frame.direction if hasattr(frame, 'direction') else 0.0,
            confidence=frame.confidence if hasattr(frame, 'confidence') else 0.0,
            urgency=frame.urgency if hasattr(frame, 'urgency') else 0.0,
            size_multiplier=frame.size_multiplier if hasattr(frame, 'size_multiplier') else 0.0,
        )
        output.level_states = [
            {
                'price': ls.price,
                'side': ls.side.value,
                'volume': ls.volume,
                'peak_volume': ls.peak_volume,
                'order_count': ls.order_count,
                'lifecycle': ls.lifecycle.value,
                'significance': ls.significance,
                'authenticity': ls.authenticity,
                'spoof_type': ls.spoof_type.value if ls.spoof_type else None,
                'iceberg_suspected': ls.iceberg_suspected,
                'fill_count': ls.fill_count,
                'refill_count': ls.refill_count,
            }
            for ls in level_states
        ]
        return output


class _CppFrame:
    """Lightweight frame object matching DepthIndicatorFrame interface for serialization."""
    __slots__ = [
        'timestamp_ms', 'symbol', 'bid_bars', 'ask_bars', 'dom_rows',
        'tape', 'stats', 'staircase', 'game_state', 'force_vector',
        'authenticity', 'regime_weights', 'direction', 'confidence',
        'urgency', 'size_multiplier', 'level_states',
        'ofi', 'vpin',
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
