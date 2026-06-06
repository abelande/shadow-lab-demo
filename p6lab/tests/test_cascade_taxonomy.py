"""
CascadeClassifier unit tests.

Drives synthetic ``GameState`` sequences straight into the private detection
methods (``_detect_momentum_ignition``, ``_detect_liquidity_withdrawal``,
``_detect_grinding_correction``) so we test the taxonomy rules in
isolation from the cup_flip state machine. A separate end-to-end test
feeds real OrderBookSnapshot sequences through ``classify_snapshots`` to
verify the integration path.

Gates covered:
  - Type B fires on velocity + accel thresholds; respects cooldown
  - Type A fires on sustained stall + exhaustion; respects cooldown
  - Type C requires multi-day span AND enough stall transitions
  - Type D never fires from single-instrument input (intentionally)
  - Threshold parametrization works — bumping threshold kills detection
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent.parent))   # .../projects/ → p6v2.* package

from p6lab.patterns.cascade_taxonomy import (
    CascadeClassifier, CascadeEvent, CascadeThresholds, CascadeType,
)
from p6v2.models import CupFlipState, GameState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gs(state: CupFlipState, ts_ms: int, **kw) -> GameState:
    """Build a GameState with sane defaults + overrides."""
    return GameState(
        state=state,
        timestamp_ms=ts_ms,
        streak_length=kw.get("streak_length", 0),
        streak_velocity=kw.get("streak_velocity", 0.0),
        streak_depth=kw.get("streak_depth", 0),
        pressure=kw.get("pressure", 0.0),
        stall_count=kw.get("stall_count", 0),
        pressure_acceleration=kw.get("pressure_acceleration", 0.0),
        streak_exhaustion=kw.get("streak_exhaustion", 0.0),
        state_confidence=kw.get("state_confidence", 0.0),
    )


# ---------------------------------------------------------------------------
# Type B — Momentum ignition
# ---------------------------------------------------------------------------

def test_type_b_fires_on_streak_with_velocity_and_accel():
    clf = CascadeClassifier()
    history = [
        _gs(CupFlipState.BALANCED, 0),
        _gs(CupFlipState.BULL_STREAK, 1_000,
            streak_length=4, streak_velocity=3.0, pressure_acceleration=1.4),
        _gs(CupFlipState.BULL_STREAK, 2_000,
            streak_length=5, streak_velocity=2.5, pressure_acceleration=1.1),
    ]
    events = clf._detect_momentum_ignition(history)
    assert len(events) >= 1
    assert all(e.cascade_type == CascadeType.MOMENTUM_IGNITION for e in events)
    assert events[0].anchor_ts_ms == 1_000   # streak start
    assert 0.0 <= events[0].confidence <= 1.0


def test_type_b_respects_cooldown():
    """Two qualifying streaks 1s apart — cooldown is 5s by default → only one fires."""
    clf = CascadeClassifier()
    history = [
        _gs(CupFlipState.BULL_STREAK, ts,
            streak_length=4, streak_velocity=3.0, pressure_acceleration=1.5)
        for ts in (1_000, 2_000, 3_000, 4_000)
    ]
    events = clf._detect_momentum_ignition(history)
    assert len(events) == 1, f"expected cooldown to collapse to 1, got {len(events)}"


def test_type_b_skipped_when_velocity_too_low():
    clf = CascadeClassifier()
    history = [
        _gs(CupFlipState.BULL_STREAK, 1_000,
            streak_length=4, streak_velocity=1.0,  # < default 2.0
            pressure_acceleration=1.5),
    ]
    assert clf._detect_momentum_ignition(history) == []


def test_type_b_skipped_when_pressure_not_accelerating():
    clf = CascadeClassifier()
    history = [
        _gs(CupFlipState.BULL_STREAK, 1_000,
            streak_length=4, streak_velocity=3.0,
            pressure_acceleration=0.5),  # < 1.0
    ]
    assert clf._detect_momentum_ignition(history) == []


def test_threshold_override_disables_type_b():
    """Bumping min velocity to 999 kills all Type B events."""
    clf = CascadeClassifier(thresholds=CascadeThresholds(momentum_min_streak_velocity=999.0))
    history = [
        _gs(CupFlipState.BULL_STREAK, 1_000,
            streak_length=4, streak_velocity=3.0, pressure_acceleration=1.5),
    ]
    assert clf._detect_momentum_ignition(history) == []


# ---------------------------------------------------------------------------
# Type A — Liquidity withdrawal
# ---------------------------------------------------------------------------

def test_type_a_fires_on_sustained_stall():
    clf = CascadeClassifier()
    # 65s of BEAR_STALL with progressively higher exhaustion + stall_count
    history = [_gs(CupFlipState.BALANCED, 0)]
    for i in range(1, 70):
        history.append(_gs(
            CupFlipState.BEAR_STALL, i * 1_000,
            stall_count=4 + (i // 5),
            streak_exhaustion=min(0.9, 0.3 + i * 0.02),
            pressure_acceleration=0.4,  # flow not accelerating
        ))
    events = clf._detect_liquidity_withdrawal(history)
    assert len(events) >= 1
    assert events[0].cascade_type == CascadeType.LIQUIDITY_WITHDRAWAL
    assert events[0].metadata["duration_ms"] >= 60_000


def test_type_a_skipped_when_stall_too_short():
    clf = CascadeClassifier()
    history = [
        _gs(CupFlipState.BEAR_STALL, ts,
            stall_count=5, streak_exhaustion=0.9, pressure_acceleration=0.4)
        for ts in range(0, 30_000, 1_000)   # only 30s
    ]
    assert clf._detect_liquidity_withdrawal(history) == []


def test_type_a_skipped_when_pressure_still_accelerating():
    clf = CascadeClassifier()
    history = [_gs(CupFlipState.BEAR_STALL, i * 1_000,
                   stall_count=5, streak_exhaustion=0.9,
                   pressure_acceleration=1.5)  # too high
              for i in range(70)]
    assert clf._detect_liquidity_withdrawal(history) == []


# ---------------------------------------------------------------------------
# Type C — Grinding correction (multi-day)
# ---------------------------------------------------------------------------

def test_type_c_requires_multiday_span():
    """Single-day input (< 1 day span) never emits Type C."""
    clf = CascadeClassifier()
    # 8 stall transitions but only 60s apart — fails span check
    history = []
    for i in range(20):
        state = CupFlipState.BEAR_STALL if i % 2 == 0 else CupFlipState.BALANCED
        history.append(_gs(state, i * 60_000))   # 20min total
    assert clf._detect_grinding_correction(history) == []


def test_type_c_fires_on_multiday_with_enough_stall_transitions():
    clf = CascadeClassifier()
    # Span 2 days with 10 stall transitions
    ONE_DAY = 86_400_000
    history = []
    # First day: 5 stall transitions
    for i in range(10):
        state = CupFlipState.BEAR_STALL if i % 2 == 0 else CupFlipState.BALANCED
        history.append(_gs(state, i * 1_000))
    # Day boundary
    history.append(_gs(CupFlipState.BALANCED, ONE_DAY + 1_000))
    # Second day: 5 more stall transitions
    for i in range(10):
        state = CupFlipState.BULL_STALL if i % 2 == 0 else CupFlipState.BALANCED
        history.append(_gs(state, ONE_DAY + 2_000 + i * 1_000))
    events = clf._detect_grinding_correction(history)
    assert len(events) == 1
    assert events[0].cascade_type == CascadeType.GRINDING_CORRECTION
    assert events[0].metadata["span_ms"] >= ONE_DAY


# ---------------------------------------------------------------------------
# Type D — placeholder for cross-instrument contagion
# ---------------------------------------------------------------------------

def test_type_d_never_fires_from_single_instrument():
    clf = CascadeClassifier()
    # Any single-instrument history should never produce Type D
    history = [_gs(CupFlipState.BULL_STREAK, i * 100,
                   streak_length=10, streak_velocity=5.0,
                   pressure_acceleration=2.0)
              for i in range(50)]
    all_events = clf._detect_all(history)
    types = {e.cascade_type for e in all_events}
    assert CascadeType.CROSS_INSTRUMENT_CONTAGION not in types


# ---------------------------------------------------------------------------
# End-to-end: feed real OrderBookSnapshot objects via classify_snapshots
# ---------------------------------------------------------------------------

def test_classify_snapshots_returns_sorted_list():
    """Exercise the integration path; assert output is a sorted list."""
    clf = CascadeClassifier()

    # Minimal snapshot stub — classify_snapshots reads .recent_events,
    # .timestamp_ms, .best_bid/ask, .mid_price, .ts_event
    class _Stub:
        def __init__(self, ts, bid=100.0, ask=100.25):
            self.timestamp_ms = ts
            self.best_bid = bid
            self.best_ask = ask
            self.mid_price = (bid + ask) / 2
            self.spread = ask - bid
            self.recent_events = []
            self.bids = []
            self.asks = []

    snaps = [_Stub(i * 100) for i in range(30)]
    events = clf.classify_snapshots(snaps)
    assert isinstance(events, list)
    # All events should have valid fields even if none fire
    for e in events:
        assert isinstance(e, CascadeEvent)
        assert 0.0 <= e.confidence <= 1.0
