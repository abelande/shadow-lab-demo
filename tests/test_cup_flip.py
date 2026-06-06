"""Tests for TapeReader / Cup Flip (Layer 2)."""
from __future__ import annotations
import pytest
from p6.cup_flip.tape_reader import TapeReader
from p6.models import Order, OrderAction, Side, CupFlipState, GameState


def _fill(oid, side, price, size, ts):
    return Order(
        order_id=oid, side=side, price=price, size=size,
        timestamp_ms=ts, action=OrderAction.FILL, is_aggressive=True,
    )


def _add(oid, side, price, size, ts):
    return Order(
        order_id=oid, side=side, price=price, size=size,
        timestamp_ms=ts, action=OrderAction.ADD,
    )


def test_tape_reader_returns_game_state():
    reader = TapeReader()
    state = reader.update([], timestamp_ms=1000)
    assert isinstance(state, GameState)


def test_tape_reader_empty_events_balanced():
    reader = TapeReader()
    state = reader.update([], timestamp_ms=1000)
    assert state.state == CupFlipState.BALANCED


def test_tape_reader_consecutive_ask_fills_bull_streak():
    reader = TapeReader()
    t0 = 1_000_000
    fills = [
        _fill("t1", Side.ASK, 100.5, 10.0, t0 + 100),
        _fill("t2", Side.ASK, 101.0, 15.0, t0 + 200),
        _fill("t3", Side.ASK, 101.5, 20.0, t0 + 300),
        _fill("t4", Side.ASK, 102.0, 10.0, t0 + 400),
    ]
    state = reader.update(fills, timestamp_ms=t0 + 500)
    assert state.state in (CupFlipState.BULL_STREAK, CupFlipState.STOP_RUN)
    assert state.streak_length >= 3


def test_tape_reader_pressure_positive_on_buy_pressure():
    reader = TapeReader()
    t0 = 1_000_000
    buys = [_fill(f"t{i}", Side.ASK, 100.0 + i * 0.5, 20.0, t0 + i * 100) for i in range(4)]
    state = reader.update(buys, timestamp_ms=t0 + 500)
    assert state.pressure > 0.0


def test_tape_reader_gap_tolerance_absorbs_single_opposing_fill():
    reader = TapeReader()
    # With gap_tolerance=1 (default), a single opposing fill is absorbed.
    # ASK-fill, BID-fill (absorbed), ASK-fill → streak of 3 on ASK side.
    t0 = 1_000_000
    events = [
        _fill("t1", Side.ASK, 100.5, 5.0, t0 + 100),
        _fill("t2", Side.BID, 100.0, 5.0, t0 + 200),
        _fill("t3", Side.ASK, 100.5, 5.0, t0 + 300),
    ]
    state = reader.update(events, timestamp_ms=t0 + 400)
    # Streak continues through 1 opposing fill (gap_tolerance=1)
    assert state.streak_length == 3

    # Two consecutive opposing fills must break the streak
    reader2 = TapeReader()
    events2 = [
        _fill("t1", Side.ASK, 100.5, 5.0, t0 + 100),
        _fill("t2", Side.BID, 100.0, 5.0, t0 + 200),
        _fill("t3", Side.BID, 100.0, 5.0, t0 + 300),
    ]
    state2 = reader2.update(events2, timestamp_ms=t0 + 400)
    # After 2 opposing fills the ASK streak resets; BID streak just started (<3)
    assert state2.streak_length < 3


def test_tape_reader_stateful_across_calls():
    reader = TapeReader()
    t0 = 1_000_000
    # First call with 2 fills (not yet streak)
    reader.update(
        [_fill("t1", Side.ASK, 100.5, 10.0, t0 + 100),
         _fill("t2", Side.ASK, 101.0, 10.0, t0 + 200)],
        timestamp_ms=t0 + 200,
    )
    # Second call with more fills to extend streak
    state = reader.update(
        [_fill("t3", Side.ASK, 101.5, 10.0, t0 + 300),
         _fill("t4", Side.ASK, 102.0, 10.0, t0 + 400)],
        timestamp_ms=t0 + 500,
    )
    assert state.streak_length >= 3
