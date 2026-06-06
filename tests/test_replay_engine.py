"""Tests for ReplayEngine — candle aggregation, level state computation, seek/step."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from p6.models import (
    ForceVector,
    GameState,
    CupFlipState,
    InstrumentVisualConfig,
    LevelLifecycle,
    LevelState,
    Order,
    OrderAction,
    OrderBookLevel,
    OrderBookSnapshot,
    ReplayCandle,
    ReplayFrame,
    Side,
    TapeSummary,
)
from p6.server.replay_engine import (
    ReplayEngine,
    SUPPORTED_TIMEFRAMES,
    _build_candles,
    _accumulate_tape,
    _empty_tape_summary,
)
from p6.level_tracker import LevelTracker


# ── Helpers ────────────────────────────────────────────────────────

def _trade(price: float, size: float, side: Side, ts: int) -> Order:
    return Order(
        order_id=f"t{ts}",
        side=side,
        price=price,
        size=size,
        timestamp_ms=ts,
        action=OrderAction.FILL,
        is_aggressive=True,
    )


def _cancel(price: float, side: Side, size: float, ts: int) -> Order:
    return Order(
        order_id=f"c{ts}",
        side=side,
        price=price,
        size=size,
        timestamp_ms=ts,
        action=OrderAction.CANCEL,
    )


def _make_records(trades_per_interval: list[dict], tf_ms: int = 1000) -> list[dict]:
    """Build snapshot records for testing, each record = one snapshot interval."""
    records = []
    t0 = 1_700_000_000_000
    for i, spec in enumerate(trades_per_interval):
        ts = t0 + i * tf_ms
        trd = [_trade(spec["price"], spec["size"], Side.ASK, ts) for _ in range(spec.get("count", 1))]
        records.append({
            "timestamp_ms": ts,
            "trades": trd,
            "events": spec.get("events", []),
            "best_bid": spec["price"] - 0.25,
            "best_ask": spec["price"],
            "level_states": spec.get("level_states", []),
            "game_state": spec.get("game_state"),
            "force_vector": spec.get("force_vector"),
            "spoof_events": spec.get("spoof_events", []),
        })
    return records


# ── Tests: Candle Aggregation ──────────────────────────────────────

class TestCandleAggregation:
    def test_1s_timeframe_one_candle_per_second(self):
        # 5 records at 200ms intervals → all fall in the same 1s bucket
        records = _make_records([
            {"price": 100.0, "size": 10.0},
            {"price": 101.0, "size": 5.0},
            {"price": 99.5, "size": 8.0},
            {"price": 100.5, "size": 3.0},
            {"price": 100.25, "size": 6.0},
        ], tf_ms=200)
        candles = _build_candles(records, timeframe_s=1)
        # All within 1 second → should produce 1 candle
        assert len(candles) == 1
        c = candles[0]
        assert c.open == 100.0
        assert c.high == 101.0
        assert c.low == 99.5
        assert c.close == 100.25
        assert abs(c.volume - 32.0) < 1e-6  # 10+5+8+3+6

    def test_15s_timeframe_aggregates_multiple_1s_buckets(self):
        # Use a t0 aligned to a 15s boundary so buckets split cleanly
        # Find the next 15s boundary after a large epoch
        base_s = (1_700_000_000 // 15 + 1) * 15  # e.g. 1_700_000_010
        t0 = base_s * 1000  # convert back to ms

        records = []
        prices_first = [100.0, 101.0, 99.5]
        prices_second = [103.0, 104.0, 102.0]

        for i, p in enumerate(prices_first):
            ts = t0 + i * 5000  # 0s, 5s, 10s — all in first 15s bucket [base_s, base_s+15)
            records.append({
                "timestamp_ms": ts,
                "trades": [_trade(p, 10.0, Side.ASK, ts)],
                "events": [],
                "best_bid": p - 0.25,
                "best_ask": p,
                "level_states": [],
                "game_state": None,
                "force_vector": None,
                "spoof_events": [],
            })
        for i, p in enumerate(prices_second):
            ts = t0 + 15000 + i * 5000  # 15s, 20s, 25s — second 15s bucket [base_s+15, base_s+30)
            records.append({
                "timestamp_ms": ts,
                "trades": [_trade(p, 10.0, Side.ASK, ts)],
                "events": [],
                "best_bid": p - 0.25,
                "best_ask": p,
                "level_states": [],
                "game_state": None,
                "force_vector": None,
                "spoof_events": [],
            })

        candles = _build_candles(records, timeframe_s=15)
        assert len(candles) == 2

        c0 = candles[0]
        assert c0.open == 100.0
        assert c0.high == 101.0
        assert c0.low == 99.5
        assert c0.close == 99.5

        c1 = candles[1]
        assert c1.open == 103.0
        assert c1.high == 104.0
        assert c1.low == 102.0

    def test_empty_records_produces_no_candles(self):
        candles = _build_candles([], timeframe_s=15)
        assert candles == []

    def test_volume_accumulates_across_records(self):
        records = _make_records([
            {"price": 100.0, "size": 10.0, "count": 1},
            {"price": 100.25, "size": 5.0, "count": 1},
            {"price": 100.50, "size": 20.0, "count": 1},
        ], tf_ms=200)
        candles = _build_candles(records, timeframe_s=1)
        assert len(candles) == 1
        assert abs(candles[0].volume - 35.0) < 1e-6

    def test_supported_timeframes_all_produce_candles(self):
        records = _make_records([
            {"price": 100.0 + i, "size": 1.0}
            for i in range(300)
        ], tf_ms=1000)
        for tf in SUPPORTED_TIMEFRAMES:
            candles = _build_candles(records, timeframe_s=tf)
            assert len(candles) > 0


# ── Tests: Tape Summary Computation ───────────────────────────────

class TestTapeSummaryComputation:
    def test_buy_sell_volume_correctly_attributed(self):
        # ASK fill = buy aggressor
        summary = _empty_tape_summary()
        trades = [
            _trade(100.0, 50.0, Side.ASK, 1000),  # buy
            _trade(100.0, 30.0, Side.BID, 1001),  # sell
        ]
        result = _accumulate_tape(summary, trades, [])
        assert result.buy_volume == 50.0
        assert result.sell_volume == 30.0
        assert result.delta == 20.0

    def test_largest_fill_tracked(self):
        summary = _empty_tape_summary()
        trades = [
            _trade(100.0, 10.0, Side.ASK, 1000),
            _trade(101.0, 85.0, Side.ASK, 1001),
            _trade(99.5, 5.0, Side.BID, 1002),
        ]
        result = _accumulate_tape(summary, trades, [])
        assert result.largest_fill_size == 85.0
        assert result.largest_fill_price == 101.0

    def test_cancel_counts_tracked(self):
        summary = _empty_tape_summary()
        events = [
            _cancel(100.0, Side.BID, 50.0, 1000),
            _cancel(100.0, Side.BID, 30.0, 1001),
            _cancel(101.0, Side.ASK, 20.0, 1002),
        ]
        result = _accumulate_tape(summary, [], events)
        assert result.cancel_count_bid == 2
        assert result.cancel_count_ask == 1

    def test_accumulate_is_additive(self):
        s1 = _empty_tape_summary()
        s1 = _accumulate_tape(s1, [_trade(100.0, 10.0, Side.ASK, 1000)], [])
        s2 = _accumulate_tape(s1, [_trade(100.0, 20.0, Side.ASK, 1001)], [])
        assert s2.buy_volume == 30.0


# ── Tests: Seek / Step ─────────────────────────────────────────────

class TestReplayEngineSeekStep:
    def _make_engine_with_candles(self, n_candles: int = 10, tf_s: int = 15) -> ReplayEngine:
        """Build a ReplayEngine with synthetic snapshot records and pre-built candles."""
        t0 = 1_700_000_000_000
        engine = ReplayEngine()
        engine._loaded = True
        engine._symbol = "TEST"
        engine._current_tf_s = tf_s

        # Build synthetic snapshot records
        records = []
        for i in range(n_candles * 10):  # 10 snapshots per candle
            ts = t0 + i * (tf_s * 100)  # 100ms intervals
            records.append({
                "timestamp_ms": ts,
                "trades": [_trade(100.0 + i * 0.01, 1.0, Side.ASK, ts)],
                "events": [],
                "best_bid": 99.75 + i * 0.01,
                "best_ask": 100.0 + i * 0.01,
                "level_states": [],
                "game_state": None,
                "force_vector": None,
                "spoof_events": [],
            })
        engine._snapshot_records = records

        # Pre-build candles
        engine._candle_cache[tf_s] = _build_candles(records, timeframe_s=tf_s)
        return engine

    def test_seek_to_valid_index(self):
        engine = self._make_engine_with_candles(10)
        engine.seek(5)
        assert engine._current_candle_idx == 5

    def test_seek_clamps_to_valid_range(self):
        engine = self._make_engine_with_candles(10)
        engine.seek(9999)
        candles = engine.get_candles(engine._current_tf_s)
        assert engine._current_candle_idx == len(candles) - 1

    def test_seek_below_zero_clamps_to_zero(self):
        engine = self._make_engine_with_candles(10)
        engine.seek(-5)
        assert engine._current_candle_idx == 0

    def test_step_forward_advances_index(self):
        engine = self._make_engine_with_candles(10)
        engine.seek(0)
        engine.step_forward()
        assert engine._current_candle_idx == 1

    def test_step_backward_decrements_index(self):
        engine = self._make_engine_with_candles(10)
        engine.seek(5)
        engine.step_backward()
        assert engine._current_candle_idx == 4

    def test_step_backward_at_zero_stays_zero(self):
        engine = self._make_engine_with_candles(10)
        engine.seek(0)
        engine.step_backward()
        assert engine._current_candle_idx == 0

    def test_step_forward_at_end_does_not_exceed(self):
        engine = self._make_engine_with_candles(5)
        candles = engine.get_candles(engine._current_tf_s)
        engine.seek(len(candles) - 1)
        engine.step_forward()
        assert engine._current_candle_idx == len(candles) - 1

    def test_get_frame_at_returns_replay_frame(self):
        engine = self._make_engine_with_candles(5)
        frame = engine.get_frame_at(0, engine._current_tf_s)
        assert frame is not None
        assert isinstance(frame, ReplayFrame)
        assert isinstance(frame.candle, ReplayCandle)

    def test_seek_by_timestamp(self):
        engine = self._make_engine_with_candles(10, tf_s=15)
        candles = engine.get_candles(15)
        if len(candles) >= 3:
            target_ts = candles[2].time * 1000
            engine.seek_by_timestamp(target_ts)
            assert engine._current_candle_idx == 2


# ── Tests: Timeframe Switching ─────────────────────────────────────

class TestTimeframeSwitching:
    def _engine(self) -> ReplayEngine:
        t0 = 1_700_000_000_000
        engine = ReplayEngine()
        engine._loaded = True
        engine._symbol = "TEST"

        records = []
        for i in range(600):  # 600 snapshots = 60 seconds at 100ms intervals
            ts = t0 + i * 100
            records.append({
                "timestamp_ms": ts,
                "trades": [_trade(100.0 + i * 0.01, 1.0, Side.ASK, ts)],
                "events": [],
                "best_bid": 99.75,
                "best_ask": 100.0,
                "level_states": [],
                "game_state": None,
                "force_vector": None,
                "spoof_events": [],
            })
        engine._snapshot_records = records
        return engine

    def test_timeframe_switch_changes_current_tf(self):
        engine = self._engine()
        engine._current_tf_s = 1
        engine._candle_cache[1] = _build_candles(engine._snapshot_records, 1)
        engine.set_timeframe(5)
        assert engine._current_tf_s == 5

    def test_timeframe_switch_preserves_approximate_position(self):
        engine = self._engine()
        # Build both timeframes
        engine._candle_cache[1] = _build_candles(engine._snapshot_records, 1)
        engine._candle_cache[5] = _build_candles(engine._snapshot_records, 5)
        engine._current_tf_s = 1

        candles_1s = engine.get_candles(1)
        # Seek to middle of 1s candles
        mid_idx = len(candles_1s) // 2
        engine.seek(mid_idx)
        mid_time = candles_1s[mid_idx].time if mid_idx < len(candles_1s) else 0

        engine.set_timeframe(5)
        candles_5s = engine.get_candles(5)
        # Position should be close to the same time in 5s candles
        current_time = candles_5s[engine._current_candle_idx].time if candles_5s else 0
        # 5s candles are coarser, so current_time should be within one candle of mid_time
        if candles_5s:
            assert abs(current_time - mid_time) <= 5  # within 5 seconds

    def test_1s_produces_more_candles_than_5m(self):
        engine = self._engine()
        c1 = _build_candles(engine._snapshot_records, 1)
        c300 = _build_candles(engine._snapshot_records, 300)
        assert len(c1) >= len(c300)

    def test_get_status_reflects_current_state(self):
        engine = self._engine()
        engine._candle_cache[15] = _build_candles(engine._snapshot_records, 15)
        engine._current_tf_s = 15
        status = engine.get_status()
        assert status["timeframe_s"] == 15
        assert status["loaded"] is True
        assert "total_candles" in status
        assert "current_candle_idx" in status
        assert "is_playing" in status
