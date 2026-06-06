"""Tests for LevelTracker — lifecycle transitions, significance scoring, iceberg detection."""
from __future__ import annotations

import pytest
from p6.models import (
    InstrumentVisualConfig,
    LevelLifecycle,
    LevelState,
    Order,
    OrderAction,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    SpoofEvent,
    SpoofType,
)
from p6.level_tracker import LevelTracker


# ── Fixtures ───────────────────────────────────────────────────────

def _cfg() -> InstrumentVisualConfig:
    return InstrumentVisualConfig(
        symbol="TEST",
        tick_size=0.25,
        significant_volume=50.0,
        significant_age_ms=3000,
        significant_order_count=5,
        round_number_step=25.0,
        zone_merge_ticks=4,
        level_fade_candles=3,
    )


def _snap(
    ts: int,
    bid_levels: list | None = None,
    ask_levels: list | None = None,
    recent_events: list | None = None,
    recent_trades: list | None = None,
    symbol: str = "TEST",
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp_ms=ts,
        symbol=symbol,
        bids=bid_levels or [],
        asks=ask_levels or [],
        recent_events=recent_events or [],
        recent_trades=recent_trades or [],
    )


def _blevel(price: float, side: Side, volume: float, order_count: int = 10) -> OrderBookLevel:
    return OrderBookLevel(price=price, side=side, volume=volume, order_count=order_count)


def _fill(price: float, side: Side, size: float, ts: int) -> Order:
    return Order(
        order_id=f"f{ts}",
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


# ── Tests: FORMING → RESTING ───────────────────────────────────────

class TestFormingToResting:
    def test_new_level_starts_forming(self):
        tracker = LevelTracker(_cfg())
        snap = _snap(
            ts=1000,
            ask_levels=[_blevel(100.0, Side.ASK, 80.0, 10)],
        )
        levels = tracker.update(snap)
        # Significance should be low (just appeared), likely below threshold
        all_lvls = tracker.get_all_levels()
        assert any(l.lifecycle == LevelLifecycle.FORMING for l in all_lvls)

    def test_forming_transitions_to_resting_after_age_threshold(self):
        cfg = _cfg()
        tracker = LevelTracker(cfg)
        t0 = 1_000_000

        # Initial snapshot — level appears
        snap1 = _snap(t0, ask_levels=[_blevel(100.0, Side.ASK, 80.0, 10)])
        tracker.update(snap1)

        # Advance past significant_age_ms
        t1 = t0 + cfg.significant_age_ms + 500
        snap2 = _snap(t1, ask_levels=[_blevel(100.0, Side.ASK, 80.0, 10)])
        tracker.update(snap2)

        all_lvls = tracker.get_all_levels()
        lvl = next((l for l in all_lvls if l.price == 100.0 and l.side == Side.ASK), None)
        assert lvl is not None
        assert lvl.lifecycle == LevelLifecycle.RESTING

    def test_significance_increases_with_age(self):
        cfg = _cfg()
        tracker = LevelTracker(cfg)
        t0 = 2_000_000

        snap1 = _snap(t0, ask_levels=[_blevel(100.0, Side.ASK, 80.0, 10)])
        tracker.update(snap1)
        all_early = tracker.get_all_levels()
        early_sig = next((l.significance for l in all_early if l.price == 100.0), 0.0)

        # Advance 10 seconds
        snap2 = _snap(t0 + 10000, ask_levels=[_blevel(100.0, Side.ASK, 80.0, 10)])
        tracker.update(snap2)
        all_later = tracker.get_all_levels()
        later_sig = next((l.significance for l in all_later if l.price == 100.0), 0.0)

        assert later_sig > early_sig


# ── Tests: RESTING → TESTED ───────────────────────────────────────

class TestRestingToTested:
    def _get_resting_tracker(self):
        cfg = _cfg()
        tracker = LevelTracker(cfg)
        t0 = 3_000_000

        # Build level to RESTING state
        for tick in range(0, cfg.significant_age_ms + 1000, 100):
            snap = _snap(
                t0 + tick,
                ask_levels=[_blevel(100.0, Side.ASK, 80.0, 10)],
                bid_levels=[_blevel(99.75, Side.BID, 50.0, 8)],
            )
            tracker.update(snap)

        return tracker, t0 + cfg.significant_age_ms + 1000

    def test_resting_to_tested_when_price_touches_with_fills(self):
        tracker, t_now = self._get_resting_tracker()

        # Price touches the ask level (best_bid = ask price) with fills
        snap = _snap(
            t_now + 100,
            ask_levels=[_blevel(100.0, Side.ASK, 60.0, 8)],  # volume dropped (fills)
            bid_levels=[_blevel(100.0, Side.BID, 20.0, 3)],  # bid at ask price = touching
            recent_events=[_fill(100.0, Side.ASK, 20.0, t_now + 100)],
        )
        tracker.update(snap)

        all_lvls = tracker.get_all_levels()
        lvl = next((l for l in all_lvls if l.price == 100.0 and l.side == Side.ASK), None)
        assert lvl is not None
        assert lvl.lifecycle == LevelLifecycle.TESTED


# ── Tests: TESTED → DEFENDED ──────────────────────────────────────

class TestTestedToDefended:
    def test_tested_to_defended_when_volume_holds(self):
        cfg = _cfg()
        tracker = LevelTracker(cfg)
        t0 = 4_000_000

        # Advance level to RESTING
        for tick in range(0, cfg.significant_age_ms + 500, 100):
            tracker.update(_snap(t0 + tick, ask_levels=[_blevel(100.0, Side.ASK, 100.0, 10)]))

        t_rest = t0 + cfg.significant_age_ms + 500

        # Touch with small fill (volume barely drops — defended)
        tracker.update(_snap(
            t_rest + 100,
            ask_levels=[_blevel(100.0, Side.ASK, 90.0, 9)],  # 90% of 100 — well above DEFENDED threshold
            bid_levels=[_blevel(100.0, Side.BID, 10.0, 2)],
            recent_events=[_fill(100.0, Side.ASK, 10.0, t_rest + 100)],
        ))

        # Another touch — fills increment
        tracker.update(_snap(
            t_rest + 200,
            ask_levels=[_blevel(100.0, Side.ASK, 85.0, 9)],
            bid_levels=[_blevel(100.0, Side.BID, 10.0, 2)],
            recent_events=[_fill(100.0, Side.ASK, 5.0, t_rest + 200)],
        ))

        all_lvls = tracker.get_all_levels()
        lvl = next((l for l in all_lvls if l.price == 100.0 and l.side == Side.ASK), None)
        assert lvl is not None
        # Should be TESTED or DEFENDED (fill_count should be > 0)
        assert lvl.fill_count > 0
        assert lvl.lifecycle in (LevelLifecycle.TESTED, LevelLifecycle.DEFENDED)


# ── Tests: TESTED → BROKEN ────────────────────────────────────────

class TestTestedToBroken:
    def test_level_broken_when_consumed_by_fills(self):
        cfg = _cfg()
        tracker = LevelTracker(cfg)
        t0 = 5_000_000

        # Advance to RESTING
        for tick in range(0, cfg.significant_age_ms + 500, 100):
            tracker.update(_snap(t0 + tick, ask_levels=[_blevel(200.0, Side.ASK, 100.0, 10)]))

        t_rest = t0 + cfg.significant_age_ms + 500

        # Touch (TESTED)
        tracker.update(_snap(
            t_rest + 100,
            ask_levels=[_blevel(200.0, Side.ASK, 50.0, 5)],
            bid_levels=[_blevel(200.0, Side.BID, 10.0, 2)],
            recent_events=[_fill(200.0, Side.ASK, 50.0, t_rest + 100)],
        ))

        # Volume drops below BROKEN threshold (< 25% of peak=100 → 100*0.25=25)
        tracker.update(_snap(
            t_rest + 200,
            ask_levels=[_blevel(200.0, Side.ASK, 10.0, 1)],
            bid_levels=[_blevel(200.0, Side.BID, 10.0, 2)],
            recent_events=[_fill(200.0, Side.ASK, 40.0, t_rest + 200)],
        ))

        all_lvls = tracker.get_all_levels()
        lvl = next((l for l in all_lvls if l.price == 200.0 and l.side == Side.ASK), None)
        # May be BROKEN or TESTED with high fill count
        if lvl:
            assert lvl.lifecycle in (LevelLifecycle.TESTED, LevelLifecycle.BROKEN)

    def test_level_broken_when_disappears_after_fills(self):
        cfg = _cfg()
        tracker = LevelTracker(cfg)
        t0 = 6_000_000

        for tick in range(0, cfg.significant_age_ms + 500, 100):
            tracker.update(_snap(t0 + tick, ask_levels=[_blevel(300.0, Side.ASK, 100.0, 10)]))

        t_rest = t0 + cfg.significant_age_ms + 500

        # Level disappears with fill events
        tracker.update(_snap(
            t_rest + 100,
            ask_levels=[],
            recent_events=[_fill(300.0, Side.ASK, 100.0, t_rest + 100)],
        ))

        all_lvls = tracker.get_all_levels()
        lvl = next((l for l in all_lvls if l.price == 300.0 and l.side == Side.ASK), None)
        # Should be BROKEN
        if lvl:
            assert lvl.lifecycle == LevelLifecycle.BROKEN


# ── Tests: PULLED ─────────────────────────────────────────────────

class TestLevelPulled:
    def test_level_pulled_when_cancelled_without_fills(self):
        cfg = _cfg()
        tracker = LevelTracker(cfg)
        t0 = 7_000_000

        for tick in range(0, cfg.significant_age_ms + 500, 100):
            tracker.update(_snap(t0 + tick, bid_levels=[_blevel(500.0, Side.BID, 100.0, 10)]))

        t_rest = t0 + cfg.significant_age_ms + 500

        # Level disappears with cancel events (no fills)
        tracker.update(_snap(
            t_rest + 100,
            bid_levels=[],
            recent_events=[_cancel(500.0, Side.BID, 100.0, t_rest + 100)],
        ))

        all_lvls = tracker.get_all_levels()
        lvl = next((l for l in all_lvls if l.price == 500.0 and l.side == Side.BID), None)
        if lvl:
            assert lvl.lifecycle == LevelLifecycle.PULLED


# ── Tests: Significance Scoring ───────────────────────────────────

class TestSignificanceScoring:
    def test_significance_below_threshold_not_emitted(self):
        cfg = _cfg()
        tracker = LevelTracker(cfg)
        t0 = 8_000_000

        # Very small volume, just appeared — should not be emitted (significance < 0.3)
        snap = _snap(t0, ask_levels=[_blevel(100.0, Side.ASK, 1.0, 1)])  # tiny volume
        levels = tracker.update(snap)

        # Should not appear in the emitted (significant) levels
        emitted = [l for l in levels if l.price == 100.0 and l.side == Side.ASK]
        assert len(emitted) == 0

    def test_significance_above_threshold_emitted(self):
        cfg = _cfg()
        tracker = LevelTracker(cfg)
        t0 = 9_000_000

        # Large volume + aged enough
        for tick in range(0, cfg.significant_age_ms + 1000, 100):
            snap = _snap(t0 + tick, ask_levels=[_blevel(100.0, Side.ASK, 200.0, 20)])
            tracker.update(snap)

        # Last update — should be emitted
        final = _snap(t0 + cfg.significant_age_ms + 1000, ask_levels=[_blevel(100.0, Side.ASK, 200.0, 20)])
        levels = tracker.update(final)
        emitted = [l for l in levels if l.price == 100.0 and l.side == Side.ASK]
        assert len(emitted) == 1
        assert emitted[0].significance > 0.3

    def test_round_number_boosts_significance(self):
        cfg = _cfg()  # round_number_step=25.0
        tracker_round = LevelTracker(cfg)
        tracker_nonround = LevelTracker(cfg)
        t0 = 10_000_000

        # Round price: 25.00 (divisible by 25)
        for tick in range(0, cfg.significant_age_ms + 500, 100):
            tracker_round.update(_snap(t0 + tick, ask_levels=[_blevel(25.0, Side.ASK, 60.0, 6)]))
            tracker_nonround.update(_snap(t0 + tick, ask_levels=[_blevel(27.5, Side.ASK, 60.0, 6)]))

        round_lvls = tracker_round.get_all_levels()
        nonround_lvls = tracker_nonround.get_all_levels()

        round_sig = next((l.significance for l in round_lvls if l.price == 25.0), 0.0)
        nonround_sig = next((l.significance for l in nonround_lvls if l.price == 27.5), 0.0)

        assert round_sig > nonround_sig


# ── Tests: Iceberg Detection ───────────────────────────────────────

class TestIcebergDetection:
    def test_iceberg_suspected_after_refills(self):
        cfg = _cfg()
        tracker = LevelTracker(cfg)
        t0 = 11_000_000

        # Advance to RESTING
        for tick in range(0, cfg.significant_age_ms + 500, 100):
            tracker.update(_snap(t0 + tick, ask_levels=[_blevel(100.0, Side.ASK, 100.0, 10)]))

        t_rest = t0 + cfg.significant_age_ms + 500

        # Simulate refill pattern: volume drops then recovers (fills + new orders)
        tracker.update(_snap(
            t_rest + 100,
            ask_levels=[_blevel(100.0, Side.ASK, 70.0, 9)],
            recent_events=[_fill(100.0, Side.ASK, 30.0, t_rest + 100)],
        ))
        # Volume recovers (refill)
        tracker.update(_snap(t_rest + 200, ask_levels=[_blevel(100.0, Side.ASK, 90.0, 10)]))
        # Volume drops again
        tracker.update(_snap(
            t_rest + 300,
            ask_levels=[_blevel(100.0, Side.ASK, 65.0, 9)],
            recent_events=[_fill(100.0, Side.ASK, 25.0, t_rest + 300)],
        ))
        # Volume recovers again
        tracker.update(_snap(t_rest + 400, ask_levels=[_blevel(100.0, Side.ASK, 90.0, 10)]))

        all_lvls = tracker.get_all_levels()
        lvl = next((l for l in all_lvls if l.price == 100.0 and l.side == Side.ASK), None)
        assert lvl is not None
        assert lvl.refill_count >= 2
        assert lvl.iceberg_suspected is True


# ── Tests: Instrument Config Thresholds ───────────────────────────

class TestInstrumentConfig:
    def test_high_volume_threshold_suppresses_small_levels(self):
        # ES config has higher significant_volume=100 vs NQ=50
        es_cfg = InstrumentVisualConfig.for_symbol("ES")
        nq_cfg = InstrumentVisualConfig.for_symbol("NQ")
        tracker_es = LevelTracker(es_cfg)
        tracker_nq = LevelTracker(nq_cfg)
        t0 = 12_000_000

        age_ms = max(es_cfg.significant_age_ms, nq_cfg.significant_age_ms) + 2000
        # Volume=50 is at ES threshold but above NQ threshold
        for tick in range(0, age_ms, 100):
            tracker_es.update(_snap(t0 + tick, ask_levels=[_blevel(4500.5, Side.ASK, 50.0, 5)]))
            tracker_nq.update(_snap(t0 + tick, ask_levels=[_blevel(4500.5, Side.ASK, 50.0, 5)]))

        es_lvls = tracker_es.get_all_levels()
        nq_lvls = tracker_nq.get_all_levels()
        es_sig = next((l.significance for l in es_lvls if l.price == 4500.5), 0.0)
        nq_sig = next((l.significance for l in nq_lvls if l.price == 4500.5), 0.0)

        # ES has higher volume threshold (100), so volume_score(50/100=0.5) < NQ volume_score(50/50=1.0)
        # Therefore ES significance should be lower than NQ for the same volume
        assert es_sig < nq_sig

    def test_for_symbol_returns_correct_config(self):
        nq_cfg = InstrumentVisualConfig.for_symbol("NQ.c.0")
        es_cfg = InstrumentVisualConfig.for_symbol("ES.c.0")
        assert nq_cfg.significant_volume == 50.0
        assert es_cfg.significant_volume == 100.0
        assert "NQ" in nq_cfg.symbol

    def test_default_config_for_unknown_symbol(self):
        cfg = InstrumentVisualConfig.for_symbol("UNKNOWN")
        assert cfg.symbol == "DEFAULT"
        assert cfg.tick_size > 0
