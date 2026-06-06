"""Tests for p6lab.patterns.labeler — forward outcome labeling."""
from __future__ import annotations

from dataclasses import dataclass

import math
import pytest

from p6lab.patterns.labeler import (
    HORIZONS,
    HORIZON_MS,
    MultiHorizonOutcome,
    OutcomeClass,
    classify_outcome,
    compute_outcome_statistics,
    label_pattern_instance,
)


@dataclass
class _Ev:
    timestamp_ms: int
    price: float


class TestClassify:
    def test_long_continuation(self):
        assert classify_outcome(+1.0, "long") == OutcomeClass.CONTINUATION

    def test_long_reversal(self):
        assert classify_outcome(-1.0, "long") == OutcomeClass.REVERSAL

    def test_long_neutral(self):
        assert classify_outcome(+0.2, "long") == OutcomeClass.NEUTRAL

    def test_short_continuation_on_drop(self):
        assert classify_outcome(-1.0, "short") == OutcomeClass.CONTINUATION

    def test_threshold_exact_boundary_is_neutral(self):
        assert classify_outcome(+0.5, "long") == OutcomeClass.NEUTRAL

    def test_nan_is_neutral(self):
        assert classify_outcome(float("nan"), "long") == OutcomeClass.NEUTRAL


class TestLabelInstance:
    def _stream(self, t0: int = 0) -> list[_Ev]:
        # flat at 100.0 until t0+1m, then +2.0 move (~1.0 ATR)
        events = [_Ev(t0 + ms, 100.0) for ms in range(0, 30_000, 1_000)]
        events += [_Ev(t0 + 60_000, 102.0)]
        events += [_Ev(t0 + 300_000, 103.0)]
        events += [_Ev(t0 + 900_000, 104.0)]
        events += [_Ev(t0 + 3_600_000, 105.0)]
        return events

    def test_produces_all_horizons(self):
        events = self._stream(0)
        out = label_pattern_instance(
            events, pattern_timestamp_ms=0, pattern_direction="long",
            instrument_atr=2.0, tick_size=0.25,
        )
        assert set(out.outcomes.keys()) == set(HORIZONS)

    def test_continuation_classified_correctly(self):
        events = self._stream(0)
        out = label_pattern_instance(
            events, pattern_timestamp_ms=0, pattern_direction="long",
            instrument_atr=2.0, tick_size=0.25,
        )
        # 1m: +2 price → 1 ATR → continuation
        assert out.outcomes["1m"].classification == OutcomeClass.CONTINUATION
        assert out.outcomes["1m"].atr_normalized_return == pytest.approx(1.0, abs=1e-6)

    def test_incomplete_when_session_ends(self):
        events = self._stream(0)
        out = label_pattern_instance(
            events, pattern_timestamp_ms=0, pattern_direction="long",
            instrument_atr=2.0, tick_size=0.25,
            session_end_ms=120_000,  # 2 minute session
        )
        assert out.outcomes["1m"].classification == OutcomeClass.CONTINUATION
        assert out.outcomes["5m"].classification == OutcomeClass.INCOMPLETE
        assert out.outcomes["1h"].classification == OutcomeClass.INCOMPLETE

    def test_incomplete_when_stream_short(self):
        events = [_Ev(0, 100.0), _Ev(10_000, 100.5)]  # only 10s of data
        out = label_pattern_instance(
            events, pattern_timestamp_ms=0, pattern_direction="long",
            instrument_atr=2.0, tick_size=0.25,
        )
        # Can't see 1m+ forward → incomplete
        assert out.outcomes["1m"].classification == OutcomeClass.INCOMPLETE

    def test_raw_return_in_ticks(self):
        events = [_Ev(0, 100.0), _Ev(60_000, 101.0)]
        out = label_pattern_instance(
            events, pattern_timestamp_ms=0, pattern_direction="long",
            instrument_atr=2.0, tick_size=0.25,
        )
        # 1.0 price move / 0.25 tick = 4 ticks
        assert out.outcomes["1m"].raw_return_ticks == pytest.approx(4.0)


class TestStatistics:
    def _make(self, returns: list[float]) -> list[MultiHorizonOutcome]:
        from p6lab.patterns.labeler import PatternOutcome
        out = []
        for i, r in enumerate(returns):
            cls = OutcomeClass.CONTINUATION if r > 0.5 else (
                OutcomeClass.REVERSAL if r < -0.5 else OutcomeClass.NEUTRAL)
            po = PatternOutcome(
                horizon="5m", raw_return_ticks=r * 8, atr_normalized_return=r,
                classification=cls, pattern_timestamp_ms=i, outcome_timestamp_ms=i + 300_000,
            )
            out.append(MultiHorizonOutcome(
                pattern_timestamp_ms=i, symbol="NQ", pattern_id="c1",
                outcomes={"5m": po},
            ))
        return out

    def test_stats_basic(self):
        outs = self._make([1.0, 1.0, -0.5, 0.1])
        stats = compute_outcome_statistics(outs, "5m")
        assert stats["n"] == 4
        assert stats["hit_rate"] == pytest.approx(2 / 4)
        assert stats["mean_atr"] == pytest.approx(0.4)

    def test_stats_unknown_horizon_raises(self):
        with pytest.raises(ValueError):
            compute_outcome_statistics([], "2m")
