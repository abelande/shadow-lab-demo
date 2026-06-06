"""Wave 5 Phase 5A — thesis-chain wire integration tests.

Verifies:
  1. Default pipeline (engine=None) still works; frame.correlation_matches = [].
  2. Pipeline with a stub engine + accumulator calls engine.match() at cadence
     and attaches serialized PatternMatch dicts to the frame.
  3. Throttling — matches only fire every ``match_interval_ms`` not every snap.
  4. Engine failures never propagate; frame remains usable.
  5. engine_runner._build_correlation_components loads live artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from p6.pipeline import OrderBookMetaPipeline


class _StubEngine:
    """In-memory CorrelationEngine double: counts match() calls and returns
    one PatternMatch-shaped object per call so the pipeline can serialize it."""
    def __init__(self) -> None:
        self.calls: int = 0

    def match(self, *, l2_window, l1_window, context):
        self.calls += 1
        return [_StubMatch(self.calls)]


@dataclass
class _StubMatch:
    """Mimics p6lab.correlation.engine.PatternMatch for asdict()."""
    pattern_id: str = "stub_pid"
    ensemble_score: float = 0.75
    confidence_tier: str = "B"
    expected_direction: str = "bull"
    expected_move_atr: float = 0.8
    template_similarity: float = 0.6
    mahalanobis_score: float = 0.5
    contextual_score: float = 0.4
    match_window_start_ms: int = 0
    match_window_end_ms: int = 0
    regime: str = "normal"
    instrument: str = "SYNTH"
    stage1_score: float = 0.9

    def __init__(self, call_num: int) -> None:
        self.pattern_id = f"stub_pid_{call_num}"
        self.ensemble_score = 0.75
        self.confidence_tier = "B"
        self.expected_direction = "bull"
        self.expected_move_atr = 0.8
        self.template_similarity = 0.6
        self.mahalanobis_score = 0.5
        self.contextual_score = 0.4
        self.match_window_start_ms = 1_700_000_000_000
        self.match_window_end_ms = 1_700_000_000_600
        self.regime = "normal"
        self.instrument = "SYNTH"
        self.stage1_score = 0.9


class _StubAccumulator:
    """Yields a 5-row window with a trivially-shaped DataFrame. The pipeline
    only checks ``len(l2_window) >= 5``; it does not inspect the columns."""
    def __init__(self, *, window_ok: bool = True) -> None:
        self.ingest_calls: int = 0
        self._window_ok = window_ok

    def ingest(self, snapshot) -> None:
        self.ingest_calls += 1

    def window(self):
        if not self._window_ok:
            return None
        idx = list(range(5))
        l2 = pd.DataFrame({"dummy": [0.0] * 5}, index=idx)
        l1 = pd.DataFrame({"dummy": [0.0] * 5}, index=idx)
        return l2, l1


def test_pipeline_default_no_correlation(sample_snapshot):
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(sample_snapshot)
    assert frame.correlation_matches == []


def test_pipeline_with_stub_engine_emits_matches(sample_snapshot):
    engine = _StubEngine()
    accum = _StubAccumulator()
    pipe = OrderBookMetaPipeline(
        correlation_engine=engine,
        feature_accumulator=accum,
        match_interval_ms=0,  # fire on every snapshot for this test
    )
    frame = pipe.run(sample_snapshot)

    assert engine.calls == 1
    assert accum.ingest_calls == 1
    assert len(frame.correlation_matches) == 1
    m = frame.correlation_matches[0]
    assert m["pattern_id"] == "stub_pid_1"
    assert m["confidence_tier"] == "B"
    assert m["expected_direction"] == "bull"
    assert 0.0 <= float(m["ensemble_score"]) <= 1.0


def test_pipeline_throttles_match_interval(sample_snapshot):
    """Second snapshot within the interval window must not trigger match()."""
    engine = _StubEngine()
    accum = _StubAccumulator()
    pipe = OrderBookMetaPipeline(
        correlation_engine=engine,
        feature_accumulator=accum,
        match_interval_ms=5_000,  # 5s throttle; two 600ms-apart snaps == 1 call
    )
    frame1 = pipe.run(sample_snapshot)
    # Second snapshot only 500ms later — should be throttled
    snap2 = sample_snapshot
    snap2 = type(snap2)(
        timestamp_ms=sample_snapshot.timestamp_ms + 500,
        symbol=sample_snapshot.symbol,
        bids=sample_snapshot.bids,
        asks=sample_snapshot.asks,
        recent_trades=sample_snapshot.recent_trades,
        recent_events=sample_snapshot.recent_events,
    )
    frame2 = pipe.run(snap2)

    assert accum.ingest_calls == 2       # accumulator keeps ingesting
    assert engine.calls == 1             # but match() only fired once
    assert len(frame1.correlation_matches) == 1
    assert frame2.correlation_matches == []


def test_pipeline_engine_failure_never_propagates(sample_snapshot):
    class _ExplodingEngine:
        def match(self, **_kw):
            raise RuntimeError("boom")
    pipe = OrderBookMetaPipeline(
        correlation_engine=_ExplodingEngine(),
        feature_accumulator=_StubAccumulator(),
        match_interval_ms=0,
    )
    frame = pipe.run(sample_snapshot)   # must not raise
    assert frame.correlation_matches == []


def test_pipeline_accumulator_failure_never_propagates(sample_snapshot):
    class _ExplodingAccumulator:
        def ingest(self, _snap):
            raise RuntimeError("ingest boom")
        def window(self):
            return None
    pipe = OrderBookMetaPipeline(
        correlation_engine=_StubEngine(),
        feature_accumulator=_ExplodingAccumulator(),
        match_interval_ms=0,
    )
    frame = pipe.run(sample_snapshot)
    assert frame.correlation_matches == []


def test_pipeline_warmup_returns_no_matches(sample_snapshot):
    """Fewer than 5 window rows must skip match() even at cadence."""
    engine = _StubEngine()
    accum = _StubAccumulator(window_ok=False)
    pipe = OrderBookMetaPipeline(
        correlation_engine=engine,
        feature_accumulator=accum,
        match_interval_ms=0,
    )
    frame = pipe.run(sample_snapshot)
    assert engine.calls == 0
    assert frame.correlation_matches == []


@pytest.mark.integration
def test_build_correlation_components_loads_live_artifacts():
    """Verify the engine_runner loader picks up library.yaml + CURRENT.json."""
    from p6v2.server.engine_runner import _build_correlation_components
    engine, accumulator = _build_correlation_components()
    # Build succeeds only when library.yaml + CURRENT.json exist — so this
    # test doubles as a smoke check for the promoted v1_nq_fwd1m model.
    if engine is None or accumulator is None:
        pytest.skip("p6lab artifacts not present in this environment")
    assert engine.model_version != "unloaded"
    assert len(engine.library.get_active_patterns()) >= 1


@pytest.mark.integration
def test_pipeline_with_live_engine_returns_frame(sample_snapshot):
    """End-to-end sanity: live engine from artifacts + sample snapshot → frame."""
    from p6v2.server.engine_runner import _build_correlation_components
    engine, accumulator = _build_correlation_components()
    if engine is None or accumulator is None:
        pytest.skip("p6lab artifacts not present in this environment")

    pipe = OrderBookMetaPipeline(
        correlation_engine=engine,
        feature_accumulator=accumulator,
        match_interval_ms=0,
    )
    frame = pipe.run(sample_snapshot)
    # Matches may be empty on the single-snapshot path (window warmup), but
    # the attribute must exist and be a list of dicts.
    assert isinstance(frame.correlation_matches, list)
    for m in frame.correlation_matches:
        assert isinstance(m, dict)
        assert "pattern_id" in m
        assert "ensemble_score" in m
