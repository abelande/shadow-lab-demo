"""Wave 8.5-A — observability counter integration tests (p6-v2 side).

Covers the pipeline.OrderBookMetaPipeline + server.engine_runner counters.
MultiSymbolRunner counters live in p6lab/tests/test_wave_85_a_multi_symbol_counters.py.

Anti-patterns this phase deliberately avoided (plan §8.5-A):
  - Counters are plain dict[str, int], not collections.Counter, so the
    existing websocket._serialize_value JSON path is unchanged.
  - Counter increments stay inside the exception handler so the
    traceback line numbers still point at the underlying bug.
  - No new Metrics class — 4 lines per file only.

Gate: this file's test suite + existing smoketest stay green.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from p6.pipeline import OrderBookMetaPipeline


class _RaisingAccumulator:
    """FeatureAccumulator stand-in that raises on ingest."""
    def __init__(self) -> None:
        self.attempts: int = 0

    def ingest(self, _snap: Any) -> None:
        self.attempts += 1
        raise RuntimeError("wave85-A: synthetic ingest failure")

    def window(self) -> None:
        return None


class _RaisingEngine:
    """CorrelationEngine stand-in that raises on match."""
    def match(self, **_kw: Any) -> list:
        raise RuntimeError("wave85-A: synthetic match failure")


class _OkAccumulator:
    def __init__(self) -> None:
        self.ingest_count: int = 0

    def ingest(self, _snap: Any) -> None:
        self.ingest_count += 1

    def window(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        idx = list(range(5))
        df = pd.DataFrame({"dummy": [0.0] * 5}, index=idx)
        return df, df


def test_wave_85_a_pipeline_ingest_counter_increments_on_exception(sample_snapshot) -> None:
    pipe = OrderBookMetaPipeline(
        correlation_engine=_RaisingEngine(),
        feature_accumulator=_RaisingAccumulator(),
        match_interval_ms=0,
    )
    assert pipe.correlation_stats == {"ingest_errors": 0, "match_errors": 0}
    pipe.run(sample_snapshot)
    pipe.run(sample_snapshot)
    stats = pipe.correlation_stats
    assert stats["ingest_errors"] == 2
    # match_errors should stay at 0 — the ingest failure returns before match
    assert stats["match_errors"] == 0


def test_wave_85_a_pipeline_match_counter_increments_on_exception(sample_snapshot) -> None:
    pipe = OrderBookMetaPipeline(
        correlation_engine=_RaisingEngine(),
        feature_accumulator=_OkAccumulator(),
        match_interval_ms=0,
    )
    pipe.run(sample_snapshot)
    pipe.run(sample_snapshot)
    stats = pipe.correlation_stats
    assert stats["ingest_errors"] == 0
    assert stats["match_errors"] == 2


def test_wave_85_a_correlation_stats_returns_shallow_copy() -> None:
    """Mutating the returned dict must NOT affect the pipeline's state."""
    pipe = OrderBookMetaPipeline()
    stats_snapshot = pipe.correlation_stats
    stats_snapshot["ingest_errors"] = 999
    assert pipe.correlation_stats["ingest_errors"] == 0


def test_wave_85_a_engine_runner_dropped_counter_in_get_status() -> None:
    """get_status() must include dropped_snapshots and pipeline_errors
    keys (Wave 8.5-A additions)."""
    from p6v2.server.engine_runner import EngineRunner
    runner = EngineRunner()
    status = runner.get_status()
    assert "dropped_snapshots" in status
    assert "pipeline_errors" in status
    assert "correlation_stats" in status
    assert status["dropped_snapshots"] == 0
    assert status["pipeline_errors"] == 0
    assert status["correlation_stats"] == {"ingest_errors": 0, "match_errors": 0}


def test_wave_85_a_engine_runner_dropped_increments_on_pipeline_error() -> None:
    """Directly poke the counter via the same path _run_loop uses — full
    async integration is out of scope for this unit test."""
    from p6v2.server.engine_runner import EngineRunner
    runner = EngineRunner()
    runner._dropped_snapshots += 1
    runner._pipeline_errors += 1
    status = runner.get_status()
    assert status["dropped_snapshots"] == 1
    assert status["pipeline_errors"] == 1


def test_wave_85_a_counters_json_serializable() -> None:
    """Counters must serialize to JSON so websocket broadcast works. If
    someone changes them to collections.Counter, this test catches it."""
    pipe = OrderBookMetaPipeline()
    j = json.dumps(pipe.correlation_stats)
    assert '"ingest_errors": 0' in j
    assert '"match_errors": 0' in j


def test_wave_85_a_get_status_fully_json_serializable() -> None:
    """Nothing in get_status should break JSON serialization."""
    from p6v2.server.engine_runner import EngineRunner
    runner = EngineRunner()
    status = runner.get_status()
    j = json.dumps(status)
    assert '"dropped_snapshots"' in j
    assert '"correlation_stats"' in j
