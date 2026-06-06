"""
LiveRunner integration smoketest.

Runs the full live pipeline end-to-end against ``MockLiveFeed`` (the
committed 15-min NQ sample) for a short duration and asserts:

  - Snapshots ingested > 0
  - Engine match() called (> 0 times)
  - Accumulator populated features
  - At least one renderer attached to the broker
  - Graceful duration-based shutdown

Does not require any network, any live credentials, or the real
DatabentoLiveFeed. Safe for CI.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PROJECTS = ROOT.parent.parent
DATA_SAMPLE = ROOT.parent / "data" / "nq-mbo-sample-15min.dbn.zst"
for p in (str(SRC), str(PROJECTS), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from p6lab.correlation.engine import CorrelationEngine
from p6lab.correlation.match_broker import MatchBroker
from p6lab.correlation.scorer import EnsembleScorer
from p6lab.live.feature_accumulator import FeatureAccumulator
from p6lab.live.runner import LiveConfig, LiveRunner
from p6lab.patterns.library import PatternLibrary
from p6lab.patterns.template_matcher import TemplateMatcher

# Load MockLiveFeed via absolute path (tests/ is not a package)
import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "mock_live_feed", ROOT / "tests" / "fixtures" / "mock_live_feed.py",
)
_mlf_mod = _iu.module_from_spec(_spec); _spec.loader.exec_module(_mlf_mod)
MockLiveFeed = _mlf_mod.MockLiveFeed


def _skip_if_no_sample():
    if not DATA_SAMPLE.is_file():
        pytest.skip(f"sample data missing: {DATA_SAMPLE}")


# ---------------------------------------------------------------------------
# FeatureAccumulator — unit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_feature_accumulator_produces_engine_ready_windows():
    _skip_if_no_sample()
    from p6.ingestion.databento_feed import DatabentoReplayFeed

    feed = DatabentoReplayFeed(
        file_path=str(DATA_SAMPLE), symbol="NQ",
        filter_symbol="NQ", num_levels=20,
    )
    await feed.connect()

    acc = FeatureAccumulator(tick_size=0.25, window_seconds=30.0, num_levels=20)
    n_ingested = 0
    for _ in range(200):
        snap = await feed.next()
        if snap is None:
            break
        row = acc.ingest(snap)
        if row is not None:
            n_ingested += 1
    await feed.disconnect()

    assert n_ingested >= 30, f"expected >=30 rows ingested, got {n_ingested}"
    windows = acc.window()
    assert windows is not None
    l2_df, l1_df = windows
    assert len(l2_df) > 0
    assert len(l1_df) == len(l2_df)
    assert "book_shape_vector" in l2_df.columns
    # BSV column is 40-dim arrays
    assert len(l2_df["book_shape_vector"].iloc[0]) == 40
    # L1 has the full feature set
    from p6lab.features.l1_features import L1FeatureNames
    assert list(l1_df.columns) == list(L1FeatureNames.ALL)


# ---------------------------------------------------------------------------
# LiveRunner — integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_runner_end_to_end_via_mock(tmp_path):
    _skip_if_no_sample()

    audit_path = tmp_path / "matches.jsonl"
    cfg = LiveConfig(
        symbol="NQ", dataset="GLBX.MDP3",
        window_seconds=20.0,
        match_interval_ms=250,   # call match() every 250ms for the test
        audit_log_path=audit_path,
        enable_metrics=True,
    )

    def feed_factory():
        return MockLiveFeed(
            source_file=str(DATA_SAMPLE), symbol="NQ",
            filter_symbol="NQ", num_levels=20,
        )

    def engine_factory(broker: MatchBroker):
        lib = PatternLibrary(tmp_path / "library.yaml"); lib.load()
        # No active patterns → match() returns empty list but the call chain still
        # exercises every integration seam. We want to assert that engine.match
        # was CALLED, not that matches were produced.
        return CorrelationEngine(
            library=lib, matcher=TemplateMatcher(),
            scorer=EnsembleScorer(), broker=broker,
        )

    runner = LiveRunner(cfg, feed_factory=feed_factory, engine_factory=engine_factory)
    stats = await runner.run(duration_seconds=3.0)

    assert stats["snapshots_ingested"] > 10, (
        f"expected >10 snapshots in 3s, got {stats['snapshots_ingested']}"
    )
    assert stats["match_calls"] > 0, "engine.match() should have been called"
    # Renderer handles are present
    assert "metrics" in runner.renderer_handles
    assert "audit" in runner.renderer_handles
    # Broker has at least the metrics + audit subscribers
    assert runner.broker.subscriber_count >= 2


@pytest.mark.asyncio
async def test_live_runner_respects_duration(tmp_path):
    """Runner exits cleanly after the configured duration."""
    _skip_if_no_sample()

    cfg = LiveConfig(symbol="NQ", enable_metrics=False)

    def feed_factory():
        return MockLiveFeed(
            source_file=str(DATA_SAMPLE), symbol="NQ",
            filter_symbol="NQ", num_levels=20,
        )

    def engine_factory(broker):
        lib = PatternLibrary(tmp_path / "library.yaml"); lib.load()
        return CorrelationEngine(
            library=lib, matcher=TemplateMatcher(),
            scorer=EnsembleScorer(), broker=broker,
        )

    runner = LiveRunner(cfg, feed_factory=feed_factory, engine_factory=engine_factory)
    import time
    t0 = time.monotonic()
    await runner.run(duration_seconds=1.5)
    elapsed = time.monotonic() - t0
    # duration-exit should happen reasonably close to 1.5s (allow 3s upper bound
    # for CI jitter + the feed shutdown path)
    assert 1.0 <= elapsed <= 3.0, f"duration-exit out of band: {elapsed:.2f}s"
