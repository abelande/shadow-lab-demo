"""
Latency benchmark for ``CorrelationEngine.match()``.

Spec target: **p95 < 50ms** per call on a synthetic 5-minute L2 window.

This is a lightweight pytest that will fail the build if the engine
regresses below the spec. Not a micro-benchmark — uses 200 repeated
calls with a realistic feature width but no I/O. Intentionally does
not depend on pytest-benchmark so `make smoketest` can include it
without a third-party plugin.

Run:
    pytest tests/test_engine_latency.py -v
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src/ + p6-v2 root to sys.path (same pattern as notebooks/_common.py).
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PROJECTS = ROOT.parent.parent
for p in (str(SRC), str(PROJECTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from p6lab.correlation.engine import CorrelationEngine   # noqa: E402
from p6lab.correlation.scorer import EnsembleScorer      # noqa: E402
from p6lab.patterns.library import (                      # noqa: E402
    PatternDefinition, PatternLibrary, PatternStatus,
)
from p6lab.patterns.template_matcher import (             # noqa: E402
    MatchContext, TemplateMatcher,
)

N_CALLS = 200
WINDOW_ROWS = 3000        # 5min × 60s × 10 snaps/sec
BOOK_SHAPE_DIM = 40
L1_FEATURES = 16
L2_FEATURES = 12
P95_THRESHOLD_MS = 50.0


@pytest.fixture
def engine(tmp_path: Path) -> CorrelationEngine:
    """Build a fresh engine with an in-memory trained model.

    Uses a single synthetic pattern template + centroid so the matcher has
    something to score against. No disk I/O; load cost doesn't bleed into
    per-call latency.
    """
    lib = PatternLibrary(tmp_path / "library.yaml")
    lib.load()
    # Give the library an active pattern matching our synthetic template so
    # the regime conditioner doesn't short-circuit match() to an empty list.
    lib.add_pattern("synthetic_up", PatternDefinition(
        name="synthetic_up",
        l3_signature="bench-only",
        l2_manifestation="bench-only",
        l1_footprint="bench-only",
        status=PatternStatus.ACTIVE,
        instruments=["NQ"],
        regime_specific=False,
    ))
    matcher = TemplateMatcher()
    scorer = EnsembleScorer()
    eng = CorrelationEngine(library=lib, matcher=matcher, scorer=scorer)

    # Inject a synthetic trained-model pickle, then reload it — exercises
    # the same code path a real notebook-trained model takes.
    model_path = tmp_path / "model.pkl"
    template = np.random.default_rng(42).normal(size=(10, BOOK_SHAPE_DIM)).astype(float)
    centroid = np.random.default_rng(43).normal(size=(L2_FEATURES,)).astype(float)
    cov = np.eye(BOOK_SHAPE_DIM, dtype=float)
    with open(model_path, "wb") as fh:
        pickle.dump(
            {
                "version": "latency_bench_v1",
                "matcher_templates": {"synthetic_up": template},
                "matcher_centroids": {"synthetic_up": centroid},
                "pattern_contexts":  {"synthetic_up": {"vix_regime": "normal"}},
                "global_covariance": cov,
                "feature_names": [f"f{i}" for i in range(L2_FEATURES)],
                "cv_auc": 0.75,
            },
            fh,
        )
    eng.reload_model(str(model_path))
    return eng


def _build_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    ts = np.arange(WINDOW_ROWS) * 100   # 100ms cadence
    l2 = pd.DataFrame(
        rng.normal(size=(WINDOW_ROWS, L2_FEATURES)),
        columns=[f"l2_f{i}" for i in range(L2_FEATURES)],
        index=ts,
    )
    l2["book_shape_vector"] = [
        rng.normal(size=BOOK_SHAPE_DIM) for _ in range(WINDOW_ROWS)
    ]
    l1 = pd.DataFrame(
        rng.normal(size=(WINDOW_ROWS, L1_FEATURES)),
        columns=[f"l1_f{i}" for i in range(L1_FEATURES)],
        index=ts,
    )
    return l2, l1


@pytest.mark.benchmark
def test_match_p95_under_50ms(engine: CorrelationEngine) -> None:
    """p95 latency of CorrelationEngine.match() must stay under 50ms."""
    l2, l1 = _build_windows()
    context = MatchContext(
        time_of_day_minutes=570,   # 09:30 NY
        vix_level=18.0,
        vix_regime="normal",
        relative_volume=1.2,
        instrument="NQ",
    )

    # Warm-up — first call pays JIT / cache setup costs we don't want
    # to charge against the steady-state p95.
    for _ in range(5):
        engine.match(l2_window=l2, l1_window=l1, context=context)

    samples_ns: list[int] = []
    for _ in range(N_CALLS):
        t0 = time.perf_counter_ns()
        engine.match(l2_window=l2, l1_window=l1, context=context)
        samples_ns.append(time.perf_counter_ns() - t0)

    samples_ms = np.asarray(samples_ns, dtype=float) / 1_000_000.0
    p50 = float(np.percentile(samples_ms, 50))
    p95 = float(np.percentile(samples_ms, 95))
    p99 = float(np.percentile(samples_ms, 99))

    print(
        f"\nCorrelationEngine.match latency over {N_CALLS} calls: "
        f"p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms"
    )
    assert p95 < P95_THRESHOLD_MS, (
        f"engine match latency regressed: p95={p95:.2f}ms "
        f"(threshold {P95_THRESHOLD_MS}ms)"
    )
