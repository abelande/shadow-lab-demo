"""Tests for p6lab.correlation.engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p6lab.correlation.engine import CorrelationEngine, MIN_MATCH_SCORE
from p6lab.correlation.scorer import EnsembleScorer
from p6lab.patterns.library import (
    OutcomeDistribution, PatternDefinition, PatternLibrary, PatternStatus,
)
from p6lab.patterns.template_matcher import (
    BOOK_SHAPE_DIM, MatchContext, PatternTemplate, TemplateMatcher,
)


def _ctx(instr: str = "NQ", vix: float = 18.0) -> MatchContext:
    return MatchContext(
        time_of_day_minutes=600, vix_level=vix,
        vix_regime="normal", relative_volume=1.0, instrument=instr,
    )


def _build_engine(tmp_path: Path) -> CorrelationEngine:
    lib = PatternLibrary(tmp_path / "library.yaml")
    lib.load()
    p = PatternDefinition(
        name="bull_breakout",
        l3_signature="burst_add", l2_manifestation="depth_lift", l1_footprint="spread_collapse",
        instruments=["NQ"], regime_specific=False,
        status=PatternStatus.ACTIVE,
        outcome_distribution={"5m": OutcomeDistribution(mean_atr=0.5, std=0.3, hit_rate=0.7, n=300)},
    )
    lib.add_pattern("bull_breakout", p)
    matcher = TemplateMatcher()
    matcher.templates["bull_breakout"] = PatternTemplate(
        pattern_id="bull_breakout",
        book_series=np.ones((10, BOOK_SHAPE_DIM)),
        feature_centroid=np.ones(12),
        pattern_context={"vix_regime": "normal"},
    )
    return CorrelationEngine(library=lib, matcher=matcher, scorer=EnsembleScorer())


def _l2_window(n: int = 10, bsv_value: float = 1.0) -> pd.DataFrame:
    df = pd.DataFrame({
        "book_shape_vector": [np.ones(BOOK_SHAPE_DIM) * bsv_value for _ in range(n)],
        "bid_ask_imbalance": np.zeros(n),
    }, index=[i * 100 for i in range(n)])
    return df


class TestMatch:
    def test_returns_matches_for_similar_state(self, tmp_path: Path):
        eng = _build_engine(tmp_path)
        matches = eng.match(_l2_window(), None, _ctx())
        assert len(matches) >= 1
        assert matches[0].pattern_id == "bull_breakout"
        assert matches[0].confidence_tier in ("A", "B", "C")
        assert matches[0].instrument == "NQ"

    def test_empty_when_no_active_patterns(self, tmp_path: Path):
        lib = PatternLibrary(tmp_path / "empty.yaml")
        lib.load()
        eng = CorrelationEngine(lib, TemplateMatcher(), EnsembleScorer())
        assert eng.match(_l2_window(), None, _ctx()) == []

    def test_capped_at_max(self, tmp_path: Path):
        from p6lab.correlation.engine import MAX_MATCHES_PER_CALL
        eng = _build_engine(tmp_path)
        # Add many duplicate templates
        for i in range(MAX_MATCHES_PER_CALL + 10):
            pid = f"copy_{i}"
            eng.matcher.templates[pid] = eng.matcher.templates["bull_breakout"]
            # Also add to library so get_active_patterns returns them
            from p6lab.patterns.library import PatternDefinition
            eng.library._data.patterns[pid] = PatternDefinition(  # type: ignore[union-attr]
                name=pid, l3_signature="x", l2_manifestation="y", l1_footprint="z",
                instruments=["NQ"], regime_specific=False, status=PatternStatus.ACTIVE,
            )
        eng._current_regime = None  # reset cache
        matches = eng.match(_l2_window(), None, _ctx())
        assert len(matches) <= MAX_MATCHES_PER_CALL


class TestReload:
    def test_reload_library_clears_cache(self, tmp_path: Path):
        eng = _build_engine(tmp_path)
        eng.match(_l2_window(), None, _ctx())  # populate cache
        assert eng._current_regime is not None
        new_lib = PatternLibrary(tmp_path / "new.yaml")
        new_lib.load()
        eng.reload_library(new_lib)
        assert eng._current_regime is None
        assert eng._active_pattern_ids == []


class TestStage1:
    def test_returns_passthrough_when_no_template(self, tmp_path: Path):
        eng = _build_engine(tmp_path)
        score = eng._stage1_prescreen(np.zeros(16), "nonexistent")
        assert score == 1.0

    def test_perfect_match_high_score(self, tmp_path: Path):
        eng = _build_engine(tmp_path)
        # Centroid is np.ones(12); l1_features matching 12 dims to it
        score = eng._stage1_prescreen(np.ones(12), "bull_breakout")
        assert score == 1.0
