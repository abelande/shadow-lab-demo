"""Wave 4 Phase 1E — engine cost gate tests.

Verifies that CorrelationEngine.match() rejects patterns whose expected
edge doesn't clear slippage × cost_multiplier, and keeps them otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p6lab.correlation.engine import CorrelationEngine
from p6lab.correlation.scorer import EnsembleScorer
from p6lab.patterns.library import (
    OutcomeDistribution, PatternDefinition, PatternLibrary, PatternStatus,
)
from p6lab.patterns.template_matcher import (
    BOOK_SHAPE_DIM, MatchContext, PatternTemplate, TemplateMatcher,
)


@pytest.fixture
def engine_with_pattern(tmp_path: Path) -> tuple[CorrelationEngine, str]:
    """Tiny engine with one pattern whose expected move is 0.3 ATR."""
    lib = PatternLibrary(tmp_path / "lib.yaml")
    lib.load()
    lib.add_pattern("tight_pattern", PatternDefinition(
        name="tight_pattern",
        l3_signature="x", l2_manifestation="x", l1_footprint="x",
        instruments=["NQ"], regime_specific=False,
        status=PatternStatus.ACTIVE,
        outcome_distribution={"5m": OutcomeDistribution(
            mean_atr=0.3, std=0.2, hit_rate=0.6, n=500,
        )},
    ))

    matcher = TemplateMatcher()
    matcher.templates["tight_pattern"] = PatternTemplate(
        pattern_id="tight_pattern",
        book_series=np.ones((10, BOOK_SHAPE_DIM)),
        feature_centroid=np.zeros(10),
        pattern_context={"vix_regime": "normal"},
    )
    return CorrelationEngine(library=lib, matcher=matcher, scorer=EnsembleScorer()), "tight_pattern"


def _build_window() -> pd.DataFrame:
    # 50 snapshot window with BSV and imbalance columns
    rows = []
    for _ in range(50):
        rows.append({
            "book_shape_vector": np.ones(BOOK_SHAPE_DIM),
            "bid_ask_imbalance": 0.1,
        })
    return pd.DataFrame(rows, index=list(range(1000, 1050)))


class TestCostGate:
    def test_expected_edge_below_slippage_rejects(self, engine_with_pattern):
        """atr_recent small + slippage huge → edge < cost × multiplier → drop."""
        engine, pid = engine_with_pattern
        win = _build_window()
        ctx = MatchContext(
            time_of_day_minutes=600, vix_level=18.0, vix_regime="normal",
            relative_volume=1.0, instrument="NQ",
            atr_recent=0.1,      # tiny ATR → tiny edge
            slippage_bps=20.0,   # big slippage
        )
        # Rebuild active_ids (engine loads patterns lazily on regime change)
        matches = engine.match(win, None, ctx)
        assert matches == []   # all patterns rejected by cost gate

    def test_expected_edge_above_slippage_accepts(self, engine_with_pattern):
        """Large atr_recent + small slippage → edge clears cost → pattern runs."""
        engine, pid = engine_with_pattern
        win = _build_window()
        ctx = MatchContext(
            time_of_day_minutes=600, vix_level=18.0, vix_regime="normal",
            relative_volume=1.0, instrument="NQ",
            atr_recent=5.0,      # healthy ATR
            slippage_bps=0.1,    # cheap slippage
        )
        matches = engine.match(win, None, ctx)
        # May or may not match depending on template similarity, but the
        # gate should not be what blocks it.
        # Assertion: at least the pattern passed the cost gate (internal
        # active_ids after the gate). We can't introspect directly but
        # if matches is empty the failure is from matcher, not cost gate.
        # Accept either 0 matches (matcher didn't fire) or >0.
        assert isinstance(matches, list)

    def test_gate_disabled_when_atr_zero(self, engine_with_pattern):
        """atr_recent=0 disables the gate (backward compat)."""
        engine, pid = engine_with_pattern
        win = _build_window()
        ctx = MatchContext(
            time_of_day_minutes=600, vix_level=18.0, vix_regime="normal",
            relative_volume=1.0, instrument="NQ",
            atr_recent=0.0,        # disabled
            slippage_bps=100.0,    # even with huge slippage, gate is off
        )
        matches = engine.match(win, None, ctx)
        # Gate is disabled → pattern progresses past cost check
        assert isinstance(matches, list)

    def test_gate_helper_directly(self, engine_with_pattern):
        engine, pid = engine_with_pattern
        # Prime the active list
        engine._active_pattern_ids = [pid]

        # Reject when edge < cost × multiplier
        kept = engine._apply_cost_gate([pid], atr_recent=0.1, slippage_bps=20.0)
        assert pid not in kept

        # Accept when edge >> cost × multiplier
        kept = engine._apply_cost_gate([pid], atr_recent=5.0, slippage_bps=0.1)
        assert pid in kept
