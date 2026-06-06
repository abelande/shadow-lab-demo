"""Tests for p6lab.execution.cost_model."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from p6lab.execution.cost_model import CostBreakdown, CostModel


@dataclass
class _Outcome:
    filled: bool
    filled_size: float
    adverse_ticks_at_fill: float = 0.0


class TestComputeOne:
    def test_passive_no_crossed_spread(self):
        m = CostModel()
        out = _Outcome(filled=True, filled_size=1.0)
        b = m.compute(out, entry_type="passive", spread_ticks=1.0)
        assert b.crossed_spread_cost == 0.0
        assert b.commission == pytest.approx(2.04)

    def test_aggressive_pays_half_spread(self):
        m = CostModel(tick_value=5.0)
        out = _Outcome(filled=True, filled_size=2.0)
        b = m.compute(out, entry_type="aggressive", spread_ticks=2.0)
        # 0.5 × 2 ticks × $5 × 2 = $10
        assert b.crossed_spread_cost == pytest.approx(10.0)

    def test_adverse_only_on_filled_size(self):
        m = CostModel(tick_value=5.0)
        out = _Outcome(filled=True, filled_size=3.0, adverse_ticks_at_fill=2.0)
        b = m.compute(out, entry_type="passive")
        # 2 × 5 × 3 = 30
        assert b.adverse_selection_cost == pytest.approx(30.0)

    def test_opportunity_cost_for_unfilled(self):
        m = CostModel(tick_value=5.0, expected_edge_ticks=2.0)
        out = _Outcome(filled=False, filled_size=0.0)
        b = m.compute(out, entry_type="passive", size=1.0, pfill=0.0)
        # 2 × 5 × 1 × (1-0) = 10
        assert b.opportunity_cost == pytest.approx(10.0)

    def test_total_is_sum(self):
        m = CostModel()
        out = _Outcome(filled=True, filled_size=1.0, adverse_ticks_at_fill=1.0)
        b = m.compute(out, entry_type="aggressive", spread_ticks=1.0)
        expected = b.crossed_spread_cost + b.commission + b.adverse_selection_cost + b.opportunity_cost
        assert b.total_cost == pytest.approx(expected)


class TestBatch:
    def test_batch_matches_individual(self):
        m = CostModel()
        outs = [_Outcome(filled=True, filled_size=1.0) for _ in range(5)]
        batch = m.compute_batch(outs)
        assert len(batch) == 5

    def test_mismatched_length_raises(self):
        m = CostModel()
        outs = [_Outcome(filled=True, filled_size=1.0) for _ in range(3)]
        with pytest.raises(ValueError):
            m.compute_batch(outs, entry_types=["passive"] * 2)


class TestCompare:
    def test_realistic_higher_when_adverse(self):
        m = CostModel()
        outs = [_Outcome(filled=True, filled_size=1.0, adverse_ticks_at_fill=2.0)]
        breakdowns = m.compute_batch(outs)
        cmp = m.compare_to_naive(breakdowns)
        assert cmp["realistic_total"] > cmp["naive_total"]
        assert cmp["adverse_selection_pct"] > 0
