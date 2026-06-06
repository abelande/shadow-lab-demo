"""Tests for p6lab.correlation.scorer."""
from __future__ import annotations

import pytest

from p6lab.correlation.scorer import EnsembleScorer
from p6lab.patterns.template_matcher import MatchResult


def _m(score: float, pid: str = "p1") -> MatchResult:
    return MatchResult(
        pattern_id=pid,
        template_cosine_similarity=0.5,
        mahalanobis_distance=1.0,
        contextual_score=0.5,
        ensemble_score=score,
        used_euclidean_fallback=False,
    )


class TestTierAssignment:
    def test_tier_a(self):
        s = EnsembleScorer().score(_m(0.90))
        assert s and s.confidence_tier == "A"
        assert s.action == "auto_alert_and_size"

    def test_tier_b(self):
        s = EnsembleScorer().score(_m(0.78))
        assert s and s.confidence_tier == "B"

    def test_tier_c(self):
        s = EnsembleScorer().score(_m(0.65))
        assert s and s.confidence_tier == "C"

    def test_below_c_discards(self):
        assert EnsembleScorer().score(_m(0.55)) is None


class TestPrecisionDemotion:
    def test_demotes_when_precision_low(self):
        # Pattern has 0.50 precision at A → must demote to B (precision 0.80)
        scorer = EnsembleScorer(precision_by_pattern={"p1": {"A": 0.50, "B": 0.80, "C": 0.90}})
        s = scorer.score(_m(0.90))
        assert s is not None
        assert s.confidence_tier == "B"
        assert s.demoted is True

    def test_discards_when_no_tier_passes(self):
        scorer = EnsembleScorer(precision_by_pattern={"p1": {"A": 0.1, "B": 0.1, "C": 0.1}})
        assert scorer.score(_m(0.90)) is None


class TestBatch:
    def test_filters_and_sorts(self):
        scorer = EnsembleScorer()
        results = scorer.score_batch([_m(0.55), _m(0.95), _m(0.65)])
        assert len(results) == 2
        assert results[0].ensemble_score > results[1].ensemble_score
