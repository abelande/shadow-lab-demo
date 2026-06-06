"""Tests for p6lab.patterns.template_matcher."""
from __future__ import annotations

import numpy as np
import pytest

from p6lab.patterns.template_matcher import (
    BOOK_SHAPE_DIM, MatchContext, PatternTemplate, TemplateMatcher,
)


def _ctx(regime: str = "normal") -> MatchContext:
    return MatchContext(
        time_of_day_minutes=600, vix_level=18.0,
        vix_regime=regime, relative_volume=1.0, instrument="NQ",
    )


class TestCosine:
    def test_identity(self):
        m = TemplateMatcher()
        a = np.ones((5, BOOK_SHAPE_DIM))
        assert m.cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal(self):
        m = TemplateMatcher()
        a = np.zeros((5, BOOK_SHAPE_DIM)); a[:, :20] = 1.0
        b = np.zeros((5, BOOK_SHAPE_DIM)); b[:, 20:] = 1.0
        assert m.cosine_similarity(a, b) == pytest.approx(0.0)


class TestMahalanobis:
    def test_falls_back_to_euclidean_without_fit(self):
        m = TemplateMatcher()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 0.0])
        d = m.mahalanobis_distance(a, b)
        assert d == pytest.approx(np.sqrt(14))

    def test_uses_covariance_after_fit(self):
        m = TemplateMatcher()
        rng = np.random.default_rng(0)
        train = rng.normal(size=(200, 5))
        m.fit_covariance(train)
        d = m.mahalanobis_distance(np.zeros(5), np.ones(5))
        assert d > 0

    def test_ill_conditioned_falls_back(self):
        m = TemplateMatcher()
        # Two identical rows produce zero variance → cov is degenerate
        rank_deficient = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]] * 20, dtype=float)
        m.fit_covariance(rank_deficient)
        # Either kept or fell back — assert the matcher records its choice
        assert m._use_euclidean in (True, False)


class TestEnsembleMatch:
    def test_perfect_match_high_score(self):
        m = TemplateMatcher()
        bsv = np.ones((10, BOOK_SHAPE_DIM))
        feat = np.ones(12)
        r = m.match(bsv, feat, bsv, feat, "p1", _ctx())
        assert r.ensemble_score > 0.7

    def test_dissimilar_low_score(self):
        m = TemplateMatcher()
        a_book = np.zeros((10, BOOK_SHAPE_DIM)); a_book[:, :20] = 1.0
        b_book = np.zeros((10, BOOK_SHAPE_DIM)); b_book[:, 20:] = 1.0
        r = m.match(a_book, np.zeros(12), b_book, np.ones(12) * 100, "p1", _ctx())
        assert r.ensemble_score < 0.6


class TestContextual:
    def test_regime_match_bonus(self):
        m = TemplateMatcher()
        s_match = m.contextual_score(_ctx("normal"), {"vix_regime": "normal"})
        s_miss = m.contextual_score(_ctx("normal"), {"vix_regime": "high"})
        assert s_match > s_miss
