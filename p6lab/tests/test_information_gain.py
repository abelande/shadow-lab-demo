"""Tests for p6lab.validation.information_gain."""
from __future__ import annotations

import numpy as np
import pytest

from p6lab.validation.information_gain import must_beat_baseline


class TestDecision:
    def test_rejects_below_min_improvement(self):
        report = must_beat_baseline(0.71, 0.70, min_improvement=0.02)
        assert report.approved is False
        assert "improvement" in report.reason.lower()

    def test_approves_clear_improvement(self):
        report = must_beat_baseline(0.80, 0.70, min_improvement=0.02)
        assert report.approved is True
        assert report.absolute_improvement == pytest.approx(0.10)

    def test_relative_improvement_calc(self):
        report = must_beat_baseline(0.55, 0.50)
        assert report.relative_improvement == pytest.approx(0.10)


class TestBootstrap:
    def test_bootstrap_with_samples(self):
        rng = np.random.default_rng(0)
        baseline_samples = rng.normal(0.50, 0.05, 200)
        candidate_samples = rng.normal(0.60, 0.05, 200)
        report = must_beat_baseline(
            candidate_metric=0.60, baseline_metric=0.50,
            candidate_samples=candidate_samples, baseline_samples=baseline_samples,
        )
        assert report.approved is True
        assert report.ci_low > 0
        assert report.p_value is not None
        assert report.p_value < 0.05

    def test_bootstrap_indistinguishable(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.5, 0.20, 100)
        b = rng.normal(0.51, 0.20, 100)
        report = must_beat_baseline(
            candidate_metric=0.51, baseline_metric=0.50,
            candidate_samples=a, baseline_samples=b,
        )
        # Tiny gap → can't approve
        assert report.approved is False
