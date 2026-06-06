"""Tests for p6lab.features.fragility_index."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p6lab.features.fragility_index import (
    FI_FAST_WEIGHTS, FI_FULL_WEIGHTS, FragilityIndex,
)


class TestSubIndices:
    def test_returns_six_clamped_values(self):
        fi = FragilityIndex()
        l1 = np.zeros(16)
        l2 = np.zeros(12)
        sub = fi.compute_sub_indices(l1, l2, vpin_value=0.5)
        for v in (sub.DF, sub.CF, sub.RF, sub.SF, sub.FT, sub.CIS):
            assert 0.0 <= v <= 1.0

    def test_depth_depletion_increases_df(self):
        fi = FragilityIndex()
        l1 = np.zeros(16)
        l2_calm = np.zeros(12)
        l2_drain = np.zeros(12); l2_drain[6] = -100  # heavy depletion
        sub_calm = fi.compute_sub_indices(l1, l2_calm, 0.0)
        sub_drain = fi.compute_sub_indices(l1, l2_drain, 0.0)
        assert sub_drain.DF > sub_calm.DF

    def test_vpin_passes_through_to_ft(self):
        fi = FragilityIndex()
        sub = fi.compute_sub_indices(np.zeros(16), np.zeros(12), vpin_value=0.7)
        assert sub.FT == pytest.approx(0.7)


class TestComposites:
    def test_fi_fast_is_weighted_sum(self):
        fi = FragilityIndex()
        v = fi.compute_fast(rf=1.0, sf=1.0, ft=1.0)
        assert v == pytest.approx(sum(FI_FAST_WEIGHTS.values()))

    def test_full_includes_all_six(self):
        fi = FragilityIndex()
        # Use a negative depth_change_5s so DF saturates too
        l1 = np.ones(16) * 100
        l2 = np.ones(12) * 100
        l2[6] = -300   # depletion → DF = 1.0
        sub = fi.compute_sub_indices(l1, l2, vpin_value=1.0, cross_instrument_stress=1.0)
        v = fi.compute_full(sub)
        assert v == pytest.approx(sum(FI_FULL_WEIGHTS.values()))


class TestSeries:
    def test_bulk_returns_correct_columns(self):
        fi = FragilityIndex()
        l1 = pd.DataFrame(np.zeros((5, 16)))
        l2 = pd.DataFrame(np.zeros((5, 12)))
        vpin = pd.Series([0.0] * 5)
        df = fi.compute_series(l1, l2, vpin)
        for col in ("DF", "CF", "RF", "SF", "FT", "CIS",
                    "fi_fast", "fi_full", "signal_threshold", "size_multiplier"):
            assert col in df.columns
