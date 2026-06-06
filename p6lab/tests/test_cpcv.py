"""Tests for p6lab.validation.cpcv — combinatorial purged CV + cascade embargo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p6lab.validation.cpcv import CascadeAwareCPCV, CPCVFold


def _ts(n: int, days_apart: int = 1) -> pd.Series:
    return pd.Series(pd.date_range("2025-01-01", periods=n, freq=f"{days_apart}D"))


class TestSplit:
    def test_returns_combinatorial_fold_count(self):
        cv = CascadeAwareCPCV(n_splits=5, n_test_groups=2)
        X = pd.DataFrame(np.zeros((100, 3)))
        ts = _ts(100)
        folds = cv.split(X, ts)
        # C(5, 2) = 10
        assert len(folds) == 10

    def test_folds_disjoint_train_test(self):
        cv = CascadeAwareCPCV(n_splits=4, n_test_groups=1)
        X = pd.DataFrame(np.zeros((40, 2)))
        folds = cv.split(X, _ts(40))
        for f in folds:
            assert len(np.intersect1d(f.train_idx, f.test_idx)) == 0

    def test_test_indices_cover_dataset(self):
        cv = CascadeAwareCPCV(n_splits=4, n_test_groups=1)
        X = pd.DataFrame(np.zeros((40, 2)))
        folds = cv.split(X, _ts(40))
        union = sorted(set().union(*[f.test_idx.tolist() for f in folds]))
        assert union == list(range(40))


class TestEmbargo:
    def test_cascade_purges_nearby_train_rows(self):
        cv = CascadeAwareCPCV(n_splits=4, n_test_groups=1, cascade_embargo_days=14)
        X = pd.DataFrame(np.zeros((100, 2)))
        ts = _ts(100, days_apart=1)
        # One cascade event at day 50
        cascades = pd.Series([ts.iloc[50]])
        folds = cv.split(X, ts, cascades)
        # Total embargoed rows across all folds should be > 0
        embargoed_total = sum(len(f.embargoed_idx) for f in folds)
        assert embargoed_total > 0

    def test_no_embargo_without_cascades(self):
        cv = CascadeAwareCPCV(n_splits=4, n_test_groups=1)
        X = pd.DataFrame(np.zeros((40, 2)))
        folds = cv.split(X, _ts(40))
        assert all(len(f.embargoed_idx) == 0 for f in folds)
