"""
Multi-day CPCV tests.

Covers the ``trading_day_aware=True`` extension to ``CascadeAwareCPCV``:

  - No fold's train and test split crosses the same calendar day
  - min_train_days / min_test_days guards drop bad folds
  - Trading-day embargo counts in day-count units, not wall-clock days
    (weekends don't consume embargo budget)
  - Backward compatibility: the old (non-day-aware) path still works

The synthetic fixture builds a 30-trading-day series at 1-minute cadence,
~450 rows/day, ~13.5k rows total — production-realistic scale.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from p6lab.validation.cpcv import CascadeAwareCPCV


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_multiday_series(n_days: int = 30, minutes_per_day: int = 450,
                           start_date: str = "2026-03-02") -> pd.DataFrame:
    """Return a DataFrame + timestamps Series simulating N trading days.

    Skips weekends (Mon-Fri only) so the calendar-day set is contiguous
    but wall-clock gaps exist every weekend. Each session runs 9:30–17:00.
    """
    timestamps = []
    day_cursor = pd.Timestamp(start_date)
    days_added = 0
    while days_added < n_days:
        if day_cursor.weekday() < 5:   # Mon-Fri
            session_start = day_cursor + pd.Timedelta(hours=9, minutes=30)
            timestamps.extend(
                session_start + pd.Timedelta(minutes=i) for i in range(minutes_per_day)
            )
            days_added += 1
        day_cursor += pd.Timedelta(days=1)
    ts = pd.Series(timestamps)
    X = pd.DataFrame({"feat_a": np.random.default_rng(0).normal(size=len(ts))})
    return X, ts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_fold_straddles_calendar_day():
    X, ts = _build_multiday_series(n_days=30)
    cv = CascadeAwareCPCV(n_splits=5, n_test_groups=2, trading_day_aware=True)
    folds = cv.split(X, ts)
    assert len(folds) > 0, "expected at least one fold"
    for f in folds:
        train_days = ts.iloc[f.train_idx].dt.normalize().unique()
        test_days  = ts.iloc[f.test_idx].dt.normalize().unique()
        overlap = set(train_days) & set(test_days)
        assert not overlap, (
            f"fold {f.fold_id} has overlapping days in train+test: {overlap}"
        )


def test_fallback_non_day_aware_still_splits():
    """Old behaviour: trading_day_aware=False should split by raw index."""
    X, ts = _build_multiday_series(n_days=30)
    cv = CascadeAwareCPCV(n_splits=5, n_test_groups=2, trading_day_aware=False)
    folds = cv.split(X, ts)
    assert len(folds) == 10   # C(5,2)
    # Non-day-aware can (and typically does) split a day across folds — that's
    # expected behaviour. Just assert the shape is right.
    for f in folds:
        assert len(f.train_idx) + len(f.test_idx) <= len(X)


def test_min_train_days_guard_drops_bad_folds():
    """If we require >100 train days but only have 30, every fold is dropped."""
    X, ts = _build_multiday_series(n_days=30)
    cv = CascadeAwareCPCV(
        n_splits=5, n_test_groups=2,
        trading_day_aware=True,
        min_train_days=100, min_test_days=1,
    )
    folds = cv.split(X, ts)
    assert folds == [], "guard should drop every fold at impossible thresholds"


def test_calendar_day_counts_per_fold():
    """30 days / 5 splits / 2 test groups → 12 test days, 18 train days per fold."""
    X, ts = _build_multiday_series(n_days=30)
    cv = CascadeAwareCPCV(n_splits=5, n_test_groups=2, trading_day_aware=True)
    folds = cv.split(X, ts)
    for f in folds:
        tr_days = ts.iloc[f.train_idx].dt.normalize().nunique()
        te_days = ts.iloc[f.test_idx].dt.normalize().nunique()
        assert te_days == 12, f"fold {f.fold_id}: test days={te_days}"
        assert tr_days == 18, f"fold {f.fold_id}: train days={tr_days}"


def test_trading_day_embargo_ignores_weekends():
    """Embargo counts in trading-day ordinals, so weekends don't consume budget.

    Construct: 10 Mon-Fri trading days. Cascade on day 5.
    With ``cascade_embargo_days=2``, days 3-7 (inclusive) are embargoed.
    That's 5 days of embargo, regardless of how many weekends fell between
    day 1 and day 10.
    """
    X, ts = _build_multiday_series(n_days=10)
    cv = CascadeAwareCPCV(
        n_splits=5, n_test_groups=1,
        trading_day_aware=True,
        cascade_embargo_days=2,
        min_train_days=1, min_test_days=1,
    )
    distinct_days = sorted(ts.dt.normalize().unique())
    cascade_day = distinct_days[5]
    folds = cv.split(X, ts,
                     cascade_timestamps=pd.Series([cascade_day + pd.Timedelta(hours=10)]))

    # Union of all embargoed rows across folds
    embargoed_rows = np.concatenate([f.embargoed_idx for f in folds]) if folds else np.array([])
    if len(embargoed_rows) > 0:
        embargoed_days = sorted(ts.iloc[embargoed_rows].dt.normalize().unique())
        # Days 3,4,5,6,7 (0-indexed) — exactly 5 days
        expected_range = set(distinct_days[3:8])
        actual = set(embargoed_days)
        assert actual == expected_range, (
            f"trading-day embargo wrong set. expected {expected_range}, got {actual}"
        )


def test_wall_clock_embargo_still_works():
    """When trading_day_aware=False, embargo is wall-clock ±N days as before."""
    X, ts = _build_multiday_series(n_days=30)
    cv = CascadeAwareCPCV(
        n_splits=5, n_test_groups=2,
        trading_day_aware=False,
        cascade_embargo_days=3,
    )
    cascade = pd.Series([ts.iloc[len(ts) // 2]])
    folds = cv.split(X, ts, cascade_timestamps=cascade)
    assert len(folds) == 10
    # At least some folds should have non-empty embargo
    assert any(len(f.embargoed_idx) > 0 for f in folds)


def test_insufficient_days_raises():
    X, ts = _build_multiday_series(n_days=3)
    cv = CascadeAwareCPCV(n_splits=5, n_test_groups=2, trading_day_aware=True)
    with pytest.raises(ValueError, match="distinct calendar days"):
        cv.split(X, ts)
