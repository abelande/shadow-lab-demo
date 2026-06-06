"""
p6lab.validation.cpcv
=====================
Combinatorial Purged Cross-Validation with cascade embargo — §8.1.

Base method
-----------
Wraps existing ``specs/cv/cv_combinatorial.py`` logic with one additional
constraint: a 14-day embargo around cascade timestamps to prevent leakage.

Embargo rule (spec §8.1, OB-reference §L1804-L1905)
----------------------------------------------------
For each cascade event timestamp Tc, remove samples in:
    [Tc - 14 days, Tc + 14 days]
from any training fold when the corresponding test fold contains events
in that vicinity.

Used by notebooks:
- Notebook 07 (required)
- Notebook 03/04/06 (available)
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CASCADE_EMBARGO_DAYS: int = 14
DEFAULT_N_SPLITS: int = 5
DEFAULT_N_TEST_GROUPS: int = 2


@dataclass
class CPCVFold:
    """One CPCV fold definition.

    Attributes
    ----------
    fold_id:
        Sequential fold id.
    train_idx:
        Numpy integer indices for training rows.
    test_idx:
        Numpy integer indices for test rows.
    embargoed_idx:
        Numpy indices dropped from training due to cascade embargo.
    """

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    embargoed_idx: np.ndarray


class CascadeAwareCPCV:
    """CPCV splitter with additional cascade embargo.

    Parameters
    ----------
    n_splits:
        Number of temporal groups.
    n_test_groups:
        Number of groups used as test in each combinatorial fold.
    cascade_embargo_days:
        Embargo window around cascade timestamps.
    trading_day_aware:
        When True, partition splits on **calendar-day boundaries** instead
        of raw-index argsort buckets. This avoids splitting a single
        trading day across train/test folds — a silent leakage vector
        when features on the same day are correlated.
    min_train_days:
        Minimum distinct calendar days that must remain in train after
        embargo. Folds failing this guard are *dropped* (not degraded).
        Ignored when ``trading_day_aware=False``.
    min_test_days:
        Same, for test. Ignored when ``trading_day_aware=False``.
    """

    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        n_test_groups: int = DEFAULT_N_TEST_GROUPS,
        cascade_embargo_days: int = CASCADE_EMBARGO_DAYS,
        *,
        trading_day_aware: bool = False,
        min_train_days: int = 3,
        min_test_days: int = 1,
    ) -> None:
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.cascade_embargo_days = cascade_embargo_days
        self.trading_day_aware = trading_day_aware
        self.min_train_days = min_train_days
        self.min_test_days = min_test_days

    def split(
        self,
        X: pd.DataFrame,
        timestamps: pd.Series,
        cascade_timestamps: pd.Series | None = None,
    ) -> list[CPCVFold]:
        """Generate CPCV folds with purging + cascade embargo.

        Process:
          1. Sort indices by timestamp, partition into ``n_splits`` groups.
             When ``trading_day_aware``, groups respect calendar-day
             boundaries (no day straddles two groups).
          2. Enumerate all ``C(n_splits, n_test_groups)`` test combinations.
          3. Test = union of test groups; train = complement.
          4. Apply ±cascade_embargo_days purge to train when cascade
             timestamps are provided.
          5. (multi-day only) drop folds whose train/test day counts fall
             below ``min_train_days`` / ``min_test_days``.
        """
        n = len(X)
        if n == 0:
            return []
        ts = pd.to_datetime(timestamps).reset_index(drop=True)
        if self.trading_day_aware:
            groups = self._day_aware_groups(ts)
        else:
            order = np.argsort(ts.values)
            groups = np.array_split(order, self.n_splits)

        folds: list[CPCVFold] = []
        fold_id_counter = 0
        for combo in itertools.combinations(range(len(groups)), self.n_test_groups):
            test_idx = np.concatenate([groups[i] for i in combo]) if combo else np.array([], dtype=int)
            if len(test_idx) == 0:
                continue
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.where(train_mask)[0]
            embargoed: np.ndarray = np.array([], dtype=int)
            if cascade_timestamps is not None and len(cascade_timestamps) > 0:
                train_idx, embargoed = self.apply_cascade_embargo(
                    train_idx, ts, pd.to_datetime(cascade_timestamps),
                )

            # Day-count guard (multi-day only)
            if self.trading_day_aware:
                train_days = self._count_distinct_days(ts, train_idx)
                test_days  = self._count_distinct_days(ts, test_idx)
                if train_days < self.min_train_days or test_days < self.min_test_days:
                    logger.info(
                        "CPCV fold skipped: train_days=%d (<%d) / test_days=%d (<%d)",
                        train_days, self.min_train_days, test_days, self.min_test_days,
                    )
                    continue

            folds.append(CPCVFold(
                fold_id=fold_id_counter,
                train_idx=np.sort(train_idx),
                test_idx=np.sort(test_idx),
                embargoed_idx=np.sort(embargoed),
            ))
            fold_id_counter += 1
        return folds

    # ------------------------------------------------------------------
    # Calendar-day-aware helpers
    # ------------------------------------------------------------------

    def _day_aware_groups(self, ts: pd.Series) -> list[np.ndarray]:
        """Partition row indices into ``n_splits`` groups of roughly equal day count.

        Each group contains a contiguous block of distinct calendar days.
        No day appears in more than one group, so train/test fold boundaries
        always fall on a midnight.
        """
        ts_sorted = ts.sort_values(kind="mergesort")
        sorted_positions = ts_sorted.index.to_numpy()
        days = ts_sorted.dt.normalize()          # drop time, keep date
        unique_days = days.drop_duplicates().values
        if len(unique_days) < self.n_splits:
            raise ValueError(
                f"trading_day_aware=True requires ≥ n_splits ({self.n_splits}) "
                f"distinct calendar days; got {len(unique_days)}"
            )
        day_buckets = np.array_split(unique_days, self.n_splits)
        day_to_bucket = {
            pd.Timestamp(d): i for i, bucket in enumerate(day_buckets) for d in bucket
        }
        # Walk sorted positions, assign to the correct bucket
        groups: list[list[int]] = [[] for _ in range(self.n_splits)]
        for pos, day in zip(sorted_positions, days.values):
            groups[day_to_bucket[pd.Timestamp(day)]].append(int(pos))
        return [np.asarray(g, dtype=int) for g in groups]

    @staticmethod
    def _count_distinct_days(ts: pd.Series, idx: np.ndarray) -> int:
        if len(idx) == 0:
            return 0
        return int(ts.iloc[idx].dt.normalize().nunique())

    def apply_cascade_embargo(
        self,
        train_idx: np.ndarray,
        timestamps: pd.Series,
        cascade_timestamps: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply ±embargo around each cascade timestamp.

        In ``trading_day_aware`` mode the embargo is measured in
        **trading-day count** (weekends / non-trading days excluded from
        the budget) by mapping the timestamp series to its sorted set of
        distinct calendar days and counting positions in that array.
        Otherwise falls back to the wall-clock ±N-day window.
        """
        if len(cascade_timestamps) == 0 or len(train_idx) == 0:
            return train_idx, np.array([], dtype=int)
        cascade_arr = pd.to_datetime(cascade_timestamps).values

        if not self.trading_day_aware:
            delta = pd.Timedelta(days=self.cascade_embargo_days)
            train_ts = pd.to_datetime(timestamps.iloc[train_idx]).values
            keep_mask = np.ones(len(train_idx), dtype=bool)
            for cts in cascade_arr:
                lo = cts - delta
                hi = cts + delta
                within = (train_ts >= lo) & (train_ts <= hi)
                keep_mask &= ~within
            return train_idx[keep_mask], train_idx[~keep_mask]

        # Trading-day mode: build a day-ordinal lookup and purge by ordinal distance.
        days_series = pd.to_datetime(timestamps).dt.normalize()
        unique_days = np.array(sorted(days_series.drop_duplicates().values))
        day_to_ord = {pd.Timestamp(d): i for i, d in enumerate(unique_days)}
        train_ords = np.array([
            day_to_ord[pd.Timestamp(days_series.iloc[i])] for i in train_idx
        ])
        keep_mask = np.ones(len(train_idx), dtype=bool)
        for cts in cascade_arr:
            c_day = pd.Timestamp(pd.Timestamp(cts).normalize())
            # Snap to nearest known trading day if the cascade lands on a non-trading day
            if c_day not in day_to_ord:
                idx = int(np.searchsorted(unique_days, np.datetime64(c_day)))
                if idx >= len(unique_days):
                    continue
                c_day = pd.Timestamp(unique_days[idx])
            c_ord = day_to_ord[c_day]
            within = np.abs(train_ords - c_ord) <= self.cascade_embargo_days
            keep_mask &= ~within
        return train_idx[keep_mask], train_idx[~keep_mask]
