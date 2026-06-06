"""Wave 8.5-F — CPCV-based out-of-fold evaluation for MetaLabeler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

from p6lab.validation.meta_labeler import MetaLabelerConfig, build_meta_features
from p6lab.validation.meta_labeler_cpcv import CPCVMetaReport, evaluate_cpcv


def _make_synthetic_cpcv_dataset(
    n: int = 800, tier_a_precision: float = 0.45, seed: int = 11,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    """Synthetic dataset with real feature signal so the secondary can
    learn. Timestamps span 160 days to give the CPCV splitter room."""
    rng = np.random.default_rng(seed)

    # ~50% tier-A rows, ~50% non-tier-A
    primary_proba = np.concatenate([
        rng.uniform(0.0, 0.85, size=n // 2),
        rng.uniform(0.85, 1.0, size=n - n // 2),
    ])
    rng.shuffle(primary_proba)

    # Label: tier-A subset has precision ~tier_a_precision
    y_true = np.zeros(n, dtype=int)
    tier_a_idx = np.where(primary_proba >= 0.85)[0]
    hit_count = int(len(tier_a_idx) * tier_a_precision)
    hits = rng.choice(tier_a_idx, size=hit_count, replace=False)
    y_true[hits] = 1

    # Features correlated with y so the secondary has signal
    fi_fast = np.where(y_true == 1, rng.normal(0.2, 0.05, n), rng.normal(0.6, 0.05, n))
    spread_bps = np.where(y_true == 1, rng.normal(1.0, 0.2, n), rng.normal(2.5, 0.3, n))
    imbalance_ema = rng.uniform(-0.3, 0.3, n)
    pnl_streak = rng.uniform(-1.0, 1.0, n)

    X = build_meta_features(
        primary_proba=primary_proba, fi_fast=fi_fast,
        imbalance_ema=imbalance_ema, spread_bps=spread_bps,
        recent_pnl_streak=pnl_streak,
    )
    # Spread timestamps across 160 days so CPCV has ample partitions
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=n, freq="4h"))
    return X, primary_proba, y_true, timestamps


def test_wave_85_f_evaluate_cpcv_returns_report() -> None:
    X, proba, y, ts = _make_synthetic_cpcv_dataset(n=400)
    report = evaluate_cpcv(
        X, primary_proba=proba, y_true=y, timestamps=ts,
        n_splits=3, n_test_groups=1,
        config=MetaLabelerConfig(min_train_samples=20),
    )
    assert isinstance(report, CPCVMetaReport)
    assert report.folds_run > 0
    assert len(report.per_fold) == report.folds_run


def test_wave_85_f_evaluate_cpcv_aggregated_has_tier_a_rows() -> None:
    """On a set with clear tier-A rows, the aggregated report should
    carry non-zero tier_a_n_before."""
    X, proba, y, ts = _make_synthetic_cpcv_dataset(n=400)
    report = evaluate_cpcv(
        X, primary_proba=proba, y_true=y, timestamps=ts,
        n_splits=3, n_test_groups=1,
        config=MetaLabelerConfig(min_train_samples=20),
    )
    assert report.aggregated.tier_a_n_before > 0


def test_wave_85_f_evaluate_cpcv_empty_input() -> None:
    empty = build_meta_features(
        primary_proba=np.asarray([]),
        fi_fast=np.asarray([]),
        imbalance_ema=np.asarray([]),
        spread_bps=np.asarray([]),
        recent_pnl_streak=np.asarray([]),
    )
    report = evaluate_cpcv(
        empty, primary_proba=np.asarray([]), y_true=np.asarray([]),
        timestamps=pd.Series([], dtype="datetime64[ns]"),
    )
    assert report.folds_run == 0
    assert report.aggregated.tier_a_n_before == 0


def test_wave_85_f_evaluate_cpcv_insufficient_samples_skip() -> None:
    """Small n + high min_train_samples → folds skipped, returned as 0."""
    X, proba, y, ts = _make_synthetic_cpcv_dataset(n=50)
    report = evaluate_cpcv(
        X, primary_proba=proba, y_true=y, timestamps=ts,
        n_splits=3, n_test_groups=1,
        config=MetaLabelerConfig(min_train_samples=10_000),
    )
    assert report.folds_run == 0


def test_wave_85_f_oof_predictions_are_boolean() -> None:
    X, proba, y, ts = _make_synthetic_cpcv_dataset(n=400)
    report = evaluate_cpcv(
        X, primary_proba=proba, y_true=y, timestamps=ts,
        n_splits=3, n_test_groups=1,
        config=MetaLabelerConfig(min_train_samples=20),
    )
    if report.folds_run > 0:
        assert report.oof_predictions.dtype == bool


def test_wave_85_f_per_fold_reports_same_length_as_folds_run() -> None:
    X, proba, y, ts = _make_synthetic_cpcv_dataset(n=400)
    report = evaluate_cpcv(
        X, primary_proba=proba, y_true=y, timestamps=ts,
        n_splits=3, n_test_groups=1,
        config=MetaLabelerConfig(min_train_samples=20),
    )
    assert len(report.per_fold) == report.folds_run


def test_wave_85_f_to_dict_serialization() -> None:
    X, proba, y, ts = _make_synthetic_cpcv_dataset(n=400)
    report = evaluate_cpcv(
        X, primary_proba=proba, y_true=y, timestamps=ts,
        n_splits=3, n_test_groups=1,
        config=MetaLabelerConfig(min_train_samples=20),
    )
    d = report.to_dict()
    assert "aggregated" in d
    assert "per_fold" in d
    assert d["folds_run"] == report.folds_run
