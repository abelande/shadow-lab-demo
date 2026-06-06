"""Tests for MetaLabeler (Wave 5 Phase 5C)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from p6lab.validation.meta_labeler import (
    META_FEATURE_COLS,
    MetaLabeler,
    MetaLabelerConfig,
    MetaLabelerReport,
    build_meta_features,
    compute_recent_pnl_streak,
)


# ---------------------------------------------------------------------------
# compute_recent_pnl_streak
# ---------------------------------------------------------------------------


@dataclass
class _Hit:
    hit: bool


def test_streak_empty_returns_zero() -> None:
    assert compute_recent_pnl_streak([], window=10) == 0.0


def test_streak_all_hits_returns_plus_one() -> None:
    assert compute_recent_pnl_streak([_Hit(True)] * 10, window=10) == pytest.approx(1.0)


def test_streak_all_misses_returns_minus_one() -> None:
    assert compute_recent_pnl_streak([_Hit(False)] * 10, window=10) == pytest.approx(-1.0)


def test_streak_mixed_is_net_hit_rate() -> None:
    # 7 hits, 3 misses → (7-3)/10 = 0.4
    seq = [_Hit(True)] * 7 + [_Hit(False)] * 3
    assert compute_recent_pnl_streak(seq, window=10) == pytest.approx(0.4)


def test_streak_window_trims_history() -> None:
    # 100 misses then 5 hits with window=5 → all hits
    seq = [_Hit(False)] * 100 + [_Hit(True)] * 5
    assert compute_recent_pnl_streak(seq, window=5) == pytest.approx(1.0)


def test_streak_handles_missing_attribute_as_miss() -> None:
    class _Bare:
        pass
    # objects without `.hit` → treated as miss
    assert compute_recent_pnl_streak([_Bare(), _Bare()], window=2) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# build_meta_features
# ---------------------------------------------------------------------------


def test_build_features_scalar_streak_broadcasts() -> None:
    df = build_meta_features(
        primary_proba=np.array([0.2, 0.8, 0.9]),
        fi_fast=np.array([0.1, 0.2, 0.3]),
        imbalance_ema=np.array([-0.1, 0.0, 0.2]),
        spread_bps=np.array([1.0, 1.2, 0.8]),
        recent_pnl_streak=0.3,
    )
    assert list(df.columns) == list(META_FEATURE_COLS)
    assert (df["recent_pnl_streak"] == 0.3).all()


def test_build_features_series_streak_passes_through() -> None:
    df = build_meta_features(
        primary_proba=np.array([0.2, 0.8, 0.9]),
        fi_fast=np.array([0.1, 0.2, 0.3]),
        imbalance_ema=np.array([-0.1, 0.0, 0.2]),
        spread_bps=np.array([1.0, 1.2, 0.8]),
        recent_pnl_streak=np.array([0.1, 0.2, 0.3]),
    )
    np.testing.assert_array_almost_equal(df["recent_pnl_streak"].to_numpy(), [0.1, 0.2, 0.3])


def test_build_features_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        build_meta_features(
            primary_proba=np.array([0.2, 0.8]),
            fi_fast=np.array([0.1, 0.2]),
            imbalance_ema=np.array([-0.1, 0.0]),
            spread_bps=np.array([1.0, 1.2]),
            recent_pnl_streak=np.array([0.1]),
        )


# ---------------------------------------------------------------------------
# MetaLabeler end-to-end
# ---------------------------------------------------------------------------


def _make_synthetic(
    n: int = 400, tier_a_precision: float = 0.45, seed: int = 7
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Construct a synthetic dataset where:
      - ~half the rows are tier-A (primary_proba >= 0.85)
      - the tier-A subset is noisy (precision defaults to 0.45)
      - fi_fast + spread_bps carry a real signal distinguishing hits/misses
        so the meta-labeler has something to learn.
    """
    rng = np.random.default_rng(seed)

    primary_proba = np.concatenate([
        rng.uniform(0.0, 0.85, size=n // 2),
        rng.uniform(0.85, 1.0, size=n - n // 2),
    ])
    rng.shuffle(primary_proba)

    y_true = np.zeros(n, dtype=int)
    tier_a_idx = np.where(primary_proba >= 0.85)[0]
    hit_count = int(len(tier_a_idx) * tier_a_precision)
    hits = rng.choice(tier_a_idx, size=hit_count, replace=False)
    y_true[hits] = 1

    # fi_fast + spread_bps: correlated with hit/miss so the secondary has signal
    fi_fast = np.where(y_true == 1, rng.normal(0.2, 0.05, n), rng.normal(0.6, 0.05, n))
    spread_bps = np.where(y_true == 1, rng.normal(1.0, 0.2, n), rng.normal(2.5, 0.3, n))
    imbalance_ema = rng.uniform(-0.3, 0.3, n)
    pnl_streak = rng.uniform(-1.0, 1.0, n)

    X = build_meta_features(
        primary_proba=primary_proba,
        fi_fast=fi_fast,
        imbalance_ema=imbalance_ema,
        spread_bps=spread_bps,
        recent_pnl_streak=pnl_streak,
    )
    return X, primary_proba, y_true


def test_meta_labeler_fits_on_sufficient_data() -> None:
    X, proba, y = _make_synthetic(n=400)
    meta = MetaLabeler(MetaLabelerConfig(min_train_samples=20))
    meta.fit(X, primary_proba=proba, y_true=y)
    assert meta.is_fitted_
    assert meta.train_samples_ >= 20
    assert meta.model_ is not None


def test_meta_labeler_skips_fit_when_below_min_samples() -> None:
    X, proba, y = _make_synthetic(n=20)
    meta = MetaLabeler(MetaLabelerConfig(min_train_samples=100))
    meta.fit(X, primary_proba=proba, y_true=y)
    assert not meta.is_fitted_


def test_meta_labeler_skips_fit_when_single_class() -> None:
    """All tier-A rows share the same label → nothing to learn."""
    n = 200
    rng = np.random.default_rng(0)
    proba = rng.uniform(0.85, 1.0, n)   # all tier-A
    y = np.ones(n, dtype=int)           # all positives
    X = build_meta_features(
        primary_proba=proba,
        fi_fast=rng.uniform(0, 1, n),
        imbalance_ema=rng.uniform(-0.5, 0.5, n),
        spread_bps=rng.uniform(0, 3, n),
        recent_pnl_streak=rng.uniform(-1, 1, n),
    )
    meta = MetaLabeler(MetaLabelerConfig(min_train_samples=50))
    meta.fit(X, primary_proba=proba, y_true=y)
    assert not meta.is_fitted_


def test_meta_labeler_predict_take_bet_shape() -> None:
    X, proba, y = _make_synthetic(n=400)
    meta = MetaLabeler().fit(X, primary_proba=proba, y_true=y)
    decisions = meta.predict_take_bet(X, primary_proba=proba)
    assert decisions.shape == (400,)
    assert decisions.dtype == bool


def test_meta_labeler_take_bet_requires_tier_a() -> None:
    """Primary proba below tier-A cutoff must always map to False."""
    X, proba, y = _make_synthetic(n=400)
    meta = MetaLabeler().fit(X, primary_proba=proba, y_true=y)
    decisions = meta.predict_take_bet(X, primary_proba=proba)
    # Rows below tier-A threshold must be gated out entirely
    below = proba < meta.config.tier_a_threshold
    assert not decisions[below].any()


def test_meta_labeler_unfitted_falls_back_to_tier_a_gate() -> None:
    X, proba, y = _make_synthetic(n=20)
    meta = MetaLabeler(MetaLabelerConfig(min_train_samples=1000))
    meta.fit(X, primary_proba=proba, y_true=y)
    assert not meta.is_fitted_
    decisions = meta.predict_take_bet(X, primary_proba=proba)
    # Equivalent to tier-A-only gate
    np.testing.assert_array_equal(decisions, proba >= meta.config.tier_a_threshold)


def test_meta_labeler_evaluate_reduces_fp_rate() -> None:
    """On a tier-A-precision-0.45 synthetic set with real feature signal,
    the secondary should reduce FP count."""
    X, proba, y = _make_synthetic(n=600, tier_a_precision=0.45)
    meta = MetaLabeler(
        MetaLabelerConfig(min_train_samples=20, take_bet_threshold=0.5)
    ).fit(X, primary_proba=proba, y_true=y)
    report = meta.evaluate(X, primary_proba=proba, y_true=y)

    assert isinstance(report, MetaLabelerReport)
    assert report.tier_a_n_before > 0
    assert report.tier_a_fp_before > 0
    assert report.tier_a_fp_after <= report.tier_a_fp_before
    # On IN-sample data with real signal we should see meaningful reduction
    assert report.fp_reduction_pct >= 0.2


def test_meta_labeler_evaluate_report_keys() -> None:
    X, proba, y = _make_synthetic(n=400)
    meta = MetaLabeler().fit(X, primary_proba=proba, y_true=y)
    report = meta.evaluate(X, primary_proba=proba, y_true=y).to_dict()
    for k in (
        "tier_a_n_before", "tier_a_n_after",
        "tier_a_precision_before", "tier_a_precision_after",
        "tier_a_recall_before", "tier_a_recall_after",
        "tier_a_fp_before", "tier_a_fp_after",
        "fp_reduction_pct", "take_bet_threshold",
    ):
        assert k in report, f"missing key {k}"


def test_meta_labeler_evaluate_handles_empty_tier_a() -> None:
    rng = np.random.default_rng(0)
    n = 100
    proba = rng.uniform(0.0, 0.5, n)  # no tier-A rows
    y = rng.integers(0, 2, n)
    X = build_meta_features(
        primary_proba=proba,
        fi_fast=rng.uniform(0, 1, n),
        imbalance_ema=rng.uniform(-0.5, 0.5, n),
        spread_bps=rng.uniform(0, 3, n),
        recent_pnl_streak=rng.uniform(-1, 1, n),
    )
    meta = MetaLabeler()
    report = meta.evaluate(X, primary_proba=proba, y_true=y)
    assert report.tier_a_n_before == 0
    assert report.tier_a_fp_before == 0
    assert report.fp_reduction_pct == 0.0


def test_meta_labeler_missing_column_raises() -> None:
    X = pd.DataFrame({
        "primary_proba": [0.1, 0.2],
        "fi_fast": [0.0, 0.1],
    })
    meta = MetaLabeler()
    with pytest.raises(ValueError, match="missing columns"):
        meta.fit(X, primary_proba=np.array([0.1, 0.2]), y_true=np.array([0, 1]))


def test_meta_labeler_length_mismatch_raises() -> None:
    X, proba, y = _make_synthetic(n=400)
    meta = MetaLabeler()
    with pytest.raises(ValueError, match="share length"):
        meta.fit(X, primary_proba=proba[:100], y_true=y)


def test_meta_labeler_predict_proba_unfitted_returns_zeros() -> None:
    rng = np.random.default_rng(0)
    X = build_meta_features(
        primary_proba=rng.uniform(0.0, 1.0, 10),
        fi_fast=rng.uniform(0, 1, 10),
        imbalance_ema=rng.uniform(-0.5, 0.5, 10),
        spread_bps=rng.uniform(0, 3, 10),
        recent_pnl_streak=0.0,
    )
    meta = MetaLabeler()
    np.testing.assert_array_equal(meta.predict_proba(X), np.zeros(10))


def test_meta_labeler_higher_threshold_prunes_more() -> None:
    """Stricter take_bet_threshold → fewer take_bets → fewer FPs."""
    X, proba, y = _make_synthetic(n=600, seed=11)
    strict = MetaLabeler(
        MetaLabelerConfig(take_bet_threshold=0.7, min_train_samples=20)
    ).fit(X, primary_proba=proba, y_true=y)
    lax = MetaLabeler(
        MetaLabelerConfig(take_bet_threshold=0.3, min_train_samples=20)
    ).fit(X, primary_proba=proba, y_true=y)

    strict_take = strict.predict_take_bet(X, primary_proba=proba).sum()
    lax_take = lax.predict_take_bet(X, primary_proba=proba).sum()
    assert strict_take <= lax_take
