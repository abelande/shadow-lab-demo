"""Wave 9+10 §04 multi-spec training — tests for train_multi_spec_cpcv."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p6lab.validation.cpcv import CascadeAwareCPCV
from p6lab.validation.multi_spec_cv import (
    SpecResult,
    aggregate_oof,
    calibration_summary,
    calibration_table,
    calibration_verdict,
    isotonic_brier_delta,
    summarize_results,
    train_multi_spec_cpcv,
)


def _build_synthetic(
    n: int = 200,
    n_features: int = 6,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Synthetic X + multi_labels + CPCV folds for testing.

    multi_labels has four columns:
      - tb_a (binary {-1, +1}, balanced, with weak X[:,0]→y signal)
      - mm_a (5-class {-2..+2}, with weak X[:,0]→y signal)
      - pf_a (binary {0, 1}, sparse 5% positives)
      - degenerate_zero (all zeros — should be skipped)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, n_features))
    # Drive each label by X[:, 0] sign so the model has *some* signal
    z = X[:, 0]
    tb = np.where(z + rng.normal(0, 0.5, n) > 0, 1, -1).astype(np.int8)
    mm_continuous = z + rng.normal(0, 0.3, n)
    mm = np.zeros(n, dtype=np.int8)
    mm[mm_continuous > 1.0] = 2
    mm[(mm_continuous > 0.3) & (mm_continuous <= 1.0)] = 1
    mm[(mm_continuous < -1.0)] = -2
    mm[(mm_continuous < -0.3) & (mm_continuous >= -1.0)] = -1
    pf = (rng.random(n) < 0.05).astype(np.int8)
    deg = np.zeros(n, dtype=np.int8)

    multi_labels = pd.DataFrame({
        "tb_a": tb,
        "mm_a": mm,
        "pf_a": pf,
        "degenerate_zero": deg,
    })
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])

    # CPCV folds
    ts = pd.Series(pd.date_range("2025-01-01", periods=n, freq="100ms"))
    cv = CascadeAwareCPCV(n_splits=4, n_test_groups=2, cascade_embargo_days=0)
    folds = list(cv.split(X_df, ts, None))
    return X_df, multi_labels, folds


class TestTrainMultiSpecCpcv:
    def test_returns_dict_with_results(self) -> None:
        X_df, multi_labels, folds = _build_synthetic(n=200)
        results = train_multi_spec_cpcv(
            X_df, multi_labels, folds, boost_rounds=20,
        )
        assert isinstance(results, dict)
        assert all(isinstance(v, SpecResult) for v in results.values())
        # Non-degenerate specs should be present; degenerate should be skipped.
        assert "tb_a" in results
        assert "mm_a" in results
        assert "degenerate_zero" not in results

    def test_binary_spec_records_correct_n_classes(self) -> None:
        X_df, multi_labels, folds = _build_synthetic(n=200)
        results = train_multi_spec_cpcv(
            X_df, multi_labels, folds, boost_rounds=20,
        )
        assert results["tb_a"].n_classes == 2
        # tb_a was constructed with weak signal → AUC > 0.5
        assert results["tb_a"].mean_auc > 0.5

    def test_multiclass_spec_handles_5_class_label(self) -> None:
        X_df, multi_labels, folds = _build_synthetic(n=400)
        results = train_multi_spec_cpcv(
            X_df, multi_labels, folds, boost_rounds=20,
        )
        # mm_a spans up to 5 classes
        assert results["mm_a"].n_classes >= 3
        # Multi-class fold_preds should have proba shape (n_test, n_classes)
        for y_te, proba, te_idx in results["mm_a"].fold_preds:
            assert proba.ndim == 2
            assert proba.shape[1] >= 3
            assert proba.shape[0] == len(y_te)

    def test_class_dist_recorded(self) -> None:
        X_df, multi_labels, folds = _build_synthetic(n=200)
        results = train_multi_spec_cpcv(
            X_df, multi_labels, folds, boost_rounds=20,
        )
        for col, res in results.items():
            assert sum(res.class_dist.values()) == len(multi_labels)
            assert all(isinstance(k, int) for k in res.class_dist)

    def test_purge_callback_invoked(self) -> None:
        """Verify apply_row_purge is called with expected signatures."""
        X_df, multi_labels, folds = _build_synthetic(n=100)
        calls = []

        def _purge(tr, te, p):
            calls.append((len(tr), len(te), p))
            return tr  # no-op pass-through

        results = train_multi_spec_cpcv(
            X_df, multi_labels[["tb_a"]], folds,
            purge_rows=5, apply_row_purge=_purge, boost_rounds=10,
        )
        assert len(calls) >= 1
        for tr_n, te_n, p in calls:
            assert p == 5
            assert tr_n > 0
            assert te_n > 0

    def test_length_mismatch_raises(self) -> None:
        X_df = pd.DataFrame(np.zeros((10, 3)))
        multi = pd.DataFrame({"a": np.zeros(20, dtype=np.int8)})
        with pytest.raises(ValueError, match="rows"):
            train_multi_spec_cpcv(X_df, multi, folds=[], boost_rounds=10)

    def test_empty_specs_returns_empty(self) -> None:
        X_df, _, folds = _build_synthetic(n=100)
        empty = pd.DataFrame(index=X_df.index)
        results = train_multi_spec_cpcv(X_df, empty, folds, boost_rounds=10)
        assert results == {}

    def test_fold_preds_alignment(self) -> None:
        """fold_preds[i] = (y_test, proba, test_idx) — lengths must match."""
        X_df, multi_labels, folds = _build_synthetic(n=200)
        results = train_multi_spec_cpcv(
            X_df, multi_labels, folds, boost_rounds=20,
        )
        for col, res in results.items():
            for y_te, proba, te_idx in res.fold_preds:
                assert len(y_te) == len(te_idx)
                if proba.ndim == 1:
                    assert len(proba) == len(y_te)
                else:
                    assert proba.shape[0] == len(y_te)

    def test_per_spec_mode_filters_each_label_independently(self) -> None:
        """valid_masks + folds_factory let each spec drop its own
        unobservable rows. Resolves the §04-multi horizon-collapse where
        a single global mask clipped longer-horizon specs."""
        X_df, multi_labels, _ = _build_synthetic(n=300)

        # Inject NaN at different positions per spec so per-spec filters
        # genuinely differ. (Coerce to float64 so NaN can live in the cell.)
        labels_with_nan = multi_labels.astype(np.float64)
        labels_with_nan.iloc[:50, labels_with_nan.columns.get_loc("tb_a")] = np.nan
        labels_with_nan.iloc[-30:, labels_with_nan.columns.get_loc("mm_a")] = np.nan

        valid_masks = {
            col: ~labels_with_nan[col].isna().to_numpy()
            for col in labels_with_nan.columns
        }

        def _folds_factory(X_filtered: pd.DataFrame) -> list:
            cv = CascadeAwareCPCV(
                n_splits=4, n_test_groups=2, cascade_embargo_days=0,
            )
            ts = pd.Series(pd.date_range(
                "2025-01-01", periods=len(X_filtered), freq="100ms",
            ))
            return list(cv.split(X_filtered, ts, None))

        results = train_multi_spec_cpcv(
            X_df, labels_with_nan,
            valid_masks=valid_masks,
            folds_factory=_folds_factory,
            boost_rounds=10,
        )

        # Both surviving specs should have run. Class distributions should
        # reflect the per-spec filter (n_rows = sum of class_dist values).
        assert "tb_a" in results
        assert "mm_a" in results
        assert sum(results["tb_a"].class_dist.values()) == 250  # 300 - 50 NaN
        assert sum(results["mm_a"].class_dist.values()) == 270  # 300 - 30 NaN

    def test_per_spec_mode_requires_folds_factory(self) -> None:
        X_df, multi_labels, _ = _build_synthetic(n=100)
        with pytest.raises(ValueError, match="folds_factory"):
            train_multi_spec_cpcv(
                X_df, multi_labels,
                valid_masks={"tb_a": np.ones(100, dtype=bool)},
                boost_rounds=10,
            )

    def test_legacy_mode_requires_folds(self) -> None:
        X_df, multi_labels, _ = _build_synthetic(n=100)
        with pytest.raises(ValueError, match="legacy mode"):
            train_multi_spec_cpcv(
                X_df, multi_labels, boost_rounds=10,
            )


class TestSummarizeResults:
    def test_returns_sorted_dataframe(self) -> None:
        X_df, multi_labels, folds = _build_synthetic(n=200)
        results = train_multi_spec_cpcv(
            X_df, multi_labels, folds, boost_rounds=20,
        )
        summary = summarize_results(results, sort_by_auc=True)
        assert isinstance(summary, pd.DataFrame)
        assert list(summary.columns) == [
            "spec", "auc", "std", "n_folds", "n_classes", "modal_class_pct",
        ]
        # Sorted descending by AUC
        if len(summary) >= 2:
            assert (summary["auc"].diff().dropna() <= 0).all()

    def test_empty_results_returns_empty_df(self) -> None:
        summary = summarize_results({})
        assert summary.shape == (0, 0) or len(summary) == 0


# ---------------------------------------------------------------------------
# Calibration helpers — Wave 9+10 §04d-multi / §04e-multi
# ---------------------------------------------------------------------------


def _stub_result(
    name: str,
    fold_preds: list,
    n_classes: int,
    auc: float = 0.55,
) -> SpecResult:
    """Synthetic SpecResult for calibration helper testing."""
    return SpecResult(
        name=name,
        mean_auc=auc,
        std_auc=0.02,
        n_folds=len(fold_preds),
        n_classes=n_classes,
        class_dist={0: 100, 1: 100},
        fold_preds=fold_preds,
    )


class TestAggregateOOF:
    def test_binary_aggregation(self) -> None:
        fp = [
            (np.array([0, 1, 0]), np.array([0.2, 0.7, 0.4]), np.array([0, 1, 2])),
            (np.array([1, 0]), np.array([0.8, 0.3]), np.array([3, 4])),
        ]
        result = _stub_result("bin", fp, n_classes=2)
        all_y, all_p = aggregate_oof(result)
        assert all_y.shape == (5,)
        assert all_p.shape == (5,)
        assert all_p.tolist() == [0.2, 0.7, 0.4, 0.8, 0.3]

    def test_multiclass_aggregation(self) -> None:
        fp = [
            (np.array([0, 1]), np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]),
             np.array([0, 1])),
            (np.array([2, 0]), np.array([[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]]),
             np.array([2, 3])),
        ]
        result = _stub_result("mc", fp, n_classes=3)
        all_y, all_p = aggregate_oof(result)
        assert all_y.shape == (4,)
        assert all_p.shape == (4, 3)

    def test_inconsistent_proba_widths_raises(self) -> None:
        fp = [
            (np.array([0]), np.array([[0.5, 0.5]]), np.array([0])),
            (np.array([0]), np.array([[0.4, 0.3, 0.3]]), np.array([1])),
        ]
        result = _stub_result("bad", fp, n_classes=3)
        with pytest.raises(ValueError, match="inconsistent proba widths"):
            aggregate_oof(result)

    def test_empty_fold_preds_returns_empty(self) -> None:
        result = _stub_result("empty", [], n_classes=2)
        all_y, all_p = aggregate_oof(result)
        assert len(all_y) == 0
        assert len(all_p) == 0


class TestCalibrationTable:
    def test_binary_decile_rows(self) -> None:
        rng = np.random.default_rng(0)
        n = 5_000
        proba = rng.uniform(0, 1, n)
        # Construct y so calibration is approximately monotonic — y=1 with prob = proba
        y = (rng.uniform(0, 1, n) < proba).astype(int)
        rows = calibration_table(y, proba, n_classes=2, deciles=10, min_bin_n=50)
        assert len(rows) > 0
        # Should be ~monotonic — hit rate increases with proba bin
        hit_rates = [r[2] for r in rows]
        assert hit_rates[-1] > hit_rates[0]

    def test_binary_wrong_proba_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="binary spec needs 1D"):
            calibration_table(
                np.zeros(5), np.zeros((5, 2)), n_classes=2,
            )

    def test_multiclass_decile_rows(self) -> None:
        rng = np.random.default_rng(1)
        n = 1_000
        # Three classes, top-class proba uniform [0.4, 1.0]
        proba = rng.uniform(0, 1, (n, 3))
        proba = proba / proba.sum(axis=1, keepdims=True)
        y = rng.integers(0, 3, n)
        rows = calibration_table(y, proba, n_classes=3, deciles=10, min_bin_n=10)
        # Should produce at least some rows and reliability should be finite
        assert len(rows) > 0
        for label, n_bin, hr, rel in rows:
            assert 0 <= hr <= 1
            assert np.isfinite(rel)

    def test_multiclass_wrong_proba_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="multi-class spec needs 2D"):
            calibration_table(
                np.zeros(5), np.zeros(5), n_classes=3,
            )

    def test_min_bin_n_filters_sparse_bins(self) -> None:
        proba = np.array([0.05, 0.15, 0.45, 0.55, 0.95])
        y = np.array([0, 0, 1, 1, 1])
        rows = calibration_table(y, proba, n_classes=2, deciles=10, min_bin_n=2)
        # Each bin only has 1 sample → all dropped
        assert rows == []


class TestIsotonicBrierDelta:
    def test_binary_perfectly_calibrated_no_change(self) -> None:
        # Already-calibrated probas: hit rate matches proba per bin
        rng = np.random.default_rng(0)
        n = 10_000
        proba = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < proba).astype(int)
        b_raw, b_cal = isotonic_brier_delta(y, proba, n_classes=2)
        # Both Brier values close (already-calibrated → small improvement)
        assert b_raw > 0
        assert b_cal > 0
        assert abs(b_raw - b_cal) < 0.02

    def test_binary_overconfident_calibration_helps(self) -> None:
        # Construct anti-calibrated probas: model says 0.9 but truth is 0.5
        n = 5_000
        rng = np.random.default_rng(0)
        proba = np.full(n, 0.9)
        y = (rng.uniform(0, 1, n) < 0.5).astype(int)
        b_raw, b_cal = isotonic_brier_delta(y, proba, n_classes=2)
        assert b_raw > b_cal  # calibration improves Brier

    def test_multiclass_returns_two_brier_values(self) -> None:
        rng = np.random.default_rng(0)
        n = 1_000
        proba = rng.uniform(0, 1, (n, 3))
        proba = proba / proba.sum(axis=1, keepdims=True)
        y = rng.integers(0, 3, n)
        b_raw, b_cal = isotonic_brier_delta(y, proba, n_classes=3)
        assert np.isfinite(b_raw)
        assert np.isfinite(b_cal)
        assert b_raw > 0


class TestCalibrationVerdict:
    def test_thresholds(self) -> None:
        assert calibration_verdict(0.0001) == "FLAT"
        assert calibration_verdict(0.003) == "MINIMAL"
        assert calibration_verdict(0.010) == "MODEST"
        assert calibration_verdict(0.050) == "MEANINGFUL"

    def test_invalid_handles_nan(self) -> None:
        assert calibration_verdict(float("nan")) == "INVALID"


class TestCalibrationSummary:
    def test_end_to_end_on_synthetic_results(self) -> None:
        X_df, multi_labels, folds = _build_synthetic(n=200)
        results = train_multi_spec_cpcv(
            X_df, multi_labels, folds, boost_rounds=20,
        )
        summary = calibration_summary(results)
        assert isinstance(summary, pd.DataFrame)
        assert list(summary.columns) == [
            "spec", "auc", "n_classes", "brier_raw", "brier_cal",
            "delta_brier", "verdict", "n_oof",
        ]
        # All non-degenerate specs should appear
        assert "tb_a" in summary["spec"].values
        # Verdicts should be one of the expected strings
        assert set(summary["verdict"]).issubset(
            {"FLAT", "MINIMAL", "MODEST", "MEANINGFUL", "INVALID"},
        )

    def test_sorted_by_delta_descending(self) -> None:
        X_df, multi_labels, folds = _build_synthetic(n=300)
        results = train_multi_spec_cpcv(
            X_df, multi_labels, folds, boost_rounds=20,
        )
        summary = calibration_summary(results, sort_by_delta=True)
        if len(summary) >= 2:
            # delta_brier descending
            deltas = summary["delta_brier"].dropna()
            assert (deltas.diff().dropna() <= 1e-9).all()

    def test_empty_results_returns_empty_df(self) -> None:
        summary = calibration_summary({})
        assert len(summary) == 0
