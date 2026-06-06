"""
p6lab.validation.meta_labeler_cpcv — Wave 8.5-F

Out-of-fold evaluation of MetaLabeler via Combinatorial Purged CV.
Replaces Wave 5's in-sample `MetaLabeler.evaluate()` — the in-sample
FP-reduction number was an artifact of the model fitting on the same
rows it was scored on. Held-out folds give a defensible reading.

Exported:
    CPCVMetaReport  dataclass
    evaluate_cpcv(X, primary_proba, y_true, timestamps, ...) → CPCVMetaReport

Contract
--------
For each fold:
  1. Fit a fresh `MetaLabeler` on the fold's train indices.
  2. Predict on the fold's test indices.
  3. Record OOF predictions + per-fold `MetaLabelerReport`.
After all folds:
  4. Aggregate OOF predictions into a single `MetaLabelerReport`.
  5. Return both the aggregated report and the per-fold list.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from p6lab.validation.cpcv import CascadeAwareCPCV
from p6lab.validation.meta_labeler import (
    META_FEATURE_COLS,
    MetaLabeler,
    MetaLabelerConfig,
    MetaLabelerReport,
)

logger = logging.getLogger(__name__)


@dataclass
class CPCVMetaReport:
    """Aggregated OOF report + per-fold breakdown."""
    aggregated: MetaLabelerReport = field(default_factory=MetaLabelerReport)
    per_fold: list[MetaLabelerReport] = field(default_factory=list)
    folds_run: int = 0
    folds_skipped_insufficient_samples: int = 0
    oof_predictions: np.ndarray = field(default_factory=lambda: np.asarray([]))

    def to_dict(self) -> dict:
        return {
            "aggregated": self.aggregated.to_dict(),
            "per_fold": [r.to_dict() for r in self.per_fold],
            "folds_run": self.folds_run,
            "folds_skipped_insufficient_samples": self.folds_skipped_insufficient_samples,
            "oof_predictions_shape": list(self.oof_predictions.shape),
        }


def evaluate_cpcv(
    X: pd.DataFrame,
    *,
    primary_proba: np.ndarray,
    y_true: np.ndarray,
    timestamps: pd.Series,
    n_splits: int = 5,
    n_test_groups: int = 2,
    config: MetaLabelerConfig | None = None,
) -> CPCVMetaReport:
    """Run MetaLabeler.fit + predict across CPCV folds, return aggregated
    OOF report + per-fold reports.

    Parameters
    ----------
    X
        5-column feature frame matching ``META_FEATURE_COLS``.
    primary_proba
        Primary model's class-1 probability per row.
    y_true
        Triple-barrier binary label per row.
    timestamps
        Row timestamps — drives the CPCV splitter's temporal grouping.
    n_splits, n_test_groups
        CPCV parameters; defaults match Wave 3's NB06 §04 choices.
    config
        Optional MetaLabelerConfig; defaults match production.

    Returns
    -------
    CPCVMetaReport
        - ``aggregated``: OOF `MetaLabelerReport` over concatenated
          held-out predictions.
        - ``per_fold``: per-fold in-fold reports (each fold's secondary
          fitted on its own train, evaluated on its own test).
        - ``oof_predictions``: boolean array aligned to ``X`` showing
          the take-bet decision across all folds.
    """
    cfg = config or MetaLabelerConfig()
    primary_proba = np.asarray(primary_proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    n = len(X)
    if n == 0:
        return CPCVMetaReport()

    cv = CascadeAwareCPCV(n_splits=n_splits, n_test_groups=n_test_groups,
                          cascade_embargo_days=0)
    folds = cv.split(pd.DataFrame(X), timestamps)
    oof_decisions = np.zeros(n, dtype=bool)
    oof_covered = np.zeros(n, dtype=bool)
    per_fold: list[MetaLabelerReport] = []
    folds_skipped = 0

    for fold in folds:
        train_idx = fold.train_idx
        test_idx = fold.test_idx
        if len(train_idx) < cfg.min_train_samples or len(test_idx) == 0:
            folds_skipped += 1
            continue
        meta = MetaLabeler(cfg)
        meta.fit(
            X.iloc[train_idx],
            primary_proba=primary_proba[train_idx],
            y_true=y_true[train_idx],
        )
        decisions = meta.predict_take_bet(
            X.iloc[test_idx],
            primary_proba=primary_proba[test_idx],
        )
        oof_decisions[test_idx] = decisions
        oof_covered[test_idx] = True
        # In-fold report for diagnostic variance inspection
        per_fold.append(
            meta.evaluate(
                X.iloc[test_idx],
                primary_proba=primary_proba[test_idx],
                y_true=y_true[test_idx],
            )
        )

    folds_run = len(per_fold)
    if folds_run == 0:
        logger.warning(
            "wave85-F evaluate_cpcv: no folds passed the min-samples gate; "
            "aggregated report is empty"
        )
        return CPCVMetaReport(
            folds_run=0,
            folds_skipped_insufficient_samples=folds_skipped,
        )

    # Build the aggregated OOF report manually — we have the decisions,
    # we know the tier-A tier mask, we can compute TP/FP directly.
    covered = oof_covered
    y_cov = y_true[covered]
    proba_cov = primary_proba[covered]
    dec_cov = oof_decisions[covered]
    tier_a_mask = proba_cov >= cfg.tier_a_threshold

    n_before = int(tier_a_mask.sum())
    tp_before = int(((y_cov == 1) & tier_a_mask).sum())
    fp_before = int(((y_cov == 0) & tier_a_mask).sum())
    total_positive = int((y_cov == 1).sum())
    precision_before = tp_before / max(n_before, 1)
    recall_before = tp_before / max(total_positive, 1)

    n_after = int(dec_cov.sum())
    tp_after = int(((y_cov == 1) & dec_cov).sum())
    fp_after = int(((y_cov == 0) & dec_cov).sum())
    precision_after = tp_after / max(n_after, 1) if n_after else 0.0
    recall_after = tp_after / max(total_positive, 1)

    fp_reduction = (fp_before - fp_after) / fp_before if fp_before else 0.0

    aggregated = MetaLabelerReport(
        tier_a_n_before=n_before,
        tier_a_n_after=n_after,
        tier_a_precision_before=precision_before,
        tier_a_precision_after=precision_after,
        tier_a_recall_before=recall_before,
        tier_a_recall_after=recall_after,
        tier_a_fp_before=fp_before,
        tier_a_fp_after=fp_after,
        fp_reduction_pct=fp_reduction,
        take_bet_threshold=cfg.take_bet_threshold,
    )
    logger.info(
        "wave85-F evaluate_cpcv: folds=%d OOF tier_a_n=%d fp_reduction=%.3f",
        folds_run, n_before, fp_reduction,
    )
    return CPCVMetaReport(
        aggregated=aggregated,
        per_fold=per_fold,
        folds_run=folds_run,
        folds_skipped_insufficient_samples=folds_skipped,
        oof_predictions=oof_decisions[covered],
    )
