"""
p6lab.validation.meta_labeler — Wave 5 Phase 5C

Lopez de Prado §3.6 meta-labeling. The primary LGBM (NB06 §04) already
produces tier-A/B/C probabilities. A secondary classifier fitted only on
tier-A candidates decides ``take_bet / skip`` with the primary_proba +
live feature snapshot + a rolling P&L streak feature.

The goal is **tier-A false-positive reduction at the same recall** — we
do not lower the primary's threshold, we gate the tier-A population a
second time so the surviving "take-bet" fraction has a cleaner
precision.

Feature vector (5 columns):
    primary_proba        — the primary model's probability for the row
    fi_fast              — live fragility index scalar
    imbalance_ema        — L2 imbalance EWMA (index 5 of L2FeatureNames)
    spread_bps           — bid-ask spread in basis points
    recent_pnl_streak    — rolling net hit-rate over the last N closed
                           outcomes ∈ [-1, 1]

Outputs two artifacts:
    - Fitted LGBM (``model``) with ``predict_take_bet(X) -> np.ndarray[bool]``
    - ``evaluate(...)`` dict reporting tier-A precision/recall before and
      after gating, so the bake-off cell can print a decision table.

Usage (NB06 §04-bis, runnable):

    from p6lab.validation.meta_labeler import (
        META_FEATURE_COLS, MetaLabeler, MetaLabelerConfig, compute_recent_pnl_streak,
    )
    streak = compute_recent_pnl_streak(closed_outcomes, window=20)
    meta_X = pd.DataFrame({
        'primary_proba': all_proba,
        'fi_fast':       fi_fast_series,
        'imbalance_ema': imb_series,
        'spread_bps':    spread_series,
        'recent_pnl_streak': streak_series,
    })
    meta = MetaLabeler().fit(meta_X, primary_proba=all_proba, y_true=all_y)
    report = meta.evaluate(meta_X, primary_proba=all_proba, y_true=all_y)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


META_FEATURE_COLS: tuple[str, ...] = (
    "primary_proba",
    "fi_fast",
    "imbalance_ema",
    "spread_bps",
    "recent_pnl_streak",
)


# ---------------------------------------------------------------------------
# Rolling P&L streak helper
# ---------------------------------------------------------------------------


def compute_recent_pnl_streak(
    outcomes: Iterable[Any],
    *,
    window: int = 20,
) -> float:
    """Return net hit-rate signal over the last ``window`` outcomes.

    Each outcome must expose a boolean ``.hit`` attribute (matches the
    ``_ClosedOutcome`` shape produced by ``OutcomeTrackerRenderer``).
    Empty history → 0.0; fewer than ``window`` items uses what's present.
    Result ∈ [-1.0, 1.0]: +1.0 means every bet was a hit, -1.0 all misses.
    """
    seq = [bool(getattr(o, "hit", False)) for o in outcomes]
    if not seq:
        return 0.0
    recent = seq[-int(window):]
    hits = sum(1 for x in recent if x)
    misses = len(recent) - hits
    return float((hits - misses) / max(len(recent), 1))


def build_meta_features(
    *,
    primary_proba: np.ndarray,
    fi_fast: np.ndarray,
    imbalance_ema: np.ndarray,
    spread_bps: np.ndarray,
    recent_pnl_streak: np.ndarray | float,
) -> pd.DataFrame:
    """Bundle the 5 column vectors into a DataFrame with the canonical
    column order. ``recent_pnl_streak`` may be scalar (broadcast to all
    rows) or a full series."""
    n = len(primary_proba)
    if np.isscalar(recent_pnl_streak):
        streak = np.full(n, float(recent_pnl_streak), dtype=float)
    else:
        streak = np.asarray(recent_pnl_streak, dtype=float)
        if len(streak) != n:
            raise ValueError(
                f"recent_pnl_streak length {len(streak)} ≠ primary_proba length {n}"
            )
    return pd.DataFrame({
        "primary_proba": np.asarray(primary_proba, dtype=float),
        "fi_fast": np.asarray(fi_fast, dtype=float),
        "imbalance_ema": np.asarray(imbalance_ema, dtype=float),
        "spread_bps": np.asarray(spread_bps, dtype=float),
        "recent_pnl_streak": streak,
    })


# ---------------------------------------------------------------------------
# MetaLabeler
# ---------------------------------------------------------------------------


@dataclass
class MetaLabelerConfig:
    """Hyperparameters for the secondary LGBM."""
    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 5
    num_leaves: int = 15
    min_child_samples: int = 10
    random_state: int = 42
    tier_a_threshold: float = 0.85
    min_train_samples: int = 50
    take_bet_threshold: float = 0.5


@dataclass
class MetaLabelerReport:
    """Before/after precision-recall report on tier-A subset."""
    tier_a_n_before: int = 0
    tier_a_n_after: int = 0
    tier_a_precision_before: float = 0.0
    tier_a_precision_after: float = 0.0
    tier_a_recall_before: float = 0.0
    tier_a_recall_after: float = 0.0
    tier_a_fp_before: int = 0
    tier_a_fp_after: int = 0
    fp_reduction_pct: float = 0.0
    take_bet_threshold: float = 0.5

    def to_dict(self) -> dict:
        return {
            "tier_a_n_before": self.tier_a_n_before,
            "tier_a_n_after": self.tier_a_n_after,
            "tier_a_precision_before": self.tier_a_precision_before,
            "tier_a_precision_after": self.tier_a_precision_after,
            "tier_a_recall_before": self.tier_a_recall_before,
            "tier_a_recall_after": self.tier_a_recall_after,
            "tier_a_fp_before": self.tier_a_fp_before,
            "tier_a_fp_after": self.tier_a_fp_after,
            "fp_reduction_pct": self.fp_reduction_pct,
            "take_bet_threshold": self.take_bet_threshold,
        }


class MetaLabeler:
    """Secondary LGBM gating the primary model's tier-A predictions.

    Fit contract
    ------------
    The secondary trains **only** on rows where primary_proba ≥
    ``tier_a_threshold``. Target is ``primary_was_correct``: 1 when the
    primary called bull and y_true == 1, else 0. At predict time, the
    secondary sees the same 5-column feature matrix; ``predict_take_bet``
    returns True only for rows where the tier-A gate AND the secondary's
    probability both clear their thresholds.
    """

    def __init__(self, config: MetaLabelerConfig | None = None) -> None:
        self.config = config or MetaLabelerConfig()
        self.model_: Any = None
        self.is_fitted_: bool = False
        self.train_samples_: int = 0

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        *,
        primary_proba: np.ndarray,
        y_true: np.ndarray,
    ) -> "MetaLabeler":
        """Fit the secondary on the tier-A subset only.

        ``X`` must carry the full 5-column ``META_FEATURE_COLS`` frame.
        ``primary_proba`` is the primary LGBM's probability per row.
        ``y_true`` is the primary's binary target (triple-barrier label).
        """
        self._validate_columns(X)
        primary_proba = np.asarray(primary_proba, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        if len(primary_proba) != len(X) or len(y_true) != len(X):
            raise ValueError("X / primary_proba / y_true must share length")

        mask = primary_proba >= self.config.tier_a_threshold
        self.train_samples_ = int(mask.sum())
        if self.train_samples_ < self.config.min_train_samples:
            # Not enough tier-A samples to train. Mark unfitted; predict
            # falls back to "pass through the primary tier-A gate only".
            logger.warning(
                "MetaLabeler: only %d tier-A samples (< %d); skipping fit",
                self.train_samples_, self.config.min_train_samples,
            )
            self.is_fitted_ = False
            return self

        y_secondary = (y_true[mask] == 1).astype(int)
        if len(np.unique(y_secondary)) < 2:
            # One-class problem — no signal to learn. Skip.
            logger.warning(
                "MetaLabeler: tier-A subset is single-class; skipping fit",
            )
            self.is_fitted_ = False
            return self

        import lightgbm as lgb   # delayed import — optional dep
        model = lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        X_train = X.loc[mask, list(META_FEATURE_COLS)]
        model.fit(X_train, y_secondary)
        self.model_ = model
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Secondary's probability-of-correctness. Unfitted → zeros."""
        self._validate_columns(X)
        if not self.is_fitted_ or self.model_ is None:
            return np.zeros(len(X), dtype=float)
        return self.model_.predict_proba(X[list(META_FEATURE_COLS)])[:, 1]

    def predict_take_bet(
        self,
        X: pd.DataFrame,
        *,
        primary_proba: np.ndarray,
    ) -> np.ndarray:
        """Return a bool per row: True if BOTH the primary tier-A gate and
        the secondary threshold pass.
        When unfitted, fall back to tier-A gate alone (identity)."""
        self._validate_columns(X)
        primary_proba = np.asarray(primary_proba, dtype=float)
        tier_a_mask = primary_proba >= self.config.tier_a_threshold
        if not self.is_fitted_:
            return tier_a_mask
        secondary_proba = self.predict_proba(X)
        return tier_a_mask & (secondary_proba >= self.config.take_bet_threshold)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: pd.DataFrame,
        *,
        primary_proba: np.ndarray,
        y_true: np.ndarray,
    ) -> MetaLabelerReport:
        """Compute before/after tier-A precision + recall + FP counts."""
        self._validate_columns(X)
        primary_proba = np.asarray(primary_proba, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        tier_a_mask = primary_proba >= self.config.tier_a_threshold
        n_before = int(tier_a_mask.sum())
        tp_before = int(((y_true == 1) & tier_a_mask).sum())
        fp_before = int(((y_true == 0) & tier_a_mask).sum())
        total_positive = int((y_true == 1).sum())

        if n_before == 0:
            return MetaLabelerReport(take_bet_threshold=self.config.take_bet_threshold)

        precision_before = tp_before / n_before
        recall_before = tp_before / max(total_positive, 1)

        gated = self.predict_take_bet(X, primary_proba=primary_proba)
        n_after = int(gated.sum())
        tp_after = int(((y_true == 1) & gated).sum())
        fp_after = int(((y_true == 0) & gated).sum())
        precision_after = tp_after / n_after if n_after else 0.0
        recall_after = tp_after / max(total_positive, 1)

        fp_reduction = 0.0
        if fp_before > 0:
            fp_reduction = (fp_before - fp_after) / fp_before

        return MetaLabelerReport(
            tier_a_n_before=n_before,
            tier_a_n_after=n_after,
            tier_a_precision_before=precision_before,
            tier_a_precision_after=precision_after,
            tier_a_recall_before=recall_before,
            tier_a_recall_after=recall_after,
            tier_a_fp_before=fp_before,
            tier_a_fp_after=fp_after,
            fp_reduction_pct=fp_reduction,
            take_bet_threshold=self.config.take_bet_threshold,
        )

    @staticmethod
    def _validate_columns(X: pd.DataFrame) -> None:
        missing = [c for c in META_FEATURE_COLS if c not in X.columns]
        if missing:
            raise ValueError(
                f"MetaLabeler: X is missing columns {missing}; "
                f"expected {list(META_FEATURE_COLS)}"
            )
