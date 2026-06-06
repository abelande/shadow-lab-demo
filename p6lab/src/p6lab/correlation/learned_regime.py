"""
p6lab.correlation.learned_regime — Wave 7 Phase 7H

Learned drop-in for the 4-way VIX-bucket regime classifier. Trains a
small classifier on (FI_fast, network_momentum, peer_correlation_avg,
imbalance_ema, spread_bps, realized_variance_30s, cross_asset_adjacency)
→ ``{low, normal, elevated, high}`` regime labels generated from the
historical VIX series or (when VIX isn't present) from a robust
regime-binning on the realized-variance column.

The trained model implements the same contract as the old VIX-threshold
classifier — callers just swap their regime_classifier callable:

    clf = LearnedRegimeClassifier(model_path=...)
    regime = clf.predict(feature_dict)   # → "normal", etc.

Exported:
    REGIME_LABELS           tuple[str, ...]
    FeatureContract         dataclass
    LearnedRegimeClassifier class
    train_learned_regime(feature_frame, vix_series) → LearnedRegimeClassifier
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


REGIME_LABELS: tuple[str, ...] = ("low", "normal", "elevated", "high")


DEFAULT_FEATURES: tuple[str, ...] = (
    "fi_fast",
    "network_momentum",
    "peer_correlation_avg",
    "imbalance_ema",
    "spread_bps",
    "realized_variance_30s",
    "cross_asset_adjacency",
)


@dataclass
class FeatureContract:
    """What columns the classifier expects at predict time.

    Defaults match the Wave 6/7 feature matrix after cross-asset + micro
    features land. Callers supply zeros for any missing key; the
    classifier treats missing features as neutral."""
    columns: tuple[str, ...] = DEFAULT_FEATURES


# ---------------------------------------------------------------------------
# Labeling helpers
# ---------------------------------------------------------------------------


def _vix_to_regime(vix: float) -> str:
    if vix < 15.0:
        return "low"
    if vix < 25.0:
        return "normal"
    if vix < 35.0:
        return "elevated"
    return "high"


def _quantile_bin(values: np.ndarray) -> np.ndarray:
    """Fallback when VIX labels aren't available — bin ``values`` into
    4 roughly-equal quartiles and assign REGIME_LABELS in ascending
    volatility order."""
    if values.size == 0:
        return np.asarray([], dtype=int)
    ranks = values.argsort().argsort()
    n = values.size
    # Split into 4 nearly-equal buckets
    bins = np.minimum(3, ranks * 4 // max(n, 1))
    return bins


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class LearnedRegimeClassifier:
    """Small multinomial-logistic classifier, serialized as a dict of
    coefficient arrays + intercepts so we don't hard-depend on a
    specific scikit-learn version at load time."""

    def __init__(
        self,
        *,
        contract: FeatureContract | None = None,
    ) -> None:
        self.contract = contract or FeatureContract()
        self._coef: np.ndarray | None = None      # (n_classes, n_features)
        self._intercept: np.ndarray | None = None # (n_classes,)
        self._labels: tuple[str, ...] = REGIME_LABELS
        self.is_fitted_: bool = False
        # Wave 8.5-D: per-instance sentinel for "unfitted + missing RV"
        # warning. Instance-level (not class-level) so two classifiers
        # in the same process warn independently — otherwise a shared
        # warning would silence the second instance incorrectly.
        self._warned_missing_rv: bool = False

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | list[str],
    ) -> "LearnedRegimeClassifier":
        """Fit on ``X`` (feature frame) + ``y`` (regime labels)."""
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - sklearn is a lab dep
            raise RuntimeError(
                "scikit-learn is required for LearnedRegimeClassifier.fit()"
            ) from exc

        mat = self._to_matrix(X)
        lab = np.asarray(y)
        model = LogisticRegression(max_iter=500, n_jobs=-1)
        model.fit(mat, lab)
        self._coef = np.asarray(model.coef_, dtype=float)
        self._intercept = np.asarray(model.intercept_, dtype=float)
        self._labels = tuple(model.classes_)
        self.is_fitted_ = True
        return self

    def predict(self, features: dict[str, float] | pd.DataFrame) -> str:
        """Single-row prediction — returns the regime label."""
        if isinstance(features, pd.DataFrame):
            X = self._to_matrix(features)
        else:
            row = np.asarray(
                [[float(features.get(c, 0.0)) for c in self.contract.columns]],
                dtype=float,
            )
            X = row
        if not self.is_fitted_ or self._coef is None or self._intercept is None:
            # Unfitted fallback: use the realized_variance_30s column
            rv_present = (
                isinstance(features, dict) and "realized_variance_30s" in features
            )
            # Wave 8.5-D: one-shot WARNING if we fall through to the
            # always-"low" default because RV key is absent. This
            # catches the case where a notebook user silently sees
            # "everything is low regime" for hours.
            if not rv_present and not self._warned_missing_rv:
                logger.warning(
                    "LearnedRegimeClassifier unfitted + 'realized_variance_30s' "
                    "missing from features dict; defaulting to 'low'. "
                    "Either fit() the classifier first or ensure the "
                    "feature dict includes 'realized_variance_30s'."
                )
                self._warned_missing_rv = True
            rv = float(features.get("realized_variance_30s", 0.0)) if isinstance(features, dict) else 0.0
            if rv < 0.25:
                return "low"
            if rv < 1.0:
                return "normal"
            if rv < 3.0:
                return "elevated"
            return "high"
        logits = X @ self._coef.T + self._intercept
        idx = int(np.argmax(logits[0]))
        return str(self._labels[idx])

    def predict_proba(
        self, features: dict[str, float] | pd.DataFrame,
    ) -> dict[str, float]:
        """Per-class probability. Uniform when unfitted."""
        if not self.is_fitted_ or self._coef is None or self._intercept is None:
            return {lbl: 1.0 / len(self._labels) for lbl in self._labels}
        if isinstance(features, pd.DataFrame):
            X = self._to_matrix(features)
        else:
            X = np.asarray(
                [[float(features.get(c, 0.0)) for c in self.contract.columns]],
                dtype=float,
            )
        logits = X @ self._coef.T + self._intercept
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = ex / ex.sum(axis=1, keepdims=True)
        return {lbl: float(probs[0, i]) for i, lbl in enumerate(self._labels)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coef": None if self._coef is None else self._coef.tolist(),
            "intercept": None if self._intercept is None else self._intercept.tolist(),
            "labels": list(self._labels),
            "columns": list(self.contract.columns),
            "is_fitted": self.is_fitted_,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path | str) -> "LearnedRegimeClassifier":
        path = Path(path)
        payload = json.loads(path.read_text())
        clf = cls(contract=FeatureContract(columns=tuple(payload.get("columns", DEFAULT_FEATURES))))
        if payload.get("is_fitted"):
            clf._coef = np.asarray(payload["coef"], dtype=float)
            clf._intercept = np.asarray(payload["intercept"], dtype=float)
            clf._labels = tuple(payload["labels"])
            clf.is_fitted_ = True
        return clf

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _to_matrix(self, X: pd.DataFrame) -> np.ndarray:
        cols = list(self.contract.columns)
        present = [c for c in cols if c in X.columns]
        missing = [c for c in cols if c not in X.columns]
        if missing:
            X = X.copy()
            for c in missing:
                X[c] = 0.0
        return X[cols].to_numpy(dtype=float, copy=True)


# ---------------------------------------------------------------------------
# Convenience training helper
# ---------------------------------------------------------------------------


def train_learned_regime(
    feature_frame: pd.DataFrame,
    *,
    vix_series: np.ndarray | pd.Series | None = None,
    realized_variance_col: str = "realized_variance_30s",
) -> LearnedRegimeClassifier:
    """Label ``feature_frame`` via VIX buckets (when provided) or by
    quantile-binning ``realized_variance_col`` and fit a classifier.

    Useful for a one-liner notebook workflow — pass in a Wave 6/7
    feature frame and receive a trained classifier ready to drop into
    ``RegimeConditioner``.
    """
    n = len(feature_frame)
    if n == 0:
        raise ValueError("feature_frame is empty")

    if vix_series is not None:
        vix = np.asarray(vix_series, dtype=float)
        if vix.size != n:
            raise ValueError("vix_series length must match feature_frame length")
        labels = np.asarray([_vix_to_regime(float(v)) for v in vix])
    else:
        if realized_variance_col not in feature_frame.columns:
            raise ValueError(
                f"vix_series not provided and '{realized_variance_col}' missing"
            )
        bins = _quantile_bin(feature_frame[realized_variance_col].to_numpy())
        labels = np.asarray([REGIME_LABELS[int(b)] for b in bins])

    clf = LearnedRegimeClassifier()
    clf.fit(feature_frame, labels)
    return clf
