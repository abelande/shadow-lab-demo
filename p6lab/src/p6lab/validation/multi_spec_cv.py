"""Multi-spec CPCV training loop — Wave 9+10 §04 multi-target adapter.

Drives the canonical Wave 9+10 diagnostic: train one LightGBM model per
label spec (TB / MFE-MAE / pattern-firing × multi-horizon) under CPCV,
aggregate per-spec OOF predictions, and emit a comparable AUC/Brier
table so §04d/§04e can identify which (label_kind, horizon) the
features actually answer.

Why a module rather than inline NB06 code:
- testable in isolation (synthetic small CPCV)
- reusable from NB07 once 10-B introduces pattern-firing-only training
- centralizes the binary-vs-multiclass dispatch + class weighting

References: ``reports/P6LAB-WAVE-9-10-BUILD-PHASES.md`` §H.1.b for the
multi-target rationale, §H.2 for class-imbalance handling.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default LightGBM hyperparameters — kept light so 16 specs train in a
# tractable wall-clock. Diagnostic relevance > optimal absolute AUC.
DEFAULT_BOOST_ROUNDS: int = 50
DEFAULT_LR: float = 0.1
DEFAULT_MAX_DEPTH: int = 6
DEFAULT_NUM_LEAVES: int = 15
DEFAULT_MIN_CHILD_SAMPLES: int = 20


@dataclass
class SpecResult:
    """One label spec's CPCV outcome."""
    name: str
    mean_auc: float
    std_auc: float
    n_folds: int
    n_classes: int
    class_dist: dict[int, int]
    fold_preds: list   # list[(y_true, proba, test_idx)]


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """sklearn-style "balanced" weights: n_samples / (n_classes * count_c)."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    n_classes = len(classes)
    if n_classes == 0:
        return np.ones(0, dtype=float)
    per_class = {
        int(c): n / (n_classes * cnt)
        for c, cnt in zip(classes, counts)
    }
    return np.asarray(
        [per_class[int(v)] for v in y], dtype=float,
    )


def _train_one_fold(
    X_df: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    is_multi: bool,
    n_classes: int,
    boost_rounds: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Train one LightGBM model on ``train_idx``, predict on ``test_idx``.

    Returns ``(y_test, proba)`` where proba is shape ``(n_test, n_classes)``
    for multi-class or ``(n_test,)`` for binary (positive-class proba).

    Returns ``None`` when the fold can't yield a usable AUC (e.g., one
    class missing from train or test).
    """
    import lightgbm as lgb

    train_y = y[train_idx]
    test_y = y[test_idx]
    if len(np.unique(train_y)) < 2:
        return None
    if len(np.unique(test_y)) < 2:
        return None

    sample_w = _balanced_sample_weights(train_y)
    if is_multi:
        clf = lgb.LGBMClassifier(
            n_estimators=boost_rounds,
            learning_rate=DEFAULT_LR,
            max_depth=DEFAULT_MAX_DEPTH,
            num_leaves=DEFAULT_NUM_LEAVES,
            min_child_samples=DEFAULT_MIN_CHILD_SAMPLES,
            objective="multiclass",
            num_class=n_classes,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        clf = lgb.LGBMClassifier(
            n_estimators=boost_rounds,
            learning_rate=DEFAULT_LR,
            max_depth=DEFAULT_MAX_DEPTH,
            num_leaves=DEFAULT_NUM_LEAVES,
            min_child_samples=DEFAULT_MIN_CHILD_SAMPLES,
            objective="binary",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    clf.fit(X_df.iloc[train_idx], train_y, sample_weight=sample_w)
    if is_multi:
        proba = clf.predict_proba(X_df.iloc[test_idx])  # (n, n_classes)
    else:
        proba = clf.predict_proba(X_df.iloc[test_idx])[:, 1]  # (n,)
    return test_y, proba


def _fold_auc(
    y_test: np.ndarray,
    proba: np.ndarray,
    is_multi: bool,
) -> float:
    """ROC-AUC for one fold (binary or multi-class macro-OVR).

    Returns ``nan`` when AUC is undefined (e.g., test fold has only one
    class for binary, or sklearn raises ValueError on a multi-class
    fold missing a class).
    """
    from sklearn.metrics import roc_auc_score

    try:
        if is_multi:
            return float(roc_auc_score(
                y_test, proba, multi_class="ovr", average="macro",
            ))
        return float(roc_auc_score(y_test, proba))
    except ValueError:
        return float("nan")


def train_multi_spec_cpcv(
    X_df: pd.DataFrame,
    multi_labels: pd.DataFrame,
    folds: Iterable | None = None,
    *,
    purge_rows: int = 0,
    boost_rounds: int = DEFAULT_BOOST_ROUNDS,
    random_state: int = 42,
    apply_row_purge: callable | None = None,
    valid_masks: dict[str, np.ndarray] | None = None,
    folds_factory: callable | None = None,
) -> dict[str, SpecResult]:
    """Train one CPCV LightGBM per column of ``multi_labels``.

    Two operating modes:

    1. **Shared-folds (legacy)** — ``valid_masks=None``, ``folds`` provided.
       Single global fold list applied to every spec. Caller is
       responsible for any pre-filtering of ``X_df`` / ``multi_labels``.
       Used by older callers that pre-aligned to a global valid mask.

    2. **Per-spec masks** — ``valid_masks`` dict provided, ``folds_factory``
       provided. Each spec gets its own row mask (e.g., ``tb_*`` drops
       timeouts, ``mm_*`` keeps all rows, ``pf_*`` drops NaN), and folds
       are rebuilt per spec on the filtered subset via ``folds_factory``.
       Resolves the §04-multi horizon-collapse bug where a single global
       mask (e.g. 60s TB validity) clipped longer-horizon specs.

    Per-spec dispatch:
      - 2 unique classes  → binary objective
      - >2 unique classes → multi-class (objective=multiclass, OVR macro AUC)
      - <2 unique classes → skipped with a log warning

    Class imbalance handled via sklearn-style "balanced" sample weights.

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature matrix. In legacy mode must have
        ``len(X_df) == len(multi_labels)`` and is used as-is. In per-spec
        mode is the *full* feature frame; per-spec filtering happens here.
    multi_labels : pd.DataFrame
        One column per label spec. Each column trained independently.
    folds : iterable, optional
        Pre-built CPCV folds (legacy mode). Ignored when ``valid_masks``
        is provided.
    purge_rows : int
        Forwarded to ``apply_row_purge`` (when supplied) to drop train
        rows within ``purge_rows`` of any test row. Default 0 = no purge.
    boost_rounds : int
        LightGBM ``n_estimators``. Default 50.
    random_state : int
        For reproducibility.
    apply_row_purge : callable, optional
        Function ``(train_idx, test_idx, purge_rows) -> filtered_train_idx``.
        When ``None``, no purge applied.
    valid_masks : dict[str, np.ndarray], optional
        Per-spec boolean row masks (length ``len(multi_labels)``). When
        provided, switches to per-spec mode; ``folds_factory`` becomes
        required. Specs not present in the dict default to "all rows
        finite" (``~pd.isna``).
    folds_factory : callable, optional
        ``(X_filtered: pd.DataFrame) -> Iterable[Fold]``. Called once per
        spec on its mask-filtered ``X``. Required in per-spec mode.

    Returns
    -------
    dict[str, SpecResult]
        Mapping spec_name → ``SpecResult``. Specs with no usable folds
        are omitted with a warning log.
    """
    per_spec_mode = valid_masks is not None
    if per_spec_mode:
        if folds_factory is None:
            raise ValueError(
                "valid_masks supplied but folds_factory is None — "
                "per-spec mode needs a folds_factory(X_filtered) callable",
            )
        if len(X_df) != len(multi_labels):
            raise ValueError(
                f"X_df rows {len(X_df)} != multi_labels rows "
                f"{len(multi_labels)} (per-spec mode requires aligned full "
                f"frames; mask filtering happens internally)",
            )
    else:
        if folds is None:
            raise ValueError(
                "legacy mode requires folds; or pass valid_masks + "
                "folds_factory for per-spec mode",
            )
        if len(X_df) != len(multi_labels):
            raise ValueError(
                f"X_df rows {len(X_df)} != multi_labels rows "
                f"{len(multi_labels)}",
            )

    # Legacy mode: pre-compute purged fold indices ONCE — the same purge
    # applies to every spec. Per-spec mode rebuilds folds inside the
    # column loop, so this optimization only fires for legacy callers.
    prepped: list[tuple[np.ndarray, np.ndarray]] | None
    if per_spec_mode:
        prepped = None
    else:
        folds_list = list(folds)
        prepped = []
        for f in folds_list:
            tr = np.asarray(f.train_idx)
            te = np.asarray(f.test_idx)
            if apply_row_purge is not None and purge_rows > 0:
                tr = np.asarray(apply_row_purge(tr, te, purge_rows))
            if len(tr) == 0:
                continue
            prepped.append((tr, te))

    results: dict[str, SpecResult] = {}

    for col in multi_labels.columns:
        y_full = multi_labels[col].to_numpy()

        if per_spec_mode:
            mask = valid_masks.get(col)
            if mask is None:
                # Default: rows where the label is finite + non-NaN
                if pd.api.types.is_float_dtype(multi_labels[col]):
                    mask = np.isfinite(y_full)
                else:
                    mask = ~pd.isna(multi_labels[col]).to_numpy()
            mask = np.asarray(mask, dtype=bool)
            if mask.shape[0] != len(y_full):
                raise ValueError(
                    f"valid_masks[{col!r}] length {mask.shape[0]} != "
                    f"label length {len(y_full)}",
                )
            if mask.sum() < 100:
                logger.warning(
                    "multi_spec_cpcv │ skip %s — only %d valid rows",
                    col, int(mask.sum()),
                )
                continue
            X_col = X_df.loc[mask].reset_index(drop=True).fillna(0)
            y = y_full[mask]
            try:
                spec_folds = list(folds_factory(X_col))
            except Exception as exc:
                logger.warning(
                    "multi_spec_cpcv │ skip %s — folds_factory raised: %s",
                    col, exc,
                )
                continue
            spec_prepped: list[tuple[np.ndarray, np.ndarray]] = []
            for f in spec_folds:
                tr = np.asarray(f.train_idx)
                te = np.asarray(f.test_idx)
                if apply_row_purge is not None and purge_rows > 0:
                    tr = np.asarray(apply_row_purge(tr, te, purge_rows))
                if len(tr) == 0:
                    continue
                spec_prepped.append((tr, te))
            X_for_fit = X_col
            iter_prepped = spec_prepped
        else:
            y = y_full
            X_for_fit = X_df
            iter_prepped = prepped or []

        unique = np.unique(y)
        # Drop NaN entries for the class count (they slipped through if
        # the label dtype was float and the mask wasn't supplied).
        unique_finite = unique[~pd.isna(unique)] if unique.dtype.kind == "f" else unique
        if len(unique_finite) < 2:
            logger.info(
                "multi_spec_cpcv │ skip %s — degenerate (only class %s)",
                col, unique_finite.tolist(),
            )
            continue

        n_classes = len(unique_finite)
        is_multi = n_classes > 2

        fold_aucs: list[float] = []
        fold_preds: list = []
        for tr, te in iter_prepped:
            out = _train_one_fold(
                X_for_fit, y, tr, te,
                is_multi=is_multi,
                n_classes=n_classes,
                boost_rounds=boost_rounds,
                random_state=random_state,
            )
            if out is None:
                continue
            y_te, proba = out
            auc = _fold_auc(y_te, proba, is_multi)
            fold_aucs.append(auc)
            fold_preds.append(
                (y_te.copy(), np.asarray(proba).copy(), np.asarray(te).copy()),
            )

        if not fold_aucs:
            logger.warning(
                "multi_spec_cpcv │ %s: no usable folds — class imbalance "
                "or purge dropped everything", col,
            )
            continue

        mean_auc = float(np.nanmean(fold_aucs))
        std_auc = float(np.nanstd(fold_aucs))
        class_dist = {
            int(c): int(cnt)
            for c, cnt in zip(*np.unique(y, return_counts=True))
            if not pd.isna(c)
        }
        results[str(col)] = SpecResult(
            name=str(col),
            mean_auc=mean_auc,
            std_auc=std_auc,
            n_folds=len(fold_aucs),
            n_classes=n_classes,
            class_dist=class_dist,
            fold_preds=fold_preds,
        )
        logger.info(
            "multi_spec_cpcv │ %s: AUC=%.4f±%.4f n_classes=%d folds=%d "
            "n_rows=%d", col, mean_auc, std_auc, n_classes, len(fold_aucs),
            len(y),
        )

    return results


def summarize_results(
    results: dict[str, SpecResult],
    *,
    sort_by_auc: bool = True,
) -> pd.DataFrame:
    """Render results as a one-row-per-spec DataFrame for printing.

    Columns: ``spec, auc, std, n_folds, n_classes, modal_class_pct``.
    """
    rows = []
    for name, r in results.items():
        total = sum(r.class_dist.values())
        modal_pct = max(r.class_dist.values()) / total if total else float("nan")
        rows.append({
            "spec": name,
            "auc": r.mean_auc,
            "std": r.std_auc,
            "n_folds": r.n_folds,
            "n_classes": r.n_classes,
            "modal_class_pct": modal_pct,
        })
    df = pd.DataFrame(rows)
    if sort_by_auc and len(df) > 0:
        df = df.sort_values("auc", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Calibration diagnostics — Wave 9+10 §04d-multi / §04e-multi
# ---------------------------------------------------------------------------
#
# The §04d-multi and §04e-multi notebook cells iterate these helpers over
# multi_results to produce per-spec calibration tables and isotonic Brier
# deltas. Helpers live here for testability; notebook cells stay thin.


def aggregate_oof(
    result: SpecResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ``fold_preds`` into a single ``(all_y, all_proba)`` pair.

    For binary specs ``all_proba`` has shape ``(n,)``; for multi-class
    specs it has shape ``(n, n_classes)``. Mismatched fold proba widths
    (rare: some fold's training data missing a class) raise ValueError.
    """
    if not result.fold_preds:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    y_parts = [np.asarray(f[0]) for f in result.fold_preds]
    p_parts = [np.asarray(f[1]) for f in result.fold_preds]
    all_y = np.concatenate(y_parts)
    if p_parts[0].ndim == 1:
        all_proba = np.concatenate(p_parts)
    else:
        widths = {p.shape[1] for p in p_parts}
        if len(widths) != 1:
            raise ValueError(
                f"inconsistent proba widths across folds: {widths} "
                f"(spec={result.name})",
            )
        all_proba = np.concatenate(p_parts, axis=0)
    return all_y, all_proba


def calibration_table(
    all_y: np.ndarray,
    all_proba: np.ndarray,
    *,
    n_classes: int,
    deciles: int = 10,
    min_bin_n: int = 30,
) -> list[tuple[str, int, float, float]]:
    """Per-decile calibration rows.

    Returns ``[(bin_label, n, hit_rate, reliability), ...]``:

    - **Binary** (n_classes=2): bin = ``[k/D, (k+1)/D)`` of positive-class
      proba. ``hit_rate`` = fraction with ``y == positive_class`` in
      that bin. ``reliability`` = ``hit_rate − bin_midpoint``.
    - **Multi-class** (n_classes>2): bin = ``[k/D, (k+1)/D)`` of
      ``max(proba)`` (top-class confidence). ``hit_rate`` = fraction
      where ``argmax(proba) == y`` (top-class accuracy).

    Bins with fewer than ``min_bin_n`` samples are dropped.
    """
    from collections import defaultdict

    bins: dict[int, list[int]] = defaultdict(list)
    if n_classes == 2:
        if all_proba.ndim != 1:
            raise ValueError(
                f"binary spec needs 1D proba; got shape {all_proba.shape}",
            )
        # Positive class = larger numeric label (matches LGBMClassifier
        # internal LabelEncoder: classes_ = sorted(np.unique(y))).
        pos_class = int(max(np.unique(all_y))) if len(all_y) else 0
        y_bin = (all_y == pos_class).astype(int)
        for p, y in zip(all_proba, y_bin):
            b = min(int(p * deciles), deciles - 1)
            bins[b].append(int(y))
    else:
        if all_proba.ndim != 2:
            raise ValueError(
                f"multi-class spec needs 2D proba; got shape {all_proba.shape}",
            )
        classes_sorted = sorted(np.unique(all_y).tolist())
        if all_proba.shape[1] != len(classes_sorted):
            raise ValueError(
                f"proba width {all_proba.shape[1]} != n_unique_classes "
                f"{len(classes_sorted)}",
            )
        top_p = np.max(all_proba, axis=1)
        pred_idx = np.argmax(all_proba, axis=1)
        pred_class = np.asarray([classes_sorted[i] for i in pred_idx])
        correct = (pred_class == all_y).astype(int)
        for p, c in zip(top_p, correct):
            b = min(int(p * deciles), deciles - 1)
            bins[b].append(int(c))

    rows: list[tuple[str, int, float, float]] = []
    for b in sorted(bins.keys()):
        hits = bins[b]
        if len(hits) < min_bin_n:
            continue
        hr = sum(hits) / len(hits)
        midpoint = b / deciles + 0.5 / deciles
        rows.append((
            f"[{b/deciles:.1f},{(b+1)/deciles:.1f})",
            len(hits),
            float(hr),
            float(hr - midpoint),
        ))
    return rows


def _binary_brier(all_y: np.ndarray, all_proba: np.ndarray) -> float:
    """Standard binary Brier — matches sklearn.metrics.brier_score_loss."""
    pos_class = int(max(np.unique(all_y))) if len(all_y) else 0
    y_pos = (all_y == pos_class).astype(int)
    return float(np.mean((all_proba - y_pos) ** 2))


def _multiclass_brier(
    all_y: np.ndarray, all_proba: np.ndarray,
) -> float:
    """Multi-class Brier — mean sum-squared-error across class probas."""
    classes_sorted = sorted(np.unique(all_y).tolist())
    n = len(all_y)
    one_hot = np.zeros((n, len(classes_sorted)), dtype=float)
    for i, c in enumerate(classes_sorted):
        one_hot[:, i] = (all_y == c)
    return float(np.mean(np.sum((all_proba - one_hot) ** 2, axis=1)))


def isotonic_brier_delta(
    all_y: np.ndarray,
    all_proba: np.ndarray,
    *,
    n_classes: int,
) -> tuple[float, float]:
    """Fit isotonic on OOF predictions; return ``(brier_raw, brier_cal)``.

    For binary specs: single ``IsotonicRegression`` on ``(proba, y)``.
    For multi-class specs: per-class one-vs-rest isotonic fit, then
    re-normalize rows so calibrated probas sum to 1 (required by the
    multi-class Brier formulation).

    Note — same caveat as ``§04e``: the isotonic is fit and evaluated on
    the *same* OOF set, so Δ Brier is mildly optimistic. Use the
    decision rubric (FLAT/MINIMAL/MODEST/MEANINGFUL) rather than
    treating the absolute number as a held-out estimate.
    """
    from sklearn.isotonic import IsotonicRegression

    if n_classes == 2:
        brier_raw = _binary_brier(all_y, all_proba)
        pos_class = int(max(np.unique(all_y))) if len(all_y) else 0
        y_pos = (all_y == pos_class).astype(int)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(all_proba, y_pos)
        proba_cal = iso.transform(all_proba)
        brier_cal = float(np.mean((proba_cal - y_pos) ** 2))
        return brier_raw, brier_cal

    brier_raw = _multiclass_brier(all_y, all_proba)
    classes_sorted = sorted(np.unique(all_y).tolist())
    proba_cal = np.zeros_like(all_proba)
    for k_idx, k in enumerate(classes_sorted):
        y_k = (all_y == k).astype(int)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(all_proba[:, k_idx], y_k)
        proba_cal[:, k_idx] = iso.transform(all_proba[:, k_idx])
    # Re-normalize to keep proba a proper distribution after independent
    # per-class isotonic fits (the row sum drifts from 1 otherwise).
    row_sums = proba_cal.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    proba_cal = proba_cal / row_sums
    brier_cal = _multiclass_brier(all_y, proba_cal)
    return brier_raw, brier_cal


def calibration_verdict(delta_brier: float) -> str:
    """Heuristic verdict from Δ Brier (raw - calibrated).

    Matches the §04e thresholds the user already calibrated against:

      < 0.001  →  FLAT       (no recoverable signal)
      < 0.005  →  MINIMAL    (calibration squeezes barely-real signal)
      < 0.015  →  MODEST     (some recoverable structure)
      ≥ 0.015  →  MEANINGFUL (signal worth calibrating)

    Caveat: Δ Brier on the same OOF set the isotonic was fit on is
    optimistic — interpret in conjunction with the decile curve in
    ``calibration_table``. A monotonic curve with even small Δ Brier is
    more trustworthy than a non-monotonic curve with high Δ Brier.
    """
    if not np.isfinite(delta_brier):
        return "INVALID"
    if delta_brier < 0.001:
        return "FLAT"
    if delta_brier < 0.005:
        return "MINIMAL"
    if delta_brier < 0.015:
        return "MODEST"
    return "MEANINGFUL"


def calibration_summary(
    results: dict[str, SpecResult],
    *,
    sort_by_delta: bool = True,
) -> pd.DataFrame:
    """One-row-per-spec calibration summary.

    Columns: ``spec, auc, n_classes, brier_raw, brier_cal, delta_brier,
    verdict, n_oof``.
    """
    rows = []
    for name, r in results.items():
        try:
            all_y, all_proba = aggregate_oof(r)
        except ValueError:
            continue
        if len(all_y) == 0:
            continue
        try:
            brier_raw, brier_cal = isotonic_brier_delta(
                all_y, all_proba, n_classes=r.n_classes,
            )
        except Exception:
            brier_raw = float("nan")
            brier_cal = float("nan")
        delta = brier_raw - brier_cal
        rows.append({
            "spec": name,
            "auc": r.mean_auc,
            "n_classes": r.n_classes,
            "brier_raw": brier_raw,
            "brier_cal": brier_cal,
            "delta_brier": delta,
            "verdict": calibration_verdict(delta),
            "n_oof": int(len(all_y)),
        })
    df = pd.DataFrame(rows)
    if sort_by_delta and len(df) > 0:
        df = df.sort_values("delta_brier", ascending=False).reset_index(drop=True)
    return df
