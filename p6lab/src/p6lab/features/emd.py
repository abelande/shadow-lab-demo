"""
p6lab.features.emd — Wave 7 Phase 7D + Wave 8.5-C

Empirical Mode Decomposition with a fixed-width IMF aggregator so we
always emit a tabular feature vector regardless of how many IMFs the
sifting produces.

We prefer PyEMD (``pip install EMD-signal``) at runtime. When not
available, we fall back to a minimal classical-sifting implementation
that's accurate enough for feature extraction.

**Degradation mode (Wave 8.5-C).**
The classical-sifting fallback is a **degradation mode** — it uses
piecewise-linear envelope interpolation (not cubic spline) and a
heuristic stopping rule, so on pathological signals it may
under-decompose. Production installs should pull ``EMD-signal`` via
the ``[lab]`` pyproject.toml extras. When classical runs, a single
``logger.warning`` fires per process so operators know they're on
the degraded path.

Exported:
    EMD_FEATURE_NAMES               tuple[str, ...]
    decompose(signal, max_imfs=5) → list[np.ndarray]
    aggregate_imfs(imfs) → dict[str, float]
    snapshot_emd_features(mid_series, max_imfs=5) → dict
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Wave 8.5-C: one-shot warning sentinel. Module-level so the warning
# fires exactly once per Python process, regardless of how many
# decompose() calls happen. Reset in tests via a fixture that
# monkeypatches this back to False.
_WARNED_CLASSICAL_FALLBACK: bool = False


MAX_IMFS_EMITTED: int = 3


EMD_FEATURE_NAMES: tuple[str, ...] = (
    *(f"emd_energy_imf_{i}" for i in range(1, MAX_IMFS_EMITTED + 1)),
    "emd_mean_frequency_imf_1",
    "emd_residual_trend_slope",
)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def decompose(
    signal: np.ndarray | list[float],
    *,
    max_imfs: int = 5,
) -> list[np.ndarray]:
    """Return the list of IMFs (high-freq → low-freq) plus the residue.

    Tries PyEMD first for accuracy; falls back to classical sifting
    when PyEMD isn't on the interpreter.
    """
    arr = np.asarray(signal, dtype=float)
    if arr.size < 8:
        return []
    try:
        from PyEMD import EMD   # type: ignore[import-not-found]
        emd = EMD()
        imfs = emd.emd(arr, max_imf=max_imfs)
        return [np.asarray(imf, dtype=float) for imf in imfs]
    except ImportError:
        # Wave 8.5-C: fire a WARNING (not DEBUG) exactly once per
        # process. Classical sifting is a degradation mode — log loud
        # enough that operators running without [lab] extras notice,
        # but don't spam every call.
        global _WARNED_CLASSICAL_FALLBACK
        if not _WARNED_CLASSICAL_FALLBACK:
            logger.warning(
                "PyEMD (EMD-signal) not installed; falling back to "
                "classical sifting (degradation mode — piecewise-linear "
                "envelope, heuristic stopping). Install `EMD-signal` via "
                "`pip install -e .[lab]` for production-grade accuracy."
            )
            _WARNED_CLASSICAL_FALLBACK = True
        return _classical_sift(arr, max_imfs=max_imfs)


def aggregate_imfs(imfs: list[np.ndarray]) -> dict[str, float]:
    """Collapse an IMF list into the ``EMD_FEATURE_NAMES`` feature vector.

    - Per-IMF energy = ``sum(x²)``; we emit the top-3 IMF energies.
    - Mean frequency of IMF-1 = average zero-crossing rate / 2.
    - Residual trend slope = OLS slope of the last IMF (slow component).
    """
    result: dict[str, float] = {name: 0.0 for name in EMD_FEATURE_NAMES}
    if not imfs:
        return result
    for i, imf in enumerate(imfs[:MAX_IMFS_EMITTED]):
        result[f"emd_energy_imf_{i + 1}"] = float(np.sum(imf * imf))

    first = imfs[0]
    if first.size > 1:
        zc = np.sum(np.diff(np.sign(first)) != 0)
        result["emd_mean_frequency_imf_1"] = float(zc / (2.0 * first.size))

    trend = imfs[-1]
    if trend.size > 1:
        x = np.arange(trend.size, dtype=float)
        x_mean = x.mean()
        y_mean = trend.mean()
        denom = float(np.sum((x - x_mean) ** 2))
        if denom > 0.0:
            slope = float(np.sum((x - x_mean) * (trend - y_mean)) / denom)
        else:
            slope = 0.0
        result["emd_residual_trend_slope"] = slope
    return result


def snapshot_emd_features(
    mid_series: np.ndarray | list[float],
    *,
    max_imfs: int = 5,
) -> dict[str, float]:
    """Decompose + aggregate in one call for the live feature matrix."""
    imfs = decompose(mid_series, max_imfs=max_imfs)
    return aggregate_imfs(imfs)


# ---------------------------------------------------------------------------
# Fallback sifting (no external deps)
# ---------------------------------------------------------------------------


def _classical_sift(
    signal: np.ndarray,
    *,
    max_imfs: int = 5,
    max_iterations: int = 20,
) -> list[np.ndarray]:
    """Minimal EMD: iterate ``_sift_once`` until the residue is monotonic
    or ``max_imfs`` IMFs have been extracted."""
    residue = signal.copy()
    imfs: list[np.ndarray] = []
    for _ in range(max_imfs):
        if _is_monotonic(residue):
            break
        h = residue.copy()
        for _ in range(max_iterations):
            maxima_idx, minima_idx = _local_extrema(h)
            if maxima_idx.size < 2 or minima_idx.size < 2:
                break
            upper = _interpolate_envelope(h, maxima_idx)
            lower = _interpolate_envelope(h, minima_idx)
            if upper is None or lower is None:
                break
            mean = 0.5 * (upper + lower)
            new_h = h - mean
            # Stop when the mean is close to zero
            if np.max(np.abs(mean)) < 1e-4 * max(np.max(np.abs(h)), 1e-9):
                h = new_h
                break
            h = new_h
        imfs.append(h)
        residue = residue - h
    imfs.append(residue)   # last element is the trend
    return imfs


def _is_monotonic(arr: np.ndarray) -> bool:
    d = np.diff(arr)
    return bool(np.all(d >= 0) or np.all(d <= 0))


def _local_extrema(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(arr)
    maxima: list[int] = []
    minima: list[int] = []
    for i in range(1, d.size):
        if d[i - 1] > 0 and d[i] <= 0:
            maxima.append(i)
        elif d[i - 1] < 0 and d[i] >= 0:
            minima.append(i)
    return np.asarray(maxima, dtype=int), np.asarray(minima, dtype=int)


def _interpolate_envelope(
    arr: np.ndarray, idx: np.ndarray,
) -> np.ndarray | None:
    """Piecewise-linear envelope passed through extrema; fast stand-in
    for the true cubic-spline envelope."""
    if idx.size < 2:
        return None
    x = idx
    y = arr[idx]
    target = np.arange(arr.size)
    return np.interp(target, x, y)
