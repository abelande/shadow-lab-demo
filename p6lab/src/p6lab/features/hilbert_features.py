"""
p6lab.features.hilbert_features — Wave 6 Phase 6C

Hilbert-transform instantaneous amplitude / phase / frequency, ported
from ``p5-dsp-signal-lab/src/spectral/hilbert.py``. Builds on
``scipy.signal.hilbert`` so we stay dependency-light (scipy already sits
in the lab extras).

Exported:
    HILBERT_FEATURE_NAMES            tuple[str, ...]
    analytic_signal(series)          → complex[]
    instantaneous_amplitude(series)  → float[]
    instantaneous_phase(series)      → float[]     (unwrapped)
    instantaneous_frequency(series, fs=1.0) → float[]
    hilbert_features(series, fs=1.0) → dict[str, float]  (scalar summary)
"""
from __future__ import annotations

import numpy as np


HILBERT_FEATURE_NAMES: tuple[str, ...] = (
    "hilbert_amplitude",
    "hilbert_phase",
    "hilbert_instantaneous_frequency",
)


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------


def analytic_signal(series: np.ndarray | list[float]) -> np.ndarray:
    """Return the analytic signal ``x(t) + i · H{x}(t)`` with the mean
    subtracted so DC doesn't swamp the envelope."""
    from scipy.signal import hilbert
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=complex)
    return hilbert(arr - np.mean(arr))


def instantaneous_amplitude(series: np.ndarray | list[float]) -> np.ndarray:
    analytic = analytic_signal(series)
    return np.abs(analytic)


def instantaneous_phase(series: np.ndarray | list[float]) -> np.ndarray:
    """Unwrapped phase of the analytic signal. Same length as input."""
    analytic = analytic_signal(series)
    return np.unwrap(np.angle(analytic))


def instantaneous_frequency(
    series: np.ndarray | list[float], *, fs: float = 1.0
) -> np.ndarray:
    """Δphase / (2π) × ``fs``. Length N-1 (first sample has no predecessor)."""
    phase = instantaneous_phase(series)
    if phase.size < 2:
        return np.asarray([])
    return np.diff(phase) / (2.0 * np.pi) * float(fs)


# ---------------------------------------------------------------------------
# Scalar summary for the live feature snapshot
# ---------------------------------------------------------------------------


def hilbert_features(
    series: np.ndarray | list[float], *, fs: float = 1.0,
) -> dict[str, float]:
    """Produce the 3 scalar feature values for the feature matrix.

    - ``hilbert_amplitude``: the most-recent envelope value.
    - ``hilbert_phase``: the most-recent unwrapped phase (radians).
    - ``hilbert_instantaneous_frequency``: the most-recent frequency
      estimate, in the same units as ``fs``.

    Returns zero values on empty / too-short input — never NaN.
    """
    arr = np.asarray(series, dtype=float)
    if arr.size < 3:
        return {
            "hilbert_amplitude": 0.0,
            "hilbert_phase": 0.0,
            "hilbert_instantaneous_frequency": 0.0,
        }
    amp = instantaneous_amplitude(arr)
    phase = instantaneous_phase(arr)
    freq = instantaneous_frequency(arr, fs=fs)
    return {
        "hilbert_amplitude": float(amp[-1]) if amp.size else 0.0,
        "hilbert_phase": float(phase[-1]) if phase.size else 0.0,
        "hilbert_instantaneous_frequency": float(freq[-1]) if freq.size else 0.0,
    }
