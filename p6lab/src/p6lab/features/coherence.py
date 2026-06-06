"""
p6lab.features.coherence — Wave 7 Phase 7C

Magnitude-squared coherence between two rolling signals, ported from
``p5-dsp-signal-lab/src/features/coherence.py``. Uses
``scipy.signal.coherence`` under the hood.

Cross-asset feature: measures the frequency-domain correlation between
two instruments' mid series. Peer pairs with high low-frequency
coherence are candidates for correlated-tape cascades; coherence that
shifts sharply flags a regime change between the pair.

Exported:
    magnitude_squared_coherence(x, y, nperseg=256, noverlap=128, fs=1.0)
        → (frequencies, coherence)
    mean_coherence(x, y, freq_range=(0.0, 0.5), ...) → float
    snapshot_coherence_features(x, y, fs=1.0) → dict
"""
from __future__ import annotations

import numpy as np


COHERENCE_FEATURE_NAMES: tuple[str, ...] = (
    "coherence_mean_low_freq",
    "coherence_mean_high_freq",
    "coherence_peak",
)


def magnitude_squared_coherence(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    nperseg: int = 256,
    noverlap: int = 128,
    fs: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frequencies, Cxy)`` where ``Cxy`` is the magnitude-squared
    coherence ∈ [0, 1] per frequency bin. Returns empty arrays when either
    input is too short for the requested ``nperseg``."""
    from scipy.signal import coherence
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = min(xa.size, ya.size)
    if n < nperseg:
        return np.asarray([]), np.asarray([])
    nper = min(int(nperseg), n)
    nover = min(int(noverlap), max(0, nper - 1))
    freqs, cxy = coherence(xa[:n], ya[:n], fs=fs, nperseg=nper, noverlap=nover)
    return freqs, cxy


def mean_coherence(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    freq_range: tuple[float, float] = (0.0, 0.5),
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: int = 128,
) -> float:
    """Mean coherence inside ``freq_range``. Returns 0.0 when coherence
    couldn't be computed (too few samples)."""
    freqs, cxy = magnitude_squared_coherence(
        x, y, nperseg=nperseg, noverlap=noverlap, fs=fs,
    )
    if freqs.size == 0 or cxy.size == 0:
        return 0.0
    lo, hi = freq_range
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    return float(np.mean(cxy[mask]))


def snapshot_coherence_features(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    fs: float = 1.0,
    nperseg: int = 64,
) -> dict[str, float]:
    """Compact 3-scalar summary used by the cross-asset runner (Phase 7A)."""
    freqs, cxy = magnitude_squared_coherence(x, y, nperseg=nperseg, noverlap=nperseg // 2, fs=fs)
    if freqs.size == 0:
        return {name: 0.0 for name in COHERENCE_FEATURE_NAMES}
    half = fs / 4.0
    low = freqs <= half
    high = freqs > half
    return {
        "coherence_mean_low_freq": float(np.mean(cxy[low])) if low.any() else 0.0,
        "coherence_mean_high_freq": float(np.mean(cxy[high])) if high.any() else 0.0,
        "coherence_peak": float(np.max(cxy)) if cxy.size else 0.0,
    }
