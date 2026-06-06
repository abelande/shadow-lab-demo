"""
p6lab.features.wavelet — Wave 6 Phase 6B

Continuous-Wavelet-Transform (CWT) features over a rolling mid-price
window. Produces per-scale energy for 6 log-spaced scales corresponding
to 2s / 5s / 15s / 60s / 120s / 300s at 1Hz sampling — roughly HFT
through swing in trading parlance.

Implementation uses PyWavelets' ``cwt`` with the Morlet wavelet. A
``WaveletState`` circular buffer keeps the most recent mids and their
timestamps so we can resample onto a uniform grid before the CWT
(``cwt`` requires evenly-spaced samples).

Exported:
    WAVELET_FEATURE_NAMES      tuple[str, ...]
    WaveletState               dataclass
    update_wavelet(state, ts_ms, mid)
    snapshot_wavelet_features(state, fs=1.0) → dict
    cwt_energies(mid_resampled, scales, wavelet="morl") → np.ndarray
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

logger = logging.getLogger(__name__)


BUFFER_SEC = 300                 # keep 5 min of mids
MIN_SAMPLES_FOR_CWT = 32
RESAMPLE_HZ = 1.0                # 1 sample / second post-resample
_SCALE_LABELS: tuple[tuple[str, float], ...] = (
    ("scale_2s", 2.0),
    ("scale_5s", 5.0),
    ("scale_15s", 15.0),
    ("scale_60s", 60.0),
    ("scale_120s", 120.0),
    ("scale_300s", 300.0),
)


WAVELET_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"wavelet_energy_{lbl}" for (lbl, _) in _SCALE_LABELS
)


@dataclass
class WaveletState:
    """Rolling (ts_ms, mid) buffer. Sized for BUFFER_SEC at ≤100 Hz."""
    samples: Deque[tuple[int, float]] = field(
        default_factory=lambda: deque(maxlen=int(BUFFER_SEC * 100))
    )

    def reset(self) -> None:
        self.samples.clear()


def update_wavelet(state: WaveletState, *, ts_ms: int, mid: float) -> None:
    """Append one (ts, mid) sample. Ring buffer trims oldest."""
    if state.samples and ts_ms < state.samples[-1][0]:
        # Discard out-of-order samples — live feeds occasionally re-emit.
        return
    state.samples.append((int(ts_ms), float(mid)))


def snapshot_wavelet_features(
    state: WaveletState,
    *,
    fs: float = RESAMPLE_HZ,
    scales: np.ndarray | str | None = None,
) -> dict[str, float]:
    """Resample the rolling buffer onto a uniform grid and emit per-scale
    mean-energy scalars. Returns zeros on warmup / insufficient data /
    constant mid.

    Wave 8.5-H: ``scales`` kwarg accepts:
      - ``None`` (default): fixed scales matching ``_SCALE_LABELS`` —
        preserves pre-8.5 behavior for backward compat.
      - ``"adaptive"``: scales derived from the signal's FFT peaks —
        per-symbol frequency content drives scale selection.
      - ``np.ndarray``: explicit scale vector provided by caller.

    When an explicit or adaptive scale vector is used, output feature
    keys remain ``wavelet_energy_scale_{label}`` using the fixed labels
    from ``_SCALE_LABELS`` (so downstream feature-store wiring is
    unchanged); the labels become a position/order tag rather than a
    time-unit tag.
    """
    zero = {name: 0.0 for name in WAVELET_FEATURE_NAMES}
    if len(state.samples) < MIN_SAMPLES_FOR_CWT:
        return zero

    ts = np.array([s[0] for s in state.samples], dtype=float)
    mid = np.array([s[1] for s in state.samples], dtype=float)
    if mid.std() <= 1e-9:
        return zero

    # Resample onto a uniform grid at ``fs`` Hz using linear interpolation.
    total_sec = (ts[-1] - ts[0]) / 1000.0
    if total_sec <= 0:
        return zero
    n_grid = max(MIN_SAMPLES_FOR_CWT, int(total_sec * fs) + 1)
    grid_ts = np.linspace(ts[0], ts[-1], n_grid)
    grid_mid = np.interp(grid_ts, ts, mid)

    if scales is None:
        scale_vec = np.asarray([sec * fs for _, sec in _SCALE_LABELS], dtype=float)
    elif isinstance(scales, str) and scales == "adaptive":
        scale_vec = adaptive_scales(grid_mid, fs=fs, n_scales=len(_SCALE_LABELS))
    else:
        scale_vec = np.asarray(scales, dtype=float)

    try:
        energies = cwt_energies(grid_mid, scales=scale_vec)
    except Exception:
        logger.exception("wavelet cwt failed; returning zeros")
        return zero

    return {
        f"wavelet_energy_{lbl}": float(energies[i])
        for i, (lbl, _) in enumerate(_SCALE_LABELS)
    }


def adaptive_scales(
    signal: np.ndarray, *, fs: float = RESAMPLE_HZ, n_scales: int = 6,
) -> np.ndarray:
    """Wave 8.5-H: data-driven scale selection via FFT peaks.

    Compute the magnitude spectrum, pick the top ``n_scales`` positive
    frequencies, convert each to a wavelet scale as ``fs / freq``. When
    the signal is too short for a meaningful FFT or all frequencies are
    zero, falls back to the fixed scales in ``_SCALE_LABELS``.

    Returns
    -------
    np.ndarray
        Sorted scale vector (ascending), same shape as ``_SCALE_LABELS``.
    """
    default = np.asarray([sec * fs for _, sec in _SCALE_LABELS], dtype=float)
    arr = np.asarray(signal, dtype=float)
    if arr.size < 8:
        return default
    arr = arr - np.mean(arr)
    # Real FFT — we only care about positive frequencies
    mag = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / fs)
    if mag.size <= 1 or freqs[1:].size == 0:
        return default
    # Drop the DC bin
    mag_nz = mag[1:]
    freqs_nz = freqs[1:]
    if not np.any(mag_nz > 0):
        return default
    # Top-N peak frequencies by magnitude
    k = min(n_scales, mag_nz.size)
    top_idx = np.argpartition(mag_nz, -k)[-k:]
    peak_freqs = freqs_nz[top_idx]
    # Drop zeros and guard against divide-by-zero
    peak_freqs = peak_freqs[peak_freqs > 1e-9]
    if peak_freqs.size == 0:
        return default
    scales = np.sort(fs / peak_freqs)
    # Pad with fixed fallback if fewer than n_scales peaks
    if scales.size < n_scales:
        pad = default[: n_scales - scales.size]
        scales = np.sort(np.concatenate([scales, pad]))
    return scales[:n_scales]


def cwt_energies(
    signal: np.ndarray,
    *,
    scales: np.ndarray,
    wavelet: str = "morl",
) -> np.ndarray:
    """Mean |coeff|² per scale for the given 1D signal.

    Mean is subtracted before transformation so DC doesn't swamp the
    low-frequency scales. Returns ``np.zeros(len(scales))`` when PyWavelets
    isn't available so downstream never sees NaN.
    """
    try:
        import pywt
    except ImportError:
        logger.warning("pywt not installed; wavelet features return zeros")
        return np.zeros(len(scales), dtype=float)

    arr = np.asarray(signal, dtype=float)
    if arr.size == 0:
        return np.zeros(len(scales), dtype=float)
    arr = arr - np.mean(arr)
    coeffs, _ = pywt.cwt(arr, scales=scales, wavelet=wavelet)
    # |coeffs| shape: (len(scales), len(signal))
    return np.mean(np.abs(coeffs) ** 2, axis=1)
