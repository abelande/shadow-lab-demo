"""
p6lab.features.energy_features — Wave 6 Phase 6D

Rolling signal-energy utilities ported from p5-dsp-signal-lab's
``features/energy.py``. These are pure numerical helpers with no market
assumptions — callers pipe in any 1D series (mid, volume, OFI, …) and
get back power / RMS / energy-ratio features.

Exported:
    RMS_POWER_FEATURE_NAMES      tuple[str, ...]
    rms_power(series, window)    → float
    mean_power(series, window)   → float
    energy_ratio(short, long)    → float
    signal_to_noise(series, window, detrend=True) → float
    snapshot_energy_features(mid_window, window_short, window_long) → dict
"""
from __future__ import annotations

import numpy as np

ENERGY_FEATURE_NAMES: tuple[str, ...] = (
    "rms_power",
    "mean_power_short",
    "mean_power_long",
    "energy_ratio",
    "signal_to_noise",
)


def rms_power(series: np.ndarray | list[float], window: int = 20) -> float:
    """Root-mean-square of the last ``window`` samples. Returns 0.0 on empty
    input — never NaN."""
    arr = _tail(series, window)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def mean_power(series: np.ndarray | list[float], window: int = 50) -> float:
    """Mean of squared values over the last ``window`` samples."""
    arr = _tail(series, window)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr * arr))


def energy_ratio(
    series: np.ndarray | list[float],
    short_window: int = 10,
    long_window: int = 50,
) -> float:
    """``mean_power(short) / mean_power(long)`` — > 1 ⇒ breakout, < 1 ⇒ fade.

    Returns 1.0 when the long window is empty / zero so downstream code
    doesn't divide by zero."""
    short = mean_power(series, short_window)
    long = mean_power(series, long_window)
    if long <= 0.0:
        return 1.0
    return float(short / long)


def signal_to_noise(
    series: np.ndarray | list[float],
    window: int = 50,
    *,
    detrend: bool = True,
) -> float:
    """Rolling SNR proxy: ``mean² / variance`` of the last ``window`` samples.
    Dimensionless. Higher ⇒ cleaner signal. Returns 0.0 when the variance
    is zero (flat series)."""
    arr = _tail(series, window)
    if arr.size < 2:
        return 0.0
    if detrend:
        arr = arr - np.mean(arr)
    var = float(np.var(arr))
    if var <= 0.0:
        return 0.0
    mean = float(np.mean(arr))
    return (mean * mean) / var


def snapshot_energy_features(
    mid_window: np.ndarray | list[float],
    *,
    window_short: int = 10,
    window_long: int = 50,
) -> dict[str, float]:
    """Bundle the 5 scalars under ``ENERGY_FEATURE_NAMES``."""
    arr = _tail(mid_window, window_long)
    if arr.size > 1:
        returns = np.diff(arr)
    else:
        returns = np.asarray([0.0])
    return {
        "rms_power": rms_power(returns, window_long),
        "mean_power_short": mean_power(returns, window_short),
        "mean_power_long": mean_power(returns, window_long),
        "energy_ratio": energy_ratio(returns, window_short, window_long),
        "signal_to_noise": signal_to_noise(arr, window_long, detrend=True),
    }


def _tail(series: np.ndarray | list[float], window: int) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    if window <= 0:
        return np.asarray([])
    return arr[-int(window):]
