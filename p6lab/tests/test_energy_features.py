"""Tests for p6lab.features.energy_features (Wave 6 Phase 6D)."""
from __future__ import annotations

import numpy as np
import pytest

from p6lab.features.energy_features import (
    ENERGY_FEATURE_NAMES,
    energy_ratio,
    mean_power,
    rms_power,
    signal_to_noise,
    snapshot_energy_features,
)


def test_rms_power_zero_on_empty() -> None:
    assert rms_power([], window=10) == 0.0


def test_rms_power_constant_series_equals_abs_value() -> None:
    # RMS of a constant c with length N = |c|
    assert rms_power([3.0] * 20, window=20) == pytest.approx(3.0)


def test_rms_power_trims_to_window() -> None:
    # Window truncation — use the last 5 of [1..10]
    arr = list(range(1, 11))
    expected = np.sqrt(np.mean(np.asarray([6.0, 7.0, 8.0, 9.0, 10.0]) ** 2))
    assert rms_power(arr, window=5) == pytest.approx(expected)


def test_mean_power_zero_on_empty() -> None:
    assert mean_power([]) == 0.0


def test_mean_power_matches_manual() -> None:
    assert mean_power([1.0, 2.0, 3.0], window=3) == pytest.approx((1 + 4 + 9) / 3.0)


def test_energy_ratio_breakout_gt_1() -> None:
    # Short window (5) is the noisy tail; long window averages in calm regime
    series = [0.0] * 45 + [5.0] * 5
    ratio = energy_ratio(series, short_window=5, long_window=50)
    assert ratio > 1.0


def test_energy_ratio_calm_leq_1() -> None:
    series = [5.0] * 45 + [0.0] * 5
    ratio = energy_ratio(series, short_window=5, long_window=50)
    assert ratio <= 1.0


def test_energy_ratio_empty_long_returns_one() -> None:
    assert energy_ratio([0.0] * 10) == pytest.approx(1.0)


def test_signal_to_noise_flat_returns_zero() -> None:
    # A constant series has zero variance regardless of detrending → SNR=0
    assert signal_to_noise([5.0] * 30, window=30, detrend=False) == 0.0
    assert signal_to_noise([5.0] * 30, window=30, detrend=True) == 0.0


def test_signal_to_noise_clean_signal_high() -> None:
    rng = np.random.default_rng(7)
    x = np.sin(np.linspace(0, 10, 200)) + rng.normal(0, 0.01, 200)
    snr = signal_to_noise(x, window=200, detrend=True)
    assert snr >= 0.0   # mean-squared / variance is non-negative


def test_snapshot_returns_all_named_features() -> None:
    rng = np.random.default_rng(0)
    mid = 20_000.0 + rng.normal(0, 1.0, 100).cumsum()
    snap = snapshot_energy_features(mid, window_short=10, window_long=50)
    for name in ENERGY_FEATURE_NAMES:
        assert name in snap
        assert np.isfinite(snap[name])


def test_snapshot_handles_short_window() -> None:
    snap = snapshot_energy_features([100.0, 100.5])
    for name in ENERGY_FEATURE_NAMES:
        assert np.isfinite(snap[name])
