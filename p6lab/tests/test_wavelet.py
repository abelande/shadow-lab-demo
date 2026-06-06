"""Tests for p6lab.features.wavelet (Wave 6 Phase 6B)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pywt")

from p6lab.features.wavelet import (
    WAVELET_FEATURE_NAMES,
    WaveletState,
    cwt_energies,
    snapshot_wavelet_features,
    update_wavelet,
)


def test_warmup_returns_zeros() -> None:
    state = WaveletState()
    snap = snapshot_wavelet_features(state, fs=1.0)
    for k in WAVELET_FEATURE_NAMES:
        assert snap[k] == 0.0


def test_cwt_energies_of_sinusoid_peaks_at_expected_scale() -> None:
    """A 5s-period sinusoid should produce the largest energy at the scale
    corresponding to 5s (or the closest scale)."""
    fs = 1.0
    n = 512
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * (1.0 / 5.0) * t)  # period = 5s
    scales = np.asarray([2.0, 5.0, 15.0, 60.0])
    energies = cwt_energies(x, scales=scales)
    # 5s scale should carry more energy than the 60s scale
    assert energies[1] > energies[3]
    assert np.all(energies >= 0.0)


def test_update_and_snapshot_runs_end_to_end() -> None:
    state = WaveletState()
    rng = np.random.default_rng(2)
    mid = 20_000.0
    for i in range(80):
        mid += rng.normal(0, 0.25)
        update_wavelet(state, ts_ms=i * 100, mid=mid)
    snap = snapshot_wavelet_features(state, fs=1.0)
    for k in WAVELET_FEATURE_NAMES:
        assert k in snap
        assert np.isfinite(snap[k])


def test_constant_mid_returns_zero_energy() -> None:
    state = WaveletState()
    for i in range(80):
        update_wavelet(state, ts_ms=i * 100, mid=20_000.0)
    snap = snapshot_wavelet_features(state, fs=1.0)
    # All scales should be 0 for constant input (std=0 short-circuits)
    for k in WAVELET_FEATURE_NAMES:
        assert snap[k] == 0.0


def test_out_of_order_samples_ignored() -> None:
    state = WaveletState()
    update_wavelet(state, ts_ms=1_000, mid=20_000.0)
    update_wavelet(state, ts_ms=500, mid=19_999.0)   # before last; ignored
    assert len(state.samples) == 1


def test_cwt_energies_empty_input() -> None:
    energies = cwt_energies(np.asarray([]), scales=np.asarray([5.0, 15.0]))
    assert energies.shape == (2,)
    assert np.all(energies == 0.0)


# Wave 8.5-H: adaptive scales tests
def test_wave_85_h_adaptive_scales_identifies_dominant_frequency() -> None:
    """A 3-Hz sinusoid should produce an adaptive scale near fs / 3."""
    from p6lab.features.wavelet import adaptive_scales
    fs = 20.0
    n = 400
    t = np.arange(n) / fs
    # Strong 3 Hz component + small noise
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 3.0 * t) + rng.normal(0, 0.05, n)
    scales = adaptive_scales(x, fs=fs, n_scales=6)
    # Expected scale for 3 Hz: fs/3 ≈ 6.67. Check the closest scale
    # in the selection is within 2.0 of target.
    target = fs / 3.0
    closest = scales[np.argmin(np.abs(scales - target))]
    assert abs(closest - target) < 2.0


def test_wave_85_h_adaptive_scales_fallback_on_short_input() -> None:
    from p6lab.features.wavelet import adaptive_scales, _SCALE_LABELS
    fs = 20.0
    # Too short — must fall back to the fixed defaults
    x = np.arange(4, dtype=float)
    scales = adaptive_scales(x, fs=fs, n_scales=len(_SCALE_LABELS))
    expected = np.asarray([sec * fs for _, sec in _SCALE_LABELS], dtype=float)
    np.testing.assert_array_equal(scales, expected)


def test_wave_85_h_adaptive_scales_fallback_on_constant_signal() -> None:
    from p6lab.features.wavelet import adaptive_scales, _SCALE_LABELS
    fs = 1.0
    x = np.full(100, 3.14, dtype=float)   # constant
    scales = adaptive_scales(x, fs=fs, n_scales=len(_SCALE_LABELS))
    expected = np.asarray([sec * fs for _, sec in _SCALE_LABELS], dtype=float)
    np.testing.assert_array_equal(scales, expected)


def test_wave_85_h_snapshot_adaptive_mode() -> None:
    """Smoke: snapshot with scales='adaptive' runs end-to-end and
    produces the canonical feature keys."""
    state = WaveletState()
    rng = np.random.default_rng(2)
    mid = 20_000.0
    for i in range(80):
        mid += rng.normal(0, 0.25)
        update_wavelet(state, ts_ms=i * 100, mid=mid)
    snap = snapshot_wavelet_features(state, fs=1.0, scales="adaptive")
    for k in WAVELET_FEATURE_NAMES:
        assert k in snap
        assert np.isfinite(snap[k])


def test_wave_85_h_default_path_unchanged() -> None:
    """Regression: calling snapshot_wavelet_features with no scales
    kwarg (the pre-8.5 API) produces identical output to explicit
    scales=None."""
    state = WaveletState()
    rng = np.random.default_rng(3)
    mid = 20_000.0
    for i in range(80):
        mid += rng.normal(0, 0.25)
        update_wavelet(state, ts_ms=i * 100, mid=mid)
    default = snapshot_wavelet_features(state, fs=1.0)
    explicit_none = snapshot_wavelet_features(state, fs=1.0, scales=None)
    for k in WAVELET_FEATURE_NAMES:
        assert default[k] == explicit_none[k]
