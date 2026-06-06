"""Tests for p6lab.features.hilbert_features (Wave 6 Phase 6C)."""
from __future__ import annotations

import numpy as np
import pytest

from p6lab.features.hilbert_features import (
    HILBERT_FEATURE_NAMES,
    analytic_signal,
    hilbert_features,
    instantaneous_amplitude,
    instantaneous_frequency,
    instantaneous_phase,
)


def test_analytic_signal_empty_returns_empty() -> None:
    out = analytic_signal([])
    assert out.size == 0
    assert out.dtype == complex


def test_amplitude_of_sine_approaches_constant() -> None:
    """For a pure sinusoid A·sin(ωt), the analytic envelope is ≈ A."""
    t = np.linspace(0, 10, 1024)
    x = 2.0 * np.sin(2 * np.pi * 3.0 * t)
    amp = instantaneous_amplitude(x)
    # Ignore the windowing transients at both ends
    core = amp[100:-100]
    assert np.all(np.isfinite(core))
    assert np.mean(core) == pytest.approx(2.0, rel=0.05)


def test_instantaneous_frequency_recovers_sine_freq() -> None:
    """For sin(2π·f·t), instantaneous frequency should be ≈ f (in units of fs)."""
    fs = 100.0
    f = 3.0
    n = 512
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f * t)
    freq = instantaneous_frequency(x, fs=fs)
    core = freq[50:-50]
    assert np.mean(core) == pytest.approx(f, rel=0.05)


def test_instantaneous_phase_is_unwrapped() -> None:
    """Unwrapped phase should be monotonically increasing for a sinusoid."""
    t = np.linspace(0, 10, 512)
    x = np.sin(2 * np.pi * 1.0 * t)
    phase = instantaneous_phase(x)
    # Allow small backtracking at the ends; interior should be monotonic
    diffs = np.diff(phase[20:-20])
    assert (diffs >= -1e-6).mean() > 0.95


def test_hilbert_features_short_input_returns_zeros() -> None:
    snap = hilbert_features([1.0])
    for k in HILBERT_FEATURE_NAMES:
        assert snap[k] == 0.0


def test_hilbert_features_returns_all_scalars() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    snap = hilbert_features(x, fs=1.0)
    for k in HILBERT_FEATURE_NAMES:
        assert k in snap
        assert np.isfinite(snap[k])
