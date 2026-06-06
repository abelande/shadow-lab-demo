"""Tests for p6lab.features.coherence (Wave 7 Phase 7C)."""
from __future__ import annotations

import numpy as np
import pytest

from p6lab.features.coherence import (
    COHERENCE_FEATURE_NAMES,
    magnitude_squared_coherence,
    mean_coherence,
    snapshot_coherence_features,
)


def test_two_coherent_sinusoids_have_high_coherence() -> None:
    n = 2048
    t = np.arange(n) / 256.0
    x = np.sin(2 * np.pi * 5.0 * t)
    y = np.sin(2 * np.pi * 5.0 * t + 0.5)  # phase-shifted — still coherent
    freqs, cxy = magnitude_squared_coherence(x, y, nperseg=256, fs=256.0)
    assert freqs.size > 0
    # Coherence at the shared frequency should be near 1
    peak = cxy[np.argmax(cxy)]
    assert peak >= 0.9


def test_uncorrelated_noise_has_low_mean_coherence() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2048)
    y = rng.normal(0, 1, 2048)
    mc = mean_coherence(x, y, nperseg=256, fs=1.0, freq_range=(0.0, 0.5))
    assert 0.0 <= mc <= 1.0
    assert mc < 0.5


def test_empty_inputs_return_zero() -> None:
    freqs, cxy = magnitude_squared_coherence([], [], nperseg=64)
    assert freqs.size == 0
    assert cxy.size == 0
    assert mean_coherence([], [], nperseg=64) == 0.0


def test_short_inputs_return_zero() -> None:
    freqs, cxy = magnitude_squared_coherence([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], nperseg=64)
    assert freqs.size == 0


def test_snapshot_returns_expected_feature_names() -> None:
    x = np.sin(np.arange(512) * 0.1)
    y = np.sin(np.arange(512) * 0.1 + 0.3)
    snap = snapshot_coherence_features(x, y, fs=1.0, nperseg=64)
    for k in COHERENCE_FEATURE_NAMES:
        assert k in snap
        assert 0.0 <= snap[k] <= 1.0
