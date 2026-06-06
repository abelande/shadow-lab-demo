"""Tests for p6lab.features.spectral_force (Wave 6 Phase 6A)."""
from __future__ import annotations

import pytest

from p6lab.features.spectral_force import (
    SPECTRAL_FORCE_FEATURE_NAMES,
    SpectralForceState,
    snapshot_spectral_force_features,
    update_spectral_force,
)


def test_snapshot_on_empty_state_returns_zeros() -> None:
    state = SpectralForceState()
    snap = snapshot_spectral_force_features(state)
    for k in SPECTRAL_FORCE_FEATURE_NAMES:
        assert snap[k] == 0.0


def test_snapshot_returns_named_features() -> None:
    state = SpectralForceState()
    update_spectral_force(
        state,
        ts_ms=1_000,
        volume_delta_series=[1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0],
    )
    snap = snapshot_spectral_force_features(state)
    for k in SPECTRAL_FORCE_FEATURE_NAMES:
        assert k in snap
        assert isinstance(snap[k], float)


def test_total_energy_equals_sum_of_bands() -> None:
    state = SpectralForceState()
    update_spectral_force(
        state,
        ts_ms=1_000,
        volume_delta_series=[1.0, -0.5, 0.8, -0.3, 0.7, 0.1, -0.2, 0.4, 0.6, -0.9],
    )
    snap = snapshot_spectral_force_features(state)
    parts = (
        snap["force_band_energy_institutional"]
        + snap["force_band_energy_fund"]
        + snap["force_band_energy_daytrading"]
        + snap["force_band_energy_hft"]
    )
    assert snap["force_total_energy"] == pytest.approx(parts, rel=1e-6)


def test_empty_series_keeps_zeros() -> None:
    state = SpectralForceState()
    update_spectral_force(state, ts_ms=1_000, volume_delta_series=[])
    snap = snapshot_spectral_force_features(state)
    assert snap["force_total_energy"] == 0.0


def test_history_length_grows_then_caps() -> None:
    state = SpectralForceState()
    for i in range(50):
        update_spectral_force(
            state, ts_ms=i * 100,
            volume_delta_series=[0.1 * i, -0.2 * i, 0.3 * i, 0.4, -0.5],
        )
    assert len(state.history) == 50
