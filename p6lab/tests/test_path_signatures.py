"""Tests for p6lab.features.path_signatures (Wave 7 Phase 7F)."""
from __future__ import annotations

import numpy as np
import pytest

from p6lab.features.path_signatures import (
    DEFAULT_SIG_COMPONENTS,
    PATH_SIGNATURE_FEATURE_NAMES,
    compute_signature,
    lead_lag_augmentation,
    signature_dim,
    snapshot_path_signature_features,
    time_augmentation,
)


def test_signature_dim_formula() -> None:
    assert signature_dim(2, 1) == 2
    assert signature_dim(2, 2) == 2 + 4
    assert signature_dim(3, 3) == 3 + 9 + 27


def test_compute_signature_short_path_returns_zeros() -> None:
    sig = compute_signature(np.asarray([[1.0]]), depth=2)
    assert np.all(sig == 0.0)


def test_compute_signature_level1_matches_endpoint() -> None:
    """Level 1 of an un-normalized signature is the path's displacement."""
    path = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    sig = compute_signature(path, depth=1, normalize=False)
    # Sum of increments in x = 3.0; in y = 4.0
    assert sig[0] == pytest.approx(3.0)
    assert sig[1] == pytest.approx(4.0)


def test_compute_signature_normalized() -> None:
    path = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    sig = compute_signature(path, depth=2, normalize=True)
    assert np.linalg.norm(sig) == pytest.approx(1.0, rel=1e-6)


def test_time_augmentation_shape() -> None:
    t_x = time_augmentation(np.asarray([1.0, 2.0, 3.0]))
    assert t_x.shape == (3, 2)
    # First column is normalized time in [0, 1]
    assert t_x[0, 0] == 0.0
    assert t_x[-1, 0] == 1.0


def test_lead_lag_augmentation_shape() -> None:
    ll = lead_lag_augmentation(np.asarray([10.0, 20.0, 30.0]))
    assert ll.shape == (2, 2)
    # Second row: (lead, lag) = (30, 20)
    assert ll[1, 0] == 30.0
    assert ll[1, 1] == 20.0


def test_snapshot_returns_fixed_feature_count() -> None:
    rng = np.random.default_rng(0)
    mid = 20_000.0 + rng.normal(0, 1.0, 100).cumsum()
    snap = snapshot_path_signature_features(mid, depth=2, augmentation="time")
    assert len(snap) == DEFAULT_SIG_COMPONENTS
    for k in PATH_SIGNATURE_FEATURE_NAMES:
        assert k in snap
        assert np.isfinite(snap[k])


def test_snapshot_lead_lag_augmentation() -> None:
    rng = np.random.default_rng(1)
    mid = 20_000.0 + rng.normal(0, 1.0, 50).cumsum()
    snap = snapshot_path_signature_features(mid, depth=2, augmentation="lead_lag")
    assert len(snap) == DEFAULT_SIG_COMPONENTS


def test_snapshot_short_input_returns_zeros() -> None:
    snap = snapshot_path_signature_features([100.0])
    for k in PATH_SIGNATURE_FEATURE_NAMES:
        assert snap[k] == 0.0
