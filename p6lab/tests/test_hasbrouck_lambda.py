"""Tests for p6lab.features.hasbrouck_lambda (Wave 6 Phase 6F)."""
from __future__ import annotations

import pytest

from p6lab.features.hasbrouck_lambda import (
    HASBROUCK_FEATURE_NAMES,
    HasbroucksLambdaState,
    compute_hasbrouck_lambda,
    snapshot_hasbrouck_features,
    update_hasbrouck_lambda,
)


def _run(state: HasbroucksLambdaState, steps: list[tuple[int, float, float]]) -> None:
    for ts_ms, mid, flow in steps:
        update_hasbrouck_lambda(state, ts_ms=ts_ms, mid=mid, signed_vol=flow)


def test_warmup_returns_zero() -> None:
    state = HasbroucksLambdaState()
    assert compute_hasbrouck_lambda(state) == 0.0
    _run(state, [(i * 100, 20_000.0 + i * 0.25, i * 1.0) for i in range(4)])
    assert compute_hasbrouck_lambda(state) == 0.0


def test_positive_lambda_with_buy_pressure() -> None:
    """When mid rises with positive signed volume, λ should be positive."""
    state = HasbroucksLambdaState()
    mid = 20_000.0
    steps: list[tuple[int, float, float]] = []
    for i in range(30):
        mid += 0.25
        steps.append((i * 100, mid, 1.0))
    _run(state, steps)
    lam = compute_hasbrouck_lambda(state)
    assert lam > 0.0


def test_negative_lambda_with_sell_pressure() -> None:
    state = HasbroucksLambdaState()
    mid = 20_000.0
    steps: list[tuple[int, float, float]] = []
    for i in range(30):
        mid -= 0.25
        steps.append((i * 100, mid, 1.0))  # positive flow but prices fall
    _run(state, steps)
    lam = compute_hasbrouck_lambda(state)
    assert lam < 0.0


def test_snapshot_returns_named_feature() -> None:
    state = HasbroucksLambdaState()
    _run(state, [(i * 100, 20_000.0 + i * 0.1, i * 1.0) for i in range(20)])
    snap = snapshot_hasbrouck_features(state)
    for name in HASBROUCK_FEATURE_NAMES:
        assert name in snap


def test_rolling_window_drops_old_samples() -> None:
    """Samples older than 5 min should be trimmed."""
    state = HasbroucksLambdaState()
    _run(state, [(i * 100, 20_000.0 + i * 0.1, 1.0) for i in range(20)])
    # Advance clock by 10 min — prior samples are now expired
    _run(state, [(i * 100 + 600_000, 21_000.0 + i * 0.1, 1.0) for i in range(20)])
    ts_oldest = state.samples[0][0]
    assert ts_oldest >= 600_000 - 5 * 60 * 1000


def test_returns_float() -> None:
    state = HasbroucksLambdaState()
    _run(state, [(i * 100, 20_000.0 + i * 0.25, i * 1.0) for i in range(30)])
    val = compute_hasbrouck_lambda(state)
    assert isinstance(val, float)
