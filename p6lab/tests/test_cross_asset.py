"""Tests for p6lab.features.cross_asset (Wave 7 Phase 7B)."""
from __future__ import annotations

import numpy as np
import pytest

from p6lab.features.cross_asset import (
    CROSS_ASSET_FEATURE_NAMES,
    CrossAssetState,
    snapshot_cross_asset_features,
    update_cross_asset,
)


def _comoving_prices(n: int, *, seed: int = 0) -> list[tuple[float, float, float]]:
    """3-symbol price stream where NQ + ES are tightly co-moving and YM
    is independent. Returns one (nq, es, ym) tuple per tick."""
    rng = np.random.default_rng(seed)
    nq = 20_000.0 + np.cumsum(rng.normal(0, 1.0, n))
    es = 5_000.0 + 0.25 * (nq - 20_000.0) + rng.normal(0, 0.05, n)
    ym = 38_000.0 + np.cumsum(rng.normal(0, 0.5, n))
    return list(zip(nq.tolist(), es.tolist(), ym.tolist()))


def test_empty_state_returns_zeros() -> None:
    state = CrossAssetState()
    snap = snapshot_cross_asset_features(state, "NQ")
    for k in CROSS_ASSET_FEATURE_NAMES:
        assert snap[k] == 0.0


def test_symbol_registry_grows() -> None:
    state = CrossAssetState()
    update_cross_asset(state, ts_ms=1_000, symbol_to_mid={"NQ": 20_000.0})
    update_cross_asset(state, ts_ms=2_000, symbol_to_mid={"ES": 5_000.0})
    assert set(state.adjacency.symbols) == {"NQ", "ES"}


def test_comoving_symbols_get_nonzero_adjacency() -> None:
    state = CrossAssetState()
    prices = _comoving_prices(150, seed=7)
    for i, (nq, es, ym) in enumerate(prices):
        update_cross_asset(
            state, ts_ms=i * 100,
            symbol_to_mid={"NQ": nq, "ES": es, "YM": ym},
        )
    # After 150 ticks the adjacency matrix should reflect NQ/ES co-movement
    matrix = state.adjacency.matrix_
    assert matrix.shape == (3, 3)
    i_nq = state.adjacency.symbols.index("NQ")
    i_es = state.adjacency.symbols.index("ES")
    i_ym = state.adjacency.symbols.index("YM")
    assert abs(matrix[i_nq, i_es]) > abs(matrix[i_nq, i_ym])


def test_snapshot_returns_all_named_features() -> None:
    state = CrossAssetState()
    prices = _comoving_prices(150, seed=5)
    for i, (nq, es, ym) in enumerate(prices):
        update_cross_asset(
            state, ts_ms=i * 100,
            symbol_to_mid={"NQ": nq, "ES": es, "YM": ym},
        )
    for sym in ("NQ", "ES", "YM"):
        snap = snapshot_cross_asset_features(state, sym)
        for k in CROSS_ASSET_FEATURE_NAMES:
            assert k in snap
            assert np.isfinite(snap[k])


def test_unknown_symbol_returns_zeros() -> None:
    state = CrossAssetState()
    update_cross_asset(state, ts_ms=100, symbol_to_mid={"NQ": 20_000.0, "ES": 5_000.0})
    snap = snapshot_cross_asset_features(state, "UNKNOWN")
    for v in snap.values():
        assert v == 0.0


def test_peer_correlation_for_single_symbol_is_zero() -> None:
    state = CrossAssetState()
    for i in range(20):
        update_cross_asset(state, ts_ms=i * 100, symbol_to_mid={"NQ": 20_000.0 + i})
    snap = snapshot_cross_asset_features(state, "NQ")
    # Only one symbol → no peers → peer_correlation_avg must be 0
    assert snap["peer_correlation_avg"] == 0.0
