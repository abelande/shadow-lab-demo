"""Tests for p6lab.live.multi_symbol_runner (Wave 7 Phase 7A)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from p6lab.features.cross_asset import snapshot_cross_asset_features
from p6lab.live.multi_symbol_runner import (
    MultiSymbolRunner,
    MultiSymbolRunnerConfig,
)


@dataclass
class _Lvl:
    price: float
    volume: float


@dataclass
class _Snap:
    timestamp_ms: int
    symbol: str
    bids: list
    asks: list
    recent_events: list
    mid_price: float


def _snap(ts: int, symbol: str, mid: float) -> _Snap:
    return _Snap(
        timestamp_ms=ts,
        symbol=symbol,
        bids=[_Lvl(mid - 0.25, 10.0)],
        asks=[_Lvl(mid + 0.25, 10.0)],
        recent_events=[],
        mid_price=mid,
    )


def test_empty_symbols_raises() -> None:
    with pytest.raises(ValueError):
        MultiSymbolRunnerConfig(symbols=[])
        MultiSymbolRunner(MultiSymbolRunnerConfig(symbols=[]))


def test_ingest_sync_registers_symbol() -> None:
    runner = MultiSymbolRunner(MultiSymbolRunnerConfig(symbols=["NQ", "ES", "YM"]))
    runner.ingest_sync("NQ", _snap(1_000, "NQ", 20_000.0))
    runner.ingest_sync("ES", _snap(1_000, "ES", 5_000.0))
    assert "NQ" in runner.cross_asset.adjacency.symbols
    assert "ES" in runner.cross_asset.adjacency.symbols


def test_cross_asset_features_appear_after_many_ticks() -> None:
    runner = MultiSymbolRunner(MultiSymbolRunnerConfig(symbols=["NQ", "ES", "YM"]))
    import numpy as np
    rng = np.random.default_rng(0)
    nq = 20_000.0
    es = 5_000.0
    ym = 38_000.0
    for i in range(150):
        dn = rng.normal(0, 1.0)
        nq += dn
        es += 0.25 * dn + rng.normal(0, 0.05)
        ym += rng.normal(0, 0.5)
        runner.ingest_sync("NQ", _snap(i * 100, "NQ", nq))
        runner.ingest_sync("ES", _snap(i * 100, "ES", es))
        runner.ingest_sync("YM", _snap(i * 100, "YM", ym))

    nq_feats = runner.snapshot_features("NQ")
    assert "peer_correlation_avg" in nq_feats
    assert nq_feats["peer_correlation_avg"] > 0.0


def test_stats_increment() -> None:
    runner = MultiSymbolRunner(MultiSymbolRunnerConfig(symbols=["NQ"]))
    for i in range(10):
        runner.ingest_sync("NQ", _snap(i * 100, "NQ", 20_000.0 + i * 0.25))
    assert runner.stats["snapshots_ingested"] == 10
    assert runner.stats["cross_asset_updates"] == 10


def test_unknown_symbol_snapshot_returns_empty() -> None:
    runner = MultiSymbolRunner(MultiSymbolRunnerConfig(symbols=["NQ"]))
    assert runner.snapshot_features("UNKNOWN") == {}
