"""Wave 8.5-A — MultiSymbolRunner observability counter tests.

Covers the p6lab-side counters (ingest_errors, match_errors,
sync_ingest_errors). Pipeline + engine_runner counters live in the
p6-v2 tests/test_error_counters.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass
class _Lvl:
    price: float
    volume: float


@dataclass
class _Snap:
    timestamp_ms: int
    symbol: str
    bids: list = field(default_factory=list)
    asks: list = field(default_factory=list)
    recent_events: list = field(default_factory=list)
    mid_price: float = 0.0


def _snap(ts: int, symbol: str, mid: float) -> _Snap:
    return _Snap(
        timestamp_ms=ts,
        symbol=symbol,
        bids=[_Lvl(mid - 0.25, 10.0)],
        asks=[_Lvl(mid + 0.25, 10.0)],
        recent_events=[],
        mid_price=mid,
    )


def test_wave_85_a_multi_symbol_sync_ingest_counter_scalar() -> None:
    """MultiSymbolRunner.stats['sync_ingest_errors'] is a scalar that
    increments on any ingest_sync failure. Our _Snap has bids/asks but
    no complete OrderBookSnapshot wiring so FeatureAccumulator raises
    deep inside — exactly the scenario the counter exists for."""
    from p6lab.live.multi_symbol_runner import (
        MultiSymbolRunner,
        MultiSymbolRunnerConfig,
    )
    runner = MultiSymbolRunner(MultiSymbolRunnerConfig(symbols=["NQ"]))
    assert runner.stats["sync_ingest_errors"] == 0
    for i in range(3):
        runner.ingest_sync("NQ", _snap(i * 100, "NQ", 20_000.0))
    # Accumulator.ingest() raises because our _Snap isn't a real
    # OrderBookSnapshot — that's the scenario sync_ingest_errors tracks.
    assert runner.stats["sync_ingest_errors"] == 3
    # snapshots_ingested still counts (cross-asset update was accepted)
    assert runner.stats["snapshots_ingested"] == 3


def test_wave_85_a_multi_symbol_isolation_between_symbols() -> None:
    """A symbol whose ingest raises must NOT prevent another symbol
    from progressing its cross-asset state."""
    from p6lab.live.multi_symbol_runner import (
        MultiSymbolRunner,
        MultiSymbolRunnerConfig,
    )
    runner = MultiSymbolRunner(MultiSymbolRunnerConfig(symbols=["NQ", "ES"]))
    # Both symbols' _Snap objects raise deep in FeatureAccumulator, but
    # cross-asset updates (which don't depend on the accumulator) still
    # register. This is exactly what makes multi-symbol isolation work.
    runner.ingest_sync("NQ", _snap(100, "NQ", 20_000.0))
    runner.ingest_sync("ES", _snap(100, "ES", 5_000.0))
    # Both symbols registered in cross-asset despite accumulator raise
    assert "NQ" in runner.cross_asset.adjacency.symbols
    assert "ES" in runner.cross_asset.adjacency.symbols
    # Sync ingest errors track both raises
    assert runner.stats["sync_ingest_errors"] == 2


def test_wave_85_a_multi_symbol_counter_keys_present_on_init() -> None:
    """The new counter keys are initialized empty at construction."""
    from p6lab.live.multi_symbol_runner import (
        MultiSymbolRunner,
        MultiSymbolRunnerConfig,
    )
    runner = MultiSymbolRunner(MultiSymbolRunnerConfig(symbols=["NQ", "ES"]))
    assert runner.stats["ingest_errors"] == {}
    assert runner.stats["match_errors"] == {}
    assert runner.stats["sync_ingest_errors"] == 0
