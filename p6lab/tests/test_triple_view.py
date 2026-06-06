"""Tests for p6lab.ingestion.triple_view — L3/L2/L1 triple emitter."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from p6lab.ingestion.triple_view import (
    EmitterConfig,
    L1_FEATURE_DIM,
    L2_BOOK_VECTOR_DIM,
    L2_FEATURE_DIM,
    TripleViewEmitter,
)


# Minimal synthetic snapshot types compatible with DatabentoReplayFeed output
@dataclass
class _Lvl:
    price: float
    volume: float


@dataclass
class _Order:
    order_id: str
    timestamp_ms: int
    side: str
    price: float
    size: float
    action: str


@dataclass
class _Snap:
    timestamp_ms: int
    symbol: str = "NQ"
    bids: list = field(default_factory=list)
    asks: list = field(default_factory=list)
    recent_events: list = field(default_factory=list)
    recent_trades: list = field(default_factory=list)


def _make_snap(ts: int, bid=100.0, ask=100.25, events=None) -> _Snap:
    return _Snap(
        timestamp_ms=ts,
        bids=[_Lvl(bid, 10.0), _Lvl(bid - 0.25, 20.0)],
        asks=[_Lvl(ask, 10.0), _Lvl(ask + 0.25, 20.0)],
        recent_events=events or [],
    )


class TestEmit:
    def test_emits_one_frame_per_granularity_per_window(self, tmp_path: Path):
        cfg = EmitterConfig(output_dir=tmp_path, symbol="NQ",
                            granularities=[1000])  # 1s only for simplicity
        emitter = TripleViewEmitter(cfg)
        snaps = [_make_snap(t) for t in [0, 500, 1500, 2500]]
        frames = list(emitter.emit(snaps, live_mode=True))
        # ts crosses 1000 and 2000 boundaries → 3 windows (0, 1000, 2000)
        assert len(frames) >= 2

    def test_frame_has_correct_feature_dims(self, tmp_path: Path):
        cfg = EmitterConfig(output_dir=tmp_path, symbol="NQ", granularities=[1000])
        emitter = TripleViewEmitter(cfg)
        snaps = [_make_snap(t * 100) for t in range(20)]
        frames = list(emitter.emit(snaps, live_mode=True))
        for f in frames:
            assert f.l1_features.shape == (L1_FEATURE_DIM,)
            assert f.l2_features.shape == (L2_FEATURE_DIM,)
            assert f.l2_book_vector.shape == (L2_BOOK_VECTOR_DIM,)

    def test_events_accumulate_into_window(self, tmp_path: Path):
        cfg = EmitterConfig(output_dir=tmp_path, symbol="NQ", granularities=[1000])
        emitter = TripleViewEmitter(cfg)
        snaps = [
            _make_snap(0, events=[_Order("a", 0, "bid", 100.0, 1, "add")]),
            _make_snap(500, events=[_Order("b", 500, "ask", 100.25, 1, "add")]),
            _make_snap(1500, events=[_Order("c", 1500, "bid", 99.75, 1, "cancel")]),
        ]
        frames = list(emitter.emit(snaps, live_mode=True))
        # First window (0-1000): 2 events. Second (1000-2000): 1 event.
        by_start = {f.timestamp_ms: f for f in frames}
        assert 0 in by_start
        assert len(by_start[0].l3_events) == 2


class TestParquetRoundTrip:
    def test_flush_writes_parquet(self, tmp_path: Path):
        cfg = EmitterConfig(output_dir=tmp_path, symbol="NQ", granularities=[1000])
        emitter = TripleViewEmitter(cfg)
        snaps = [_make_snap(t * 100) for t in range(30)]
        list(emitter.emit(snaps))  # live_mode=False → flush on completion
        path = tmp_path / "NQ_1s.parquet"
        assert path.exists()

    def test_load_parquet_recovers_features(self, tmp_path: Path):
        cfg = EmitterConfig(output_dir=tmp_path, symbol="NQ", granularities=[1000])
        emitter = TripleViewEmitter(cfg)
        snaps = [_make_snap(t * 100) for t in range(30)]
        list(emitter.emit(snaps))
        df = TripleViewEmitter.load_parquet(tmp_path / "NQ_1s.parquet", "1s")
        assert len(df) > 0
        assert "l1_features" in df.columns
        assert isinstance(df.iloc[0]["l1_features"], np.ndarray)
        assert df.iloc[0]["l1_features"].shape == (L1_FEATURE_DIM,)
