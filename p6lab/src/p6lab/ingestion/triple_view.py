"""
p6lab.ingestion.triple_view — Time-aligned L3/L2/L1 triple emitter.

Spec: p6-notebook-lab-spec.md §3.1

Responsibility:
  Given a prepared list of p6-v2 ``OrderBookSnapshot`` frames, emit
  time-aligned triples of (L3 truth, L2 shadow, L1 footprint) at three
  granularities (100ms / 1s / 5s) and write one parquet per granularity:

    {output_dir}/{symbol}_{granularity_label}.parquet

  L2 feature computation is deferred to Phase 4; until then L2 columns
  are zero-filled with the correct shape so the downstream schema is
  stable.

Contract:
  TripleViewEmitter is framework-agnostic. Callers drive ingestion
  themselves (e.g. DatabentoReplayFeed) and hand the emitter a list of
  already-collected snapshots. This keeps the module synchronous,
  testable, and trivially composable with async replays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Iterable, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from p6lab.features._l1_adapter import L1Adapter, L1AdapterConfig
from p6lab.features.l1_features import compute_l1_features

logger = logging.getLogger(__name__)

GRANULARITY_FAST_MS: int = 100
GRANULARITY_STD_MS: int = 1_000
GRANULARITY_SLOW_MS: int = 5_000

GRANULARITIES = Literal["100ms", "1s", "5s"]

# Keep in sync with p6lab.features.l1_features.L1_FEATURE_DIM; Wave 3
# Phase 5A expanded L1 from 16 → 19 (added 3 unit-vector components).
L1_FEATURE_DIM: int = 19
L2_FEATURE_DIM: int = 18  # Wave 9 A2a + cup_flip refactor (replaced trade_streak with detector-derived length/velocity/vw_strength)
L2_BOOK_VECTOR_DIM: int = 40

_LABEL_BY_MS: dict[int, str] = {
    GRANULARITY_FAST_MS: "100ms",
    GRANULARITY_STD_MS: "1s",
    GRANULARITY_SLOW_MS: "5s",
}


@runtime_checkable
class Order(Protocol):
    order_id: str
    timestamp_ms: int
    side: str
    price: float
    size: float
    action: str


@runtime_checkable
class OrderBookSnapshot(Protocol):
    timestamp_ms: int
    symbol: str
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    last_trade_price: float | None
    last_trade_size: float | None


@dataclass(frozen=True)
class TripleFrame:
    """Time-aligned L3 truth, L2 shadow, L1 footprint for one window."""
    timestamp_ms: int
    symbol: str
    l3_events: list[dict]
    l3_book_snapshot: dict
    l2_features: np.ndarray
    l2_book_vector: np.ndarray
    l1_features: np.ndarray
    granularity_ms: int
    window_seq: int

    def to_dict(self) -> dict:
        return {
            "timestamp_ms": self.timestamp_ms,
            "symbol": self.symbol,
            "l3_events": self.l3_events,
            "l3_book_snapshot": self.l3_book_snapshot,
            "l2_features": self.l2_features.tolist(),
            "l2_book_vector": self.l2_book_vector.tolist(),
            "l1_features": self.l1_features.tolist(),
            "granularity_ms": self.granularity_ms,
            "window_seq": self.window_seq,
        }


@dataclass
class EmitterConfig:
    output_dir: Path
    symbol: str
    granularities: list[int] = field(
        default_factory=lambda: [GRANULARITY_FAST_MS, GRANULARITY_STD_MS, GRANULARITY_SLOW_MS]
    )
    preserve_order_ids: bool = True
    batch_size: int = 1_000
    tick_size: float = 0.25


def _serialize_event(ev) -> dict:
    """Convert a p6-v2 Order/event into a JSON-compatible dict."""
    return {
        "order_id": str(getattr(ev, "order_id", "") or ""),
        "timestamp_ms": int(getattr(ev, "timestamp_ms", 0) or 0),
        "side": _stringify(getattr(ev, "side", "")),
        "price": float(getattr(ev, "price", 0.0) or 0.0),
        "size": float(getattr(ev, "size", getattr(ev, "volume", 0.0)) or 0.0),
        "action": _stringify(getattr(ev, "action", "")),
    }


def _serialize_book(snap) -> dict:
    bids = [(float(lvl.price), float(lvl.volume))
            for lvl in (getattr(snap, "bids", None) or [])]
    asks = [(float(lvl.price), float(lvl.volume))
            for lvl in (getattr(snap, "asks", None) or [])]
    return {
        "timestamp_ms": int(getattr(snap, "timestamp_ms", 0) or 0),
        "symbol": getattr(snap, "symbol", ""),
        "bids": bids,
        "asks": asks,
    }


def _stringify(v) -> str:
    if hasattr(v, "value"):  # Enum
        return str(v.value)
    if hasattr(v, "name"):
        return str(v.name)
    return str(v)


class TripleViewEmitter:
    """Emits TripleFrame objects aligned across L3/L2/L1 at multiple granularities."""

    def __init__(self, config: EmitterConfig) -> None:
        self.config = config
        self._buffers: dict[int, list[TripleFrame]] = {g: [] for g in config.granularities}
        self._adapters: dict[int, L1Adapter] = {
            g: L1Adapter(L1AdapterConfig(tick_size=config.tick_size))
            for g in config.granularities
        }
        logger.info("TripleViewEmitter initialized: symbol=%s granularities=%s",
                    config.symbol, config.granularities)

    def emit(
        self,
        snapshots: Iterable,
        *,
        live_mode: bool = False,
    ) -> Iterator[TripleFrame]:
        """
        Consume already-collected snapshots and yield TripleFrames per granularity.

        For each granularity g, a frame is emitted whenever a snapshot's
        timestamp crosses the next boundary (floor(ts / g) advances).
        All events within the closing window accumulate into ``l3_events``.

        In batch mode (default), frames are also buffered and flushed to
        parquet when the stream ends (and in ``batch_size`` chunks).
        """
        # per-granularity accumulators
        window_start: dict[int, int | None] = {g: None for g in self.config.granularities}
        pending_events: dict[int, list[dict]] = {g: [] for g in self.config.granularities}
        window_seq: dict[int, int] = {g: 0 for g in self.config.granularities}
        last_snap = None

        for snap in snapshots:
            ts = int(getattr(snap, "timestamp_ms", 0) or 0)
            last_snap = snap

            # Collect recent events once, reuse serialized form across granularities
            recent = list(getattr(snap, "recent_events", None) or [])
            serialized_events = [_serialize_event(e) for e in recent]

            for g in self.config.granularities:
                bucket_id = ts // g
                start = window_start[g]
                if start is None:
                    window_start[g] = bucket_id * g
                elif bucket_id * g > start:
                    # Close the prior window at (old start)
                    frame = self._build_frame(
                        snap_for_book=snap,
                        l3_events=pending_events[g],
                        granularity_ms=g,
                        window_start_ms=start,
                        window_seq=window_seq[g],
                    )
                    self._buffers[g].append(frame)
                    yield frame
                    window_seq[g] += 1
                    pending_events[g] = []
                    window_start[g] = bucket_id * g
                pending_events[g].extend(serialized_events)
                # Drive adapter once per snapshot (idempotent via seen_events set)
                self._adapters[g].ingest(snap)

        # Final flush — emit trailing partial windows
        if last_snap is not None:
            for g in self.config.granularities:
                if window_start[g] is not None and (pending_events[g] or True):
                    frame = self._build_frame(
                        snap_for_book=last_snap,
                        l3_events=pending_events[g],
                        granularity_ms=g,
                        window_start_ms=window_start[g],  # type: ignore[arg-type]
                        window_seq=window_seq[g],
                    )
                    self._buffers[g].append(frame)
                    yield frame
                    window_seq[g] += 1

        if not live_mode:
            for g in self.config.granularities:
                self._flush_to_parquet(g)

    def _build_frame(
        self,
        snap_for_book,
        l3_events: list[dict],
        granularity_ms: int,
        window_start_ms: int,
        window_seq: int,
    ) -> TripleFrame:
        adapter = self._adapters[granularity_ms]
        # Most recent L1 snapshot gives the "current" L1 feature vector
        if adapter.history.snapshots:
            l1_snap = adapter.history.snapshots[-1]
            l1_vec = compute_l1_features(l1_snap, adapter.history)
        else:
            l1_vec = np.zeros(L1_FEATURE_DIM, dtype=float)

        # L2 features / book vector: Phase 4 placeholder (zero-filled)
        l2_features = np.zeros(L2_FEATURE_DIM, dtype=float)
        l2_book_vector = np.zeros(L2_BOOK_VECTOR_DIM, dtype=float)

        return TripleFrame(
            timestamp_ms=window_start_ms,
            symbol=self.config.symbol,
            l3_events=l3_events,
            l3_book_snapshot=_serialize_book(snap_for_book),
            l2_features=l2_features,
            l2_book_vector=l2_book_vector,
            l1_features=l1_vec,
            granularity_ms=granularity_ms,
            window_seq=window_seq,
        )

    def _flush_to_parquet(self, granularity_ms: int) -> None:
        """Flush buffer for given granularity to parquet."""
        buf = self._buffers[granularity_ms]
        if not buf:
            return
        label = _LABEL_BY_MS[granularity_ms]
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / f"{self.config.symbol}_{label}.parquet"
        rows = [
            {
                "timestamp_ms": f.timestamp_ms,
                "symbol": f.symbol,
                "l3_events": f.l3_events,
                "l3_book_snapshot": f.l3_book_snapshot,
                "l1_features": f.l1_features.tolist(),
                "l2_features": f.l2_features.tolist(),
                "l2_book_vector": f.l2_book_vector.tolist(),
                "granularity_ms": f.granularity_ms,
                "window_seq": f.window_seq,
            }
            for f in buf
        ]
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        logger.info("Wrote %d frames to %s", len(rows), path)
        # Clear buffer after flush so emit() remains idempotent on repeated runs
        self._buffers[granularity_ms] = []

    @staticmethod
    def load_parquet(path: Path, granularity: GRANULARITIES = "1s") -> pd.DataFrame:
        """Load a triple_view parquet file produced by this emitter."""
        df = pd.read_parquet(path)
        # Convert list-columns back to numpy arrays for feature columns
        for col in ("l1_features", "l2_features", "l2_book_vector"):
            if col in df.columns:
                df[col] = df[col].apply(lambda v: np.asarray(v, dtype=float))
        return df
