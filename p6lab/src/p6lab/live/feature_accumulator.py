"""
FeatureAccumulator — rolling L1/L2/FI state for live engine consumption.

Live feeds emit one ``OrderBookSnapshot`` at a time; ``engine.match()``
needs two DataFrames covering a recent window (``l2_window`` with the
40-dim ``book_shape_vector`` column + the 12 L2 scalar features,
``l1_window`` with the 16-dim L1 feature set). This class stores the
per-snapshot feature rows in a ring buffer, and ``window()`` returns
the last ``window_seconds`` of rows as two aligned DataFrames.

Reuse of Wave 1/2 components:
  - ``L1Adapter`` / ``compute_l1_features`` (L1)
  - ``L2History`` / ``compute_l2_features`` / ``compute_book_shape_vector`` (L2)
  - ``FragilityIndex`` (FI — scalar side-product)
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from p6lab.features._l1_adapter import L1Adapter, L1AdapterConfig
from p6lab.features.l1_features import L1FeatureNames, compute_l1_features
from p6lab.features.fragility_index import FragilityIndex
from p6lab.features.l2_features import (
    L2FeatureNames, L2History, L2Snapshot,
    compute_book_shape_vector, compute_l2_features,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureRow:
    """One snapshot's worth of derived state — fed into the ring buffer."""
    timestamp_ms: int
    l1: np.ndarray          # (16,) L1 feature values
    l2_scalars: dict        # 12 L2 scalar features, keyed by L2FeatureNames
    book_shape_vector: np.ndarray   # (40,) — depth pyramid
    fi_fast: float
    fi_full: float


class FeatureAccumulator:
    """Per-snapshot feature ingestor; yields engine-ready sliding windows.

    Parameters
    ----------
    tick_size
        Instrument tick size (e.g. 0.25 for NQ). Feeds into L1Adapter.
    window_seconds
        Max age of rows kept in the ring buffer. Default 300s (5min) —
        matches the engine's spec of "last 30s–5m of L2 features".
    num_levels
        Book depth; passed to the L2 snapshot construction.
    vpin_source
        Optional callable returning current VPIN scalar. If absent,
        FI uses 0.0 as a safe default — FI_full is still meaningful.
    """

    def __init__(
        self,
        *,
        tick_size: float = 0.25,
        window_seconds: float = 300.0,
        num_levels: int = 20,
        vpin_source: Any = None,
        normalizer: Any = None,
    ) -> None:
        """
        Parameters
        ----------
        normalizer
            Optional ``InstrumentNormalizer`` instance (Wave 4 Phase 1F).
            When provided, ``ingest()`` produces an additional normalized
            view on ``FeatureRow.l2_scalars`` via
            ``normalizer.normalize_depth()`` + ``spread_to_bps()`` so
            cross-instrument features stay comparable.
        """
        self.tick_size = tick_size
        self.window_seconds = float(window_seconds)
        self.num_levels = num_levels
        self.vpin_source = vpin_source
        self.normalizer = normalizer   # Wave 4 Phase 1F

        self._l1_adapter = L1Adapter(L1AdapterConfig(tick_size=tick_size))
        self._l2_hist = L2History()
        self._fi = FragilityIndex()

        # Ring buffer of FeatureRow — sized so the window fits with margin
        max_rows = int(window_seconds * 10) + 100   # 100ms cadence assumed
        self._rows: deque[FeatureRow] = deque(maxlen=max_rows)

    def set_normalizer(self, normalizer: Any) -> None:
        """Install an InstrumentNormalizer after construction."""
        self.normalizer = normalizer

    # ------------------------------------------------------------------
    # Per-snapshot ingestion
    # ------------------------------------------------------------------

    def ingest(self, snap: Any) -> FeatureRow | None:
        """Compute features for one snapshot, append to ring buffer.

        Returns the new ``FeatureRow`` — or ``None`` when the snapshot
        has no book state (live feed warming up; ignore silently).
        """
        bids = getattr(snap, "bids", None) or []
        asks = getattr(snap, "asks", None) or []
        if not bids or not asks:
            return None

        # --- L1 ---
        l1_snap = self._l1_adapter.ingest(snap)
        l1_vec = compute_l1_features(l1_snap, self._l1_adapter.history)

        # --- L2 ---
        bid_map = {lvl.price: lvl.volume for lvl in bids[: self.num_levels]}
        ask_map = {lvl.price: lvl.volume for lvl in asks[: self.num_levels]}
        prices = sorted(set(bid_map) | set(ask_map), reverse=True)
        book_levels = [
            (p, bid_map.get(p, 0.0), ask_map.get(p, 0.0)) for p in prices
        ]
        bp, ap = bids[0].price, asks[0].price
        l2_snap = L2Snapshot(
            timestamp_ms=int(snap.timestamp_ms),
            symbol=getattr(snap, "symbol", ""),
            mid_price=(bp + ap) / 2,
            book_levels=book_levels,
            # Wave 4 Phase 1A: pass recent_events through so L2 features
            # can populate refresh_event_timestamps (fixes dead refresh_rate).
            recent_events=list(getattr(snap, "recent_events", []) or []),
        )
        # Wave 4 Phase 1F: optional InstrumentNormalizer for cross-instrument
        # scale-invariant features. When present, applied to l1_vec + l2
        # scalars post-compute so tree models stay comparable across
        # instruments. Initialized via accumulator.set_normalizer().
        self._l2_hist.append(l2_snap)
        feat_vec = compute_l2_features(l2_snap, self._l2_hist)
        # ``book_shape_vector`` sits at INDEX 10 in L2FeatureNames.ALL (not
        # at the end), so pair by explicit index — zip() position would
        # silently shift trade_flow_toxicity over the BSV-norm value. See
        # NB06 §01 note (Phase 5C bug).
        scalar_indices = [i for i, n in enumerate(L2FeatureNames.ALL)
                          if n != "book_shape_vector"]
        l2_scalars = {L2FeatureNames.ALL[i]: float(feat_vec[i]) for i in scalar_indices}
        scalar_names = [L2FeatureNames.ALL[i] for i in scalar_indices]
        bsv = compute_book_shape_vector(l2_snap)

        # --- Fragility ---
        l1_np = np.asarray(l1_vec, dtype=float)
        l2_np = np.asarray([l2_scalars.get(n, 0.0) for n in scalar_names], dtype=float)
        vpin = float(self.vpin_source()) if callable(self.vpin_source) else 0.0
        sub = self._fi.compute_sub_indices(l1_np, l2_np, vpin_value=vpin)
        fi_fast = self._fi.compute_fast(sub.RF, sub.SF, sub.FT)
        fi_full = self._fi.compute_full(sub)

        row = FeatureRow(
            timestamp_ms=int(snap.timestamp_ms),
            l1=l1_np,
            l2_scalars=l2_scalars,
            book_shape_vector=np.asarray(bsv, dtype=float),
            fi_fast=float(fi_fast),
            fi_full=float(fi_full),
        )
        self._rows.append(row)
        return row

    # ------------------------------------------------------------------
    # Engine-ready windows
    # ------------------------------------------------------------------

    def window(self) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Return the ``(l2_window, l1_window)`` pair for ``engine.match()``.

        Filters the ring buffer to the last ``window_seconds`` of rows.
        Returns ``None`` when the window is empty.
        """
        if not self._rows:
            return None

        latest_ts = self._rows[-1].timestamp_ms
        cutoff_ts = latest_ts - int(self.window_seconds * 1000)
        recent = [r for r in self._rows if r.timestamp_ms >= cutoff_ts]
        if not recent:
            return None

        # Build l2_window: 12 scalars + 40-dim book_shape_vector, indexed by ts
        l2_rows = []
        for r in recent:
            row = dict(r.l2_scalars)
            row["book_shape_vector"] = r.book_shape_vector
            l2_rows.append(row)
        l2_df = pd.DataFrame(l2_rows, index=[r.timestamp_ms for r in recent])

        # Build l1_window: 16 features, indexed by ts
        l1_df = pd.DataFrame(
            np.stack([r.l1 for r in recent]),
            columns=L1FeatureNames.ALL,
            index=[r.timestamp_ms for r in recent],
        )
        return l2_df, l1_df

    # ------------------------------------------------------------------
    # Read-only introspection (useful for telemetry)
    # ------------------------------------------------------------------

    def snapshot_count(self) -> int:
        return len(self._rows)

    def latest_fi(self) -> tuple[float, float] | None:
        if not self._rows:
            return None
        last = self._rows[-1]
        return last.fi_fast, last.fi_full

    def reset(self) -> None:
        """Drop ring buffer + per-adapter state. Call between replay + live switches."""
        self._rows.clear()
        self._l1_adapter = L1Adapter(L1AdapterConfig(tick_size=self.tick_size))
        self._l2_hist = L2History()
