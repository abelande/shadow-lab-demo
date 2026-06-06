"""
p6lab.live.multi_symbol_runner — Wave 7 Phase 7A

Drives N parallel ``FeatureAccumulator`` instances + a shared
``CrossAssetState`` so the engine can see both per-symbol features
AND cross-asset adjacency / network momentum at match time.

Design rules
------------
- **Single event loop.** One asyncio loop pulls ``(symbol, snapshot)``
  tuples off a single queue. Each snapshot is routed to its per-symbol
  accumulator and the cross-asset aggregator.
- **Shared broker.** One ``MatchBroker`` is the fan-out for every
  instrument's matches, so renderers (outcome tracker, audit log, …)
  get a single unified stream tagged with ``instrument``.
- **Fail-soft.** A failing feed, accumulator, or engine for one symbol
  never stops the others.

Exported:
    MultiSymbolRunnerConfig
    MultiSymbolRunner
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from p6lab.correlation.match_broker import MatchBroker
from p6lab.features.cross_asset import (
    CrossAssetState,
    snapshot_cross_asset_features,
    update_cross_asset,
)
from p6lab.live.feature_accumulator import FeatureAccumulator
from p6lab.patterns.template_matcher import MatchContext

logger = logging.getLogger(__name__)


@dataclass
class MultiSymbolRunnerConfig:
    symbols: list[str]
    tick_size: float = 0.25
    window_seconds: float = 300.0
    num_levels: int = 20
    match_interval_ms: int = 1_000
    default_vix_level: float = 18.0
    default_regime: str = "normal"
    default_relative_volume: float = 1.0


class MultiSymbolRunner:
    """Orchestrate per-symbol accumulators + a shared cross-asset state."""

    def __init__(
        self,
        config: MultiSymbolRunnerConfig,
        *,
        engine_factory: Callable[[MatchBroker], Any] | None = None,
        feed_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not config.symbols:
            raise ValueError("MultiSymbolRunnerConfig.symbols must be non-empty")
        self.config = config
        self._engine_factory = engine_factory
        self._feed_factory = feed_factory
        self._broker: MatchBroker | None = None
        self._accumulators: dict[str, FeatureAccumulator] = {}
        self._engine: Any = None
        self._cross_asset = CrossAssetState()
        self._last_match_ts: dict[str, float] = {s: 0.0 for s in config.symbols}
        self._shutdown = asyncio.Event()
        self.stats: dict[str, Any] = {
            "snapshots_ingested": 0,
            "match_calls": 0,
            "matches_emitted": 0,
            "cross_asset_updates": 0,
            # Wave 8.5-A: per-symbol error counters. Plain dict (not
            # defaultdict) on read so JSON serialization is transparent;
            # increments use .setdefault(sym, 0). sync_ingest_errors is
            # a single scalar (not per-symbol) since ingest_sync is
            # called from tests / notebook wiring, not live feeds.
            "ingest_errors": {},        # dict[symbol, int]
            "match_errors": {},         # dict[symbol, int]
            "sync_ingest_errors": 0,
        }

    @property
    def broker(self) -> MatchBroker | None:
        return self._broker

    @property
    def cross_asset(self) -> CrossAssetState:
        return self._cross_asset

    def accumulator(self, symbol: str) -> FeatureAccumulator | None:
        return self._accumulators.get(symbol)

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def run(self, *, duration_seconds: float | None = None) -> dict:
        """Poll each symbol's feed, drive accumulators + cross-asset,
        invoke engine.match() at cadence. Returns stats at shutdown."""
        self._broker = MatchBroker()
        self._accumulators = {
            sym: FeatureAccumulator(
                tick_size=self.config.tick_size,
                window_seconds=self.config.window_seconds,
                num_levels=self.config.num_levels,
            )
            for sym in self.config.symbols
        }
        self._engine = self._engine_factory(self._broker) if self._engine_factory else None
        feeds = [
            (sym, self._feed_factory(sym) if self._feed_factory else None)
            for sym in self.config.symbols
        ]
        for sym, feed in feeds:
            if feed is not None and hasattr(feed, "connect"):
                await feed.connect()

        started_at = time.monotonic()
        try:
            while not self._shutdown.is_set():
                if duration_seconds is not None and \
                   time.monotonic() - started_at >= duration_seconds:
                    break

                any_progress = False
                for sym, feed in feeds:
                    if feed is None:
                        continue
                    snap = await feed.next()
                    if snap is None:
                        continue
                    any_progress = True
                    await self._handle_snapshot(sym, snap)

                if not any_progress:
                    await asyncio.sleep(0.01)
        finally:
            for sym, feed in feeds:
                if feed is not None and hasattr(feed, "disconnect"):
                    try:
                        await feed.disconnect()
                    except Exception:
                        logger.warning("disconnect for %s raised; ignoring", sym)
        return dict(self.stats)

    def request_shutdown(self) -> None:
        self._shutdown.set()

    # ------------------------------------------------------------------
    # Per-snapshot wiring
    # ------------------------------------------------------------------

    async def _handle_snapshot(self, symbol: str, snapshot: Any) -> None:
        accum = self._accumulators.get(symbol)
        if accum is None:
            return
        try:
            accum.ingest(snapshot)
        except Exception:
            # Wave 8.5-A: per-symbol ingest error counter. Other symbols'
            # streams continue unaffected — multi-symbol isolation preserved.
            self.stats["ingest_errors"][symbol] = self.stats["ingest_errors"].get(symbol, 0) + 1
            logger.exception("accumulator.ingest for %s raised", symbol)
            return
        self.stats["snapshots_ingested"] += 1

        # Update cross-asset state
        mid = _snapshot_mid(snapshot)
        if mid is not None:
            update_cross_asset(
                self._cross_asset,
                ts_ms=int(getattr(snapshot, "timestamp_ms", 0) or 0),
                symbol_to_mid={symbol: mid},
            )
            self.stats["cross_asset_updates"] += 1

        # Throttle engine match()
        now = time.monotonic()
        if now - self._last_match_ts.get(symbol, 0.0) < self.config.match_interval_ms / 1000.0:
            return
        self._last_match_ts[symbol] = now

        if self._engine is None:
            return
        windows = accum.window()
        if windows is None:
            return
        l2_window, l1_window = windows
        if len(l2_window) < 5:
            return
        ctx = MatchContext(
            time_of_day_minutes=0,
            vix_level=self.config.default_vix_level,
            vix_regime=self.config.default_regime,
            relative_volume=self.config.default_relative_volume,
            instrument=symbol,
        )
        try:
            matches = self._engine.match(
                l2_window=l2_window, l1_window=l1_window, context=ctx,
            )
            self.stats["match_calls"] += 1
            self.stats["matches_emitted"] += len(matches)
        except Exception:
            # Wave 8.5-A: per-symbol match error counter.
            self.stats["match_errors"][symbol] = self.stats["match_errors"].get(symbol, 0) + 1
            logger.exception("engine.match for %s raised", symbol)

    def ingest_sync(self, symbol: str, snapshot: Any) -> None:
        """Synchronous ingestion entry — used by tests + notebook wiring
        that don't want to spin an asyncio loop.

        Cross-asset updates are decoupled from accumulator ingestion: even
        when the accumulator can't build a full FeatureRow (e.g. shallow
        book state), the cross-asset adjacency still sees the mid and
        continues to build correlation history.
        """
        self.stats["snapshots_ingested"] += 1
        mid = _snapshot_mid(snapshot)
        if mid is not None:
            update_cross_asset(
                self._cross_asset,
                ts_ms=int(getattr(snapshot, "timestamp_ms", 0) or 0),
                symbol_to_mid={symbol: mid},
            )
            self.stats["cross_asset_updates"] += 1

        accum = self._accumulators.setdefault(
            symbol,
            FeatureAccumulator(
                tick_size=self.config.tick_size,
                window_seconds=self.config.window_seconds,
                num_levels=self.config.num_levels,
            ),
        )
        try:
            accum.ingest(snapshot)
        except Exception:
            # Wave 8.5-A: sync-path ingest error counter. Scalar (not
            # per-symbol) — this is a test / notebook entry, not live.
            self.stats["sync_ingest_errors"] += 1
            logger.debug("accumulator.ingest_sync for %s raised; continuing", symbol)

    def snapshot_features(self, symbol: str) -> dict[str, float]:
        """Per-symbol cross-asset feature dict. Empty when the symbol
        isn't registered or hasn't received data yet."""
        if symbol not in self._cross_asset.adjacency.symbols:
            return {}
        return snapshot_cross_asset_features(self._cross_asset, symbol)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot_mid(snap: Any) -> float | None:
    mid = getattr(snap, "mid_price", None)
    if mid is not None:
        return float(mid)
    bids = getattr(snap, "bids", None) or []
    asks = getattr(snap, "asks", None) or []
    if bids and asks:
        b = getattr(bids[0], "price", None)
        a = getattr(asks[0], "price", None)
        if b is not None and a is not None:
            return (float(b) + float(a)) / 2.0
    return None
