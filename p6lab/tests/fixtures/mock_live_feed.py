"""
Mock ``DatabentoLiveFeed`` that replays a .dbn.zst file through the same
queue-based interface the real live feed exposes.

Purpose — deterministic parity testing. Given a fixed .dbn.zst input,
the real replay feed and this mocked live feed should produce identical
``OrderBookSnapshot`` sequences when consumed via ``next()``, and
identical ``Order`` streams when consumed via ``iter_mbo_events()``.

Usage
-----

    from tests.fixtures.mock_live_feed import MockLiveFeed
    feed = MockLiveFeed(source_file="data/nq-mbo-sample.dbn.zst",
                        symbol="NQ", filter_symbol="NQ", num_levels=20)
    await feed.connect()
    snap = await feed.next()
    ...
    await feed.disconnect()

The mock derives from the real ``DatabentoLiveFeed`` so callers treat it
as a genuine live feed — the only difference is the event source.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add p6 package to sys.path (same pattern as notebooks/_common.py).
_PROJECTS = Path(__file__).resolve().parents[4]   # .../projects/
if str(_PROJECTS) not in sys.path:
    sys.path.insert(0, str(_PROJECTS))

from p6.ingestion.databento_feed import DatabentoLiveFeed, DatabentoReplayFeed


class MockLiveFeed(DatabentoLiveFeed):
    """A ``DatabentoLiveFeed`` whose ingest loop reads from a file, not a WebSocket.

    Overrides ``connect()`` to skip the real WebSocket subscription and
    instead spawns a background task that pipes events from a
    ``DatabentoReplayFeed`` into ``self._event_queue``. Everything
    downstream (``next()``, ``iter_mbo_events()``, ``disconnect()``) uses
    the real parent-class code paths unchanged.
    """

    def __init__(
        self,
        source_file: str | Path,
        *,
        symbol: str = "NQ",
        filter_symbol: Optional[str] = None,
        snapshot_interval_ms: int = 100,
        num_levels: int = 20,
        real_time: bool = False,
    ) -> None:
        # Skip the real env-var check — mock doesn't call Databento's API.
        super().__init__(
            symbol=symbol,
            snapshot_interval_ms=snapshot_interval_ms,
            num_levels=num_levels,
            api_key="mock",
        )
        self.source_file = str(source_file)
        self.filter_symbol = filter_symbol or symbol
        self.real_time = real_time   # True → pace events by actual ts gaps

    async def connect(self) -> None:
        """Open the replay file + spawn the piping task."""
        # Build the replay feed that backs us
        self._source = DatabentoReplayFeed(
            file_path=self.source_file,
            symbol=self.symbol,
            filter_symbol=self.filter_symbol,
            num_levels=self.num_levels,
        )
        await self._source.connect()
        self._connected = True
        self._ingest_task = asyncio.create_task(self._pipe_from_source())

    async def _pipe_from_source(self) -> None:
        """Drain the replay feed into the live feed's queue."""
        last_ts: Optional[int] = None
        try:
            for order in self._source.iter_mbo_events():
                if not self._connected:
                    break
                if self.real_time and last_ts is not None:
                    delta_ms = order.timestamp_ms - last_ts
                    if 0 < delta_ms < 5_000:   # avoid long waits on gaps
                        await asyncio.sleep(delta_ms / 1_000.0)
                last_ts = order.timestamp_ms
                await self._event_queue.put(order)
                # Yield periodically so the event loop can process consumers
                if self._event_queue.qsize() % 100 == 0:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass

    async def disconnect(self) -> None:
        self._connected = False
        if getattr(self, "_source", None) is not None:
            try:
                await self._source.disconnect()
            except Exception:
                pass
            self._source = None
        # Let the parent class clean up the ingest_task
        if self._ingest_task and not self._ingest_task.done():
            self._ingest_task.cancel()
            try:
                await self._ingest_task
            except (asyncio.CancelledError, Exception):
                pass
        self._ingest_task = None
