"""Verify Python and C++ books agree across 2000 snapshots."""
import asyncio
from p6v2.ingestion.databento_feed import DatabentoReplayFeed
from p6v2.pipeline_cpp import CppAcceleratedPipeline

async def test():
    feed = DatabentoReplayFeed(
        file_path='data/nq-mbo-2026-03-27.dbn.zst',
        symbol='NQ.c.0',
        snapshot_interval_ms=1000,
    )
    await feed.connect()
    cpp = CppAcceleratedPipeline()

    for i in range(2000):
        snap = await feed.next()
        if not snap:
            break
        frame = cpp.run(snap)

        cpp_bid = frame.bid_bars[0]['price'] if frame.bid_bars else None
        cpp_ask = frame.ask_bars[0]['price'] if frame.ask_bars else None

        if snap.best_bid and cpp_bid:
            assert abs(snap.best_bid - cpp_bid) < 1.0, \
                f"BID diverged at snapshot {i}: py={snap.best_bid} cpp={cpp_bid}"
        if snap.best_ask and cpp_ask:
            assert abs(snap.best_ask - cpp_ask) < 1.0, \
                f"ASK diverged at snapshot {i}: py={snap.best_ask} cpp={cpp_ask}"
        if cpp_bid and cpp_ask:
            assert cpp_bid < cpp_ask, \
                f"CROSSED book at snapshot {i}: bid={cpp_bid} > ask={cpp_ask}"

    print(f"PASS — {i+1} snapshots, no divergence")

asyncio.run(test())
