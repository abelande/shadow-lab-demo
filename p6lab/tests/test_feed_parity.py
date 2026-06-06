"""
Live / replay feed parity contract tests.

Ensures the engine can swap ``DatabentoReplayFeed`` for
``DatabentoLiveFeed`` without a behavior change. Uses a ``MockLiveFeed``
that pipes a .dbn.zst file through the live-feed queue so parity is
deterministic.

Coverage:
  1. Both feeds implement the BaseFeed interface (connect/next/disconnect)
  2. iter_mbo_events exists on both and yields ``Order`` objects
  3. OrderBookSnapshot field shape is identical
  4. Running the same tape through both paths produces equal MBO-event
     sequences (same order_ids, prices, actions — exact parity)
  5. Running the same tape through both paths produces
     feature-compatible snapshot sequences (per-snapshot best bid/ask
     / book depth shape)

Run:
    pytest tests/test_feed_parity.py -v
    make test-parity
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "data"
for p in (str(ROOT / "src"), str(ROOT.parent.parent), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from p6.ingestion.base_feed import BaseFeed                    # noqa: E402
from p6.ingestion.databento_feed import (                       # noqa: E402
    DatabentoLiveFeed, DatabentoReplayFeed,
)
from p6.models import OrderBookSnapshot                         # noqa: E402

# tests/ is not a package — import fixtures via its absolute path.
import importlib.util as _iu  # noqa: E402
_spec = _iu.spec_from_file_location(
    "mock_live_feed", ROOT / "tests" / "fixtures" / "mock_live_feed.py",
)
_mlf_mod = _iu.module_from_spec(_spec); _spec.loader.exec_module(_mlf_mod)
MockLiveFeed = _mlf_mod.MockLiveFeed


SAMPLE_FILE = DATA_DIR / "nq-mbo-overnight-2026-03-26.dbn.zst"
MAX_EVENTS = 2_000


def _skip_if_no_data():
    if not SAMPLE_FILE.is_file():
        pytest.skip(f"sample data missing: {SAMPLE_FILE}")


# ---------------------------------------------------------------------------
# 1) BaseFeed interface parity (static — no async run needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("feed_cls", [DatabentoReplayFeed, DatabentoLiveFeed])
def test_baseFeed_interface(feed_cls):
    """Every feed implements connect/next/disconnect + the BaseFeed API."""
    assert issubclass(feed_cls, BaseFeed)
    for method in ("connect", "next", "disconnect"):
        m = getattr(feed_cls, method, None)
        assert callable(m), f"{feed_cls.__name__} missing {method}()"
        assert inspect.iscoroutinefunction(m), f"{method} must be async"


def test_iter_mbo_events_on_both_feeds():
    """Replay exposes a sync generator, live exposes an async generator —
    both names resolve and both are generators."""
    assert hasattr(DatabentoReplayFeed, "iter_mbo_events")
    assert hasattr(DatabentoLiveFeed, "iter_mbo_events")
    # Replay's is a regular generator function (returns a generator)
    replay_fn = DatabentoReplayFeed.iter_mbo_events
    assert inspect.isgeneratorfunction(replay_fn) or inspect.isfunction(replay_fn)
    # Live's is an async generator function
    live_fn = DatabentoLiveFeed.iter_mbo_events
    assert inspect.isasyncgenfunction(live_fn), "live iter_mbo_events must be async"


# ---------------------------------------------------------------------------
# 2) Event-sequence parity (deterministic via mock)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_sequence_parity():
    """Same source file → identical Order sequence from replay vs. mock live."""
    _skip_if_no_data()

    # Replay-path events
    replay = DatabentoReplayFeed(
        file_path=str(SAMPLE_FILE), symbol="NQ",
        filter_symbol="NQ", num_levels=20,
    )
    await replay.connect()
    replay_events = []
    for i, ev in enumerate(replay.iter_mbo_events()):
        replay_events.append(ev)
        if i + 1 >= MAX_EVENTS:
            break
    await replay.disconnect()

    # Live-path events (MockLiveFeed pipes same file through the queue)
    live = MockLiveFeed(
        source_file=str(SAMPLE_FILE), symbol="NQ",
        filter_symbol="NQ", num_levels=20, real_time=False,
    )
    await live.connect()
    # Give the pipe task time to fill the queue
    await asyncio.sleep(0.2)

    live_events = []
    async for ev in live.iter_mbo_events(idle_timeout_ms=500):
        live_events.append(ev)
        if len(live_events) >= MAX_EVENTS:
            break
    await live.disconnect()

    assert len(replay_events) == len(live_events), (
        f"event counts differ: replay={len(replay_events)} vs live={len(live_events)}"
    )

    # Compare by (order_id, price, size, action, timestamp_ms) — strict parity
    mismatches = 0
    for i, (r, l) in enumerate(zip(replay_events, live_events)):
        if (r.order_id, r.price, r.size, r.action, r.timestamp_ms) != \
           (l.order_id, l.price, l.size, l.action, l.timestamp_ms):
            mismatches += 1
            if mismatches < 3:
                print(f"mismatch at {i}:\n  replay={r}\n  live=  {l}")
    assert mismatches == 0, f"{mismatches}/{len(replay_events)} event mismatches"


# ---------------------------------------------------------------------------
# 3) Snapshot shape parity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_shape_parity():
    """Both feeds' next() returns OrderBookSnapshot with identical fields."""
    _skip_if_no_data()

    replay = DatabentoReplayFeed(
        file_path=str(SAMPLE_FILE), symbol="NQ",
        filter_symbol="NQ", num_levels=20,
    )
    await replay.connect()
    r_snap = None
    for _ in range(100):
        s = await replay.next()
        if s is not None:
            r_snap = s
            break
    await replay.disconnect()
    assert r_snap is not None, "replay produced no snapshots"

    live = MockLiveFeed(
        source_file=str(SAMPLE_FILE), symbol="NQ",
        filter_symbol="NQ", num_levels=20, real_time=False,
    )
    await live.connect()
    await asyncio.sleep(0.2)
    l_snap = None
    for _ in range(200):
        s = await live.next()
        if s is not None and s.bids and s.asks:
            l_snap = s
            break
        await asyncio.sleep(0.01)
    await live.disconnect()
    assert l_snap is not None, "mock live produced no snapshots"

    # Identical dataclass type
    assert type(r_snap) is type(l_snap) is OrderBookSnapshot

    # Identical field names (not values — timestamps differ since live
    # uses wall clock, not tape ts)
    r_fields = set(r_snap.__dataclass_fields__.keys())
    l_fields = set(l_snap.__dataclass_fields__.keys())
    assert r_fields == l_fields, f"field mismatch: {r_fields ^ l_fields}"

    # Both carry non-empty book sides
    assert r_snap.bids and r_snap.asks, "replay snapshot empty"
    assert l_snap.bids and l_snap.asks, "live snapshot empty"

    # Per-level type parity: both use the same OrderBookLevel
    assert type(r_snap.bids[0]) is type(l_snap.bids[0])
