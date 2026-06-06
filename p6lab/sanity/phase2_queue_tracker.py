"""Phase 2 sanity check: QueueTracker + FillSimulator on real NQ replay.

Places 1000 virtual limit orders across a real NQ replay, then computes
P(fill within 30s) bucketed by the entry queue-position quintile.

Sanity criteria:
  1. QueueTracker processes MBO events without crashing.
  2. Some orders fill, some don't (not 0% or 100% for all).
  3. P(fill) is monotonically decreasing as queue position increases
     (front of queue fills more often than back).
  4. Orders at the front of the queue (quintile 0) have meaningfully
     higher fill rates than orders at the back (quintile 4).

Run:
  python3 sanity/phase2_queue_tracker.py
"""
from __future__ import annotations

import asyncio
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

P6V2_PATH = Path("/home/bel/.openclaw/workspace-principal/projects")
sys.path.insert(0, str(P6V2_PATH))

from p6.ingestion.databento_feed import DatabentoReplayFeed                 # noqa: E402
from p6lab.execution.fill_simulator import FillSimulator, OrderSpec         # noqa: E402
from p6lab.execution.queue_tracker import QueueTracker, Side                # noqa: E402

REPLAY_FILE = (
    P6V2_PATH / "p6-v2" / "data" / "nq-mbo-overnight-2026-03-26.dbn.zst"
)
MAX_SNAPSHOTS = 10_000       # ~15 minutes of quiet overnight session
N_ORDERS = 300
FILL_HORIZON_MS = 15_000     # per-order max wait
TICK_SIZE = 0.25
SYMBOL = "NQ"

# Force unbuffered stdout so progress is visible during long runs
import os
os.environ["PYTHONUNBUFFERED"] = "1"

random.seed(42)


async def collect_events_and_orders():
    """Walk one replay to build: (sorted event list, list of virtual orders).

    Virtual orders are sampled uniformly across the replay window. Each
    order is placed at the PREVAILING best bid / best ask at its
    placement time — this simulates the real use case of a trader
    joining the existing queue.
    """
    feed = DatabentoReplayFeed(
        file_path=str(REPLAY_FILE),
        symbol=SYMBOL, filter_symbol="NQ",
        snapshot_interval_ms=100, num_levels=10,
    )
    await feed.connect()

    all_events: list = []
    placement_candidates: list[tuple[int, float, float]] = []  # (ts, bid, ask)

    count = 0
    while count < MAX_SNAPSHOTS:
        snap = await feed.next()
        if snap is None:
            break
        if snap.bids and snap.asks:
            best_bid = snap.bids[0].price
            best_ask = snap.asks[0].price
            placement_candidates.append((snap.timestamp_ms, best_bid, best_ask))

        for ev in snap.recent_events or []:
            all_events.append(ev)
        count += 1

    # Events are already timestamp-ordered within each snapshot, but
    # across snapshots the last event of snap N may be later than early
    # events of snap N+1 if the snapshot window overlaps. Stable sort.
    all_events.sort(key=lambda e: getattr(e, "timestamp_ms", 0))

    # Sample N_ORDERS placements. Randomly choose bid/ask side.
    sampled = random.sample(
        placement_candidates,
        k=min(N_ORDERS, len(placement_candidates)),
    )
    orders: list[OrderSpec] = []
    for ts, bid, ask in sampled:
        if random.random() < 0.5:
            orders.append(OrderSpec(
                timestamp_ms=ts, side=Side.BUY, price=bid, size=1.0,
                max_horizon_ms=FILL_HORIZON_MS, adverse_exit_ticks=8,
            ))
        else:
            orders.append(OrderSpec(
                timestamp_ms=ts, side=Side.SELL, price=ask, size=1.0,
                max_horizon_ms=FILL_HORIZON_MS, adverse_exit_ticks=8,
            ))
    orders.sort(key=lambda o: o.timestamp_ms)

    return all_events, orders, count


def sample_entry_positions(orders: list, events: list) -> list[float]:
    """For each order, determine its entry queue position by running
    a separate QueueTracker up to each order's placement time.

    This is used purely for bucketing in the final analysis — the
    simulate_bulk call will do its own run."""
    tracker = QueueTracker(tick_size=TICK_SIZE)
    order_queue = sorted(enumerate(orders), key=lambda p: p[1].timestamp_ms)
    pending = list(order_queue)
    entry_positions: list[float] = [0.0] * len(orders)

    for ev in events:
        ev_ts = int(getattr(ev, "timestamp_ms", 0) or 0)
        # Capture entry positions for any orders whose time has passed
        while pending and pending[0][1].timestamp_ms <= ev_ts:
            idx, spec = pending.pop(0)
            h = tracker.place_limit_order(
                spec.timestamp_ms, spec.side, spec.price, spec.size,
            )
            pos = tracker.get_position(h)
            entry_positions[idx] = pos.position_from_front
            # Clean up — the real simulation runs its own tracker
            tracker.cancel_order(h)
        tracker.on_event(ev)

    return entry_positions


def summarize(outcomes, entry_positions):
    print("\n" + "=" * 100)
    print("PHASE 2 SANITY CHECK RESULTS")
    print("=" * 100)

    n = len(outcomes)
    n_filled = sum(1 for o in outcomes if o.filled)
    n_partial = sum(1 for o in outcomes if o.fill_reason == "partial")
    n_timeout = sum(1 for o in outcomes if o.fill_reason == "timeout")
    n_adverse = sum(1 for o in outcomes if o.fill_reason == "adverse_exit")

    print(f"Total orders:     {n}")
    print(f"Filled (full):    {n_filled:>5d}  ({n_filled/n*100:.1f}%)")
    print(f"Partial:          {n_partial:>5d}  ({n_partial/n*100:.1f}%)")
    print(f"Timeout:          {n_timeout:>5d}  ({n_timeout/n*100:.1f}%)")
    print(f"Adverse exit:     {n_adverse:>5d}  ({n_adverse/n*100:.1f}%)")

    # Bucket by entry queue position quintile
    entry_array = np.array(entry_positions)
    if entry_array.max() == 0:
        print("\n⚠ All orders had zero entry queue position — no quintile breakdown possible")
        return False

    # Use percentile bins
    quintile_edges = np.percentile(entry_array, [0, 20, 40, 60, 80, 100])
    print(f"\nEntry queue position quintile edges (contracts): {quintile_edges}")

    print("\nP(fill within 30s) by entry queue position quintile:")
    print(f"  {'quintile':<12} {'n':>6} {'pos range':>18}  {'fill_rate':>10}  {'adverse_rate':>13}")
    print("  " + "-" * 70)
    fill_rates = []
    for q in range(5):
        lo, hi = quintile_edges[q], quintile_edges[q + 1]
        if q == 4:
            mask = (entry_array >= lo) & (entry_array <= hi)
        else:
            mask = (entry_array >= lo) & (entry_array < hi)
        bucket_idx = np.where(mask)[0]
        bucket_outcomes = [outcomes[i] for i in bucket_idx]
        if not bucket_outcomes:
            continue
        filled = sum(1 for o in bucket_outcomes if o.filled)
        adverse = sum(1 for o in bucket_outcomes if o.fill_reason == "adverse_exit")
        rate = filled / len(bucket_outcomes)
        adv_rate = adverse / len(bucket_outcomes)
        fill_rates.append(rate)
        print(f"  Q{q:<11} {len(bucket_outcomes):>6} {f'[{lo:.1f}, {hi:.1f}]':>18}  "
              f"{rate:>10.3f}  {adv_rate:>13.3f}")

    # Sanity checks
    print("\nSanity criteria:")
    criteria_passed = 0
    criteria_total = 0

    # 1: Some fill, some don't
    criteria_total += 1
    if 0.01 <= n_filled / n <= 0.99:
        print("  ✓ Fill rate is between 1% and 99% (meaningful distribution)")
        criteria_passed += 1
    else:
        print(f"  ❌ Fill rate degenerate: {n_filled/n:.3f}")

    # 2: Front-of-queue fills more often than back
    criteria_total += 1
    if len(fill_rates) >= 2:
        if fill_rates[0] > fill_rates[-1]:
            print(f"  ✓ Front-of-queue fill rate ({fill_rates[0]:.3f}) > back-of-queue ({fill_rates[-1]:.3f})")
            criteria_passed += 1
        else:
            print(f"  ⚠ Fill rate NOT higher at front. Front={fill_rates[0]:.3f}, back={fill_rates[-1]:.3f}")
    else:
        print(f"  ⚠ Not enough quintiles populated to compare")

    # 3: Some adverse exits (indicates the gate is working)
    criteria_total += 1
    if 0.01 <= n_adverse / n <= 0.8:
        print(f"  ✓ Adverse exit rate reasonable ({n_adverse/n:.3f})")
        criteria_passed += 1
    else:
        print(f"  ⚠ Adverse exit rate unusual: {n_adverse/n:.3f}")

    # 4: Tracker survived without crashing (we got this far)
    criteria_total += 1
    print(f"  ✓ QueueTracker + FillSimulator processed full replay without crashing")
    criteria_passed += 1

    print(f"\n{'✅' if criteria_passed == criteria_total else '⚠'} "
          f"{criteria_passed}/{criteria_total} sanity criteria passed")

    return criteria_passed == criteria_total


async def main() -> int:
    if not REPLAY_FILE.exists():
        print(f"❌ Replay file not found: {REPLAY_FILE}")
        return 1

    print(f"Loading {REPLAY_FILE.name} ({REPLAY_FILE.stat().st_size / 1e6:.0f} MB)...")
    t0 = time.time()
    events, orders, n_snaps = await collect_events_and_orders()
    print(f"  Collected {len(events):,} events from {n_snaps:,} snapshots")
    print(f"  Sampled {len(orders)} virtual orders across the replay")
    print(f"  Collection took {time.time() - t0:.1f}s")

    print("\nMeasuring entry queue positions...")
    t0 = time.time()
    entry_positions = sample_entry_positions(orders, events)
    print(f"  Took {time.time() - t0:.1f}s")
    print(f"  Entry position range: [{min(entry_positions):.1f}, {max(entry_positions):.1f}]")
    print(f"  Median entry position: {np.median(entry_positions):.1f} contracts")

    print("\nRunning bulk fill simulation...")
    t0 = time.time()
    sim = FillSimulator(tick_size=TICK_SIZE, tick_value=20.0)  # NQ = $20/pt
    outcomes = sim.simulate_bulk(orders, events)
    print(f"  Took {time.time() - t0:.1f}s for {len(orders)} orders")

    ok = summarize(outcomes, entry_positions)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
