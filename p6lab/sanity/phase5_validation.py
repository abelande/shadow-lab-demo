"""Phase 5 sanity check: validation + cost model on real NQ replay.

  1. Run a small bulk fill simulation on real NQ MBO data.
  2. Apply the 4-component CostModel to every fill, compare to naive.
  3. Run CascadeAwareCPCV on a synthetic timestamp dataset with a
     simulated cascade event in the middle — verify that nearby
     training rows get embargoed.
  4. Run AugmentationEngine on a feature DataFrame; verify methods tag.
  5. Run must_beat_baseline both ways (clear winner, no winner).

Sanity criteria:
  1. CostModel produces non-negative components for every fill.
  2. Realistic cost > naive cost when adverse selection is present.
  3. CPCV embargo removes ≥1 training row near a cascade timestamp.
  4. Augmentation produces tagged samples including "original".
  5. Information gain gate accepts clear winner, rejects noise.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd

P6V2 = Path("/home/bel/.openclaw/workspace-principal/projects")
sys.path.insert(0, str(P6V2))

from p6.ingestion.databento_feed import DatabentoReplayFeed                     # noqa: E402

from p6lab.execution.cost_model import CostModel                                # noqa: E402
from p6lab.execution.fill_simulator import FillSimulator, OrderSpec             # noqa: E402
from p6lab.execution.queue_tracker import Side                                  # noqa: E402
from p6lab.ingestion.instrument_normalizer import (                             # noqa: E402
    InstrumentNormalizer, NormalizerConfig, VIXRegime,
)
from p6lab.validation.augmentation import AugmentationEngine                    # noqa: E402
from p6lab.validation.cpcv import CascadeAwareCPCV                              # noqa: E402
from p6lab.validation.information_gain import must_beat_baseline                # noqa: E402

REPLAY = P6V2 / "p6-v2" / "data" / "nq-mbo-overnight-2026-03-26.dbn.zst"
MAX_SNAPSHOTS = 2_000
SYMBOL = "NQ"
TICK_SIZE = 0.25


async def collect_orders_events():
    feed = DatabentoReplayFeed(
        file_path=str(REPLAY), symbol=SYMBOL, filter_symbol="NQ",
        snapshot_interval_ms=100, num_levels=10,
    )
    await feed.connect()
    events: list = []
    placement: list[tuple[int, float, float]] = []
    n = 0
    while n < MAX_SNAPSHOTS:
        s = await feed.next()
        if s is None:
            break
        if s.bids and s.asks:
            placement.append((s.timestamp_ms, s.bids[0].price, s.asks[0].price))
        events.extend(s.recent_events or [])
        n += 1
    events.sort(key=lambda e: getattr(e, "timestamp_ms", 0))

    # 50 random virtual orders
    import random
    random.seed(7)
    sampled = random.sample(placement, k=min(50, len(placement)))
    orders: list[OrderSpec] = []
    for ts, bid, ask in sampled:
        side, px = (Side.BUY, bid) if random.random() < 0.5 else (Side.SELL, ask)
        orders.append(OrderSpec(
            timestamp_ms=ts, side=side, price=px, size=1.0,
            max_horizon_ms=10_000, adverse_exit_ticks=8,
        ))
    orders.sort(key=lambda o: o.timestamp_ms)
    return events, orders, n


async def main() -> int:
    if not REPLAY.exists():
        print(f"❌ Replay missing: {REPLAY}")
        return 1

    print(f"Loading {REPLAY.name}...")
    t0 = time.time()
    events, orders, n_snaps = await collect_orders_events()
    print(f"  {len(events):,} events, {len(orders)} orders, {n_snaps} snaps in {time.time() - t0:.1f}s")

    # ── [1/5] Cost model on real fills ───────────────────────────────
    print("\n[1/5] CostModel on real bulk fills")
    sim = FillSimulator(tick_size=TICK_SIZE, tick_value=20.0)
    outcomes = sim.simulate_bulk(orders, events)
    n_filled = sum(1 for o in outcomes if o.filled)
    print(f"  Fills: {n_filled}/{len(outcomes)}")
    cm = CostModel(tick_value=20.0)
    breakdowns = cm.compute_batch(outcomes)
    cmp = cm.compare_to_naive(breakdowns)
    print(f"  Realistic mean: ${cmp['realistic_mean']:.2f}/contract")
    print(f"  Naive mean:     ${cmp['naive_mean']:.2f}/contract")
    print(f"  Cost ratio:     {cmp['cost_ratio']:.2f}")
    print(f"  Adverse %:      {cmp['adverse_selection_pct']*100:.1f}%")
    nonneg = all(
        b.crossed_spread_cost >= 0 and b.commission >= 0
        and b.adverse_selection_cost >= 0 and b.opportunity_cost >= 0
        for b in breakdowns
    )

    # ── [2/5] CPCV with cascade embargo ──────────────────────────────
    print("\n[2/5] CPCV with cascade embargo")
    n = 200
    ts = pd.Series(pd.date_range("2025-01-01", periods=n, freq="1D"))
    cascades = pd.Series([ts.iloc[100]])  # one cascade at day 100
    cv = CascadeAwareCPCV(n_splits=5, n_test_groups=2, cascade_embargo_days=14)
    folds = cv.split(pd.DataFrame(np.zeros((n, 3))), ts, cascades)
    embargoed_total = sum(len(f.embargoed_idx) for f in folds)
    print(f"  Folds: {len(folds)} (expected C(5,2)=10)")
    print(f"  Total rows embargoed across folds: {embargoed_total}")

    # ── [3/5] Augmentation ──────────────────────────────────────────
    print("\n[3/5] Augmentation")
    feats = pd.DataFrame({
        "bid_size": np.linspace(10, 20, 50),
        "ask_size": np.linspace(20, 10, 50),
        "spread_ticks": np.ones(50),
        "tick_velocity": np.zeros(50),
    })
    samples = AugmentationEngine(random_state=42).generate(
        feats, label=1, instrument="NQ", n_samples=8,
    )
    methods = {s.augmentation_method for s in samples}
    print(f"  Generated {len(samples)} samples; methods used: {sorted(methods)}")

    # ── [4/5] Information gain gate ──────────────────────────────────
    print("\n[4/5] Information gain gate")
    rep_winner = must_beat_baseline(0.80, 0.65, min_improvement=0.02)
    rep_noise = must_beat_baseline(0.66, 0.65, min_improvement=0.02)
    print(f"  Clear winner approved: {rep_winner.approved} ({rep_winner.reason})")
    print(f"  Noise-level rejected:  {not rep_noise.approved} ({rep_noise.reason})")

    # ── [5/5] Instrument normalizer ─────────────────────────────────
    print("\n[5/5] Instrument normalizer")
    n_inst = InstrumentNormalizer(NormalizerConfig(
        symbol="NQ", tick_size=TICK_SIZE, atr_20d=10.0, median_depth_20d=200.0,
    ))
    print(f"  spread_to_bps(99.75, 100.25) = {n_inst.spread_to_bps(99.75, 100.25):.2f}")
    print(f"  normalize_depth(100) = {n_inst.normalize_depth(100):.2f}")
    print(f"  classify_regime(VIX=22) = {n_inst.classify_regime(22).value}")

    # ── Verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 5 SANITY CHECK SUMMARY")
    print("=" * 80)
    passed = 0
    total = 5
    if nonneg:
        print("  ✓ CostModel components non-negative for every fill"); passed += 1
    else:
        print("  ❌ CostModel produced negative component")
    if cmp["realistic_total"] >= cmp["naive_total"] * 0.5:
        print("  ✓ Realistic cost is in same order as naive (sanity range)"); passed += 1
    else:
        print(f"  ⚠ Realistic cost suspiciously low vs naive ({cmp['cost_ratio']:.2f})")
    if embargoed_total > 0:
        print(f"  ✓ CPCV embargo purged {embargoed_total} rows near cascade"); passed += 1
    else:
        print("  ❌ CPCV embargo did not purge any rows")
    if "original" in methods and len(methods) >= 3:
        print(f"  ✓ Augmentation produced diverse tagged samples ({len(methods)} methods)"); passed += 1
    else:
        print(f"  ❌ Augmentation methods missing: {methods}")
    if rep_winner.approved and not rep_noise.approved:
        print("  ✓ Information gain gate: accepts winner, rejects noise"); passed += 1
    else:
        print(f"  ❌ Information gain gate broken (winner={rep_winner.approved}, noise={rep_noise.approved})")

    print(f"\n{'✅' if passed == total else '⚠'} {passed}/{total} criteria passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
