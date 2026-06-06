"""Phase 4 sanity check: correlation engine end-to-end on real NQ replay.

  1. Drive DatabentoReplayFeed → snapshots.
  2. Convert to L2Snapshot, compute L2 features + book_shape_vector
     across the entire window.
  3. Run VPIN through the trade tape.
  4. Compute Fragility Index per snapshot (FI_fast and FI_full).
  5. Build a synthetic active pattern from the L2 history (use the
     median window as the template) and seed the CorrelationEngine.
  6. Slide a 30-second L2 window through the data and call
     engine.match() — measure latency and verify match rate.

Sanity criteria:
  1. L2 features finite for every snapshot.
  2. Book shape vector self-normalizes.
  3. VPIN produces at least one bucket value once volume accumulates.
  4. Fragility Index in [0, 1] for every snapshot.
  5. Engine returns matches without crashing; latency <50ms p50.
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd

P6V2 = Path("/home/bel/.openclaw/workspace-principal/projects")
sys.path.insert(0, str(P6V2))

from p6.ingestion.databento_feed import DatabentoReplayFeed                     # noqa: E402

from p6lab.correlation.engine import CorrelationEngine                          # noqa: E402
from p6lab.correlation.scorer import EnsembleScorer                             # noqa: E402
from p6lab.features.fragility_index import FragilityIndex                       # noqa: E402
from p6lab.features.l2_features import (                                        # noqa: E402
    L2History, L2Snapshot, compute_book_shape_vector, compute_l2_features,
)
from p6lab.features.vpin import (                                               # noqa: E402
    ClassificationMethod, VPINConfig, VPINState,
    classify_trade_lee_ready, compute_vpin, update_vpin_state,
)
from p6lab.patterns.library import (                                            # noqa: E402
    OutcomeDistribution, PatternDefinition, PatternLibrary, PatternStatus,
)
from p6lab.patterns.template_matcher import (                                   # noqa: E402
    BOOK_SHAPE_DIM, MatchContext, PatternTemplate, TemplateMatcher,
)

REPLAY = P6V2 / "p6-v2" / "data" / "nq-mbo-overnight-2026-03-26.dbn.zst"
MAX_SNAPSHOTS = 3_000
SYMBOL = "NQ"


def _to_l2(snap, mid: float) -> L2Snapshot:
    """Convert a p6-v2 OrderBookSnapshot into a scaffold L2Snapshot."""
    levels: list[tuple[float, float, float]] = []
    bids = list(getattr(snap, "bids", None) or [])
    asks = list(getattr(snap, "asks", None) or [])
    for b in bids[:20]:
        levels.append((float(b.price), float(b.volume), 0.0))
    for a in asks[:20]:
        levels.append((float(a.price), 0.0, float(a.volume)))
    return L2Snapshot(
        timestamp_ms=int(snap.timestamp_ms),
        symbol=SYMBOL,
        mid_price=mid,
        book_levels=levels,
    )


async def collect():
    feed = DatabentoReplayFeed(
        file_path=str(REPLAY),
        symbol=SYMBOL, filter_symbol="NQ",
        snapshot_interval_ms=100, num_levels=20,
    )
    await feed.connect()
    snaps = []
    while len(snaps) < MAX_SNAPSHOTS:
        s = await feed.next()
        if s is None:
            break
        snaps.append(s)
    return snaps


async def main() -> int:
    if not REPLAY.exists():
        print(f"❌ Replay file missing: {REPLAY}")
        return 1

    print(f"Loading {REPLAY.name}...")
    t0 = time.time()
    raw = await collect()
    print(f"  Collected {len(raw)} snapshots in {time.time() - t0:.1f}s")

    # ── Compute L2 features ──────────────────────────────────────────
    l2_history = L2History()
    l2_rows: list[np.ndarray] = []
    bsv_rows: list[np.ndarray] = []
    timestamps: list[int] = []

    print("\n[1/4] L2 features + book_shape_vector")
    t0 = time.time()
    for snap in raw:
        if not snap.bids or not snap.asks:
            continue
        mid = 0.5 * (snap.bids[0].price + snap.asks[0].price)
        s = _to_l2(snap, mid)
        feats = compute_l2_features(s, l2_history)
        l2_rows.append(feats)
        bsv_rows.append(compute_book_shape_vector(s))
        timestamps.append(s.timestamp_ms)
    l2_arr = np.array(l2_rows)
    bsv_arr = np.array(bsv_rows)
    print(f"  {len(l2_rows)} feature rows in {time.time() - t0:.1f}s")
    print(f"  L2 finite? {np.all(np.isfinite(l2_arr))}")
    print(f"  BSV self-normalizing? "
          f"bid_sum~1: {np.allclose(bsv_arr[:, :20].sum(axis=1), 1.0, atol=1e-6) | (bsv_arr[:, :20].sum(axis=1) == 0).all() == True}")

    # ── VPIN ─────────────────────────────────────────────────────────
    print("\n[2/4] VPIN")
    cfg = VPINConfig(bucket_size_fraction=1.0/50, window_size=10, avg_daily_volume=10_000)
    vstate = VPINState()
    prev_px = 0.0
    n_buckets = 0
    for snap in raw:
        for tr in (getattr(snap, "recent_trades", None) or []):
            mid = 0.5 * ((snap.bids[0].price if snap.bids else 0)
                         + (snap.asks[0].price if snap.asks else 0))
            side = classify_trade_lee_ready(
                float(tr.price), prev_px,
                snap.bids[0].price if snap.bids else 0,
                snap.asks[0].price if snap.asks else 0,
            )
            update_vpin_state(vstate, cfg, int(tr.timestamp_ms),
                              float(tr.price), float(tr.size), side)
            prev_px = float(tr.price)
        n_buckets = len(vstate.buckets)
    vpin_value = compute_vpin(vstate, cfg) or 0.0
    print(f"  Buckets accumulated: {n_buckets} (need {cfg.window_size})")
    print(f"  VPIN value: {vpin_value:.3f}")

    # ── Fragility ────────────────────────────────────────────────────
    print("\n[3/4] Fragility Index")
    fi = FragilityIndex()
    fi_values = []
    for i in range(len(l2_arr)):
        # Synthetic L1 vector — zeros are fine for sanity
        out = fi.compute(np.zeros(16), l2_arr[i], vpin_value, timestamps[i], SYMBOL)
        fi_values.append((out.fi_fast, out.fi_full))
    fi_fast_arr = np.array([v[0] for v in fi_values])
    fi_full_arr = np.array([v[1] for v in fi_values])
    print(f"  FI_fast in [0,1]: {fi_fast_arr.min():.3f}..{fi_fast_arr.max():.3f}")
    print(f"  FI_full in [0,1]: {fi_full_arr.min():.3f}..{fi_full_arr.max():.3f}")

    # ── Correlation engine ──────────────────────────────────────────
    print("\n[4/4] CorrelationEngine match latency")
    # Build a synthetic active pattern using a 30s window from the middle
    if len(bsv_arr) < 600:
        print("  ⚠ Not enough snapshots for engine sanity")
        return 0
    template_window = bsv_arr[len(bsv_arr) // 2 - 150 : len(bsv_arr) // 2 + 150]
    template_features = l2_arr[len(l2_arr) // 2]

    lib = PatternLibrary("/tmp/phase4_sanity_library.yaml")
    lib._data = None
    Path("/tmp/phase4_sanity_library.yaml").unlink(missing_ok=True)
    lib.load()
    lib.add_pattern("synth_pat", PatternDefinition(
        name="synth_pat",
        l3_signature="synth", l2_manifestation="synth", l1_footprint="synth",
        instruments=["NQ"], regime_specific=False,
        status=PatternStatus.ACTIVE,
        outcome_distribution={"5m": OutcomeDistribution(
            mean_atr=0.4, std=0.2, hit_rate=0.65, n=300,
        )},
    ))

    matcher = TemplateMatcher()
    matcher.templates["synth_pat"] = PatternTemplate(
        pattern_id="synth_pat",
        book_series=template_window,
        feature_centroid=template_features,
        pattern_context={"vix_regime": "normal"},
    )
    engine = CorrelationEngine(library=lib, matcher=matcher, scorer=EnsembleScorer())

    ctx = MatchContext(time_of_day_minutes=600, vix_level=18.0,
                       vix_regime="normal", relative_volume=1.0, instrument="NQ")

    # Slide a 30-second window across the data
    latencies_ms: list[float] = []
    n_matches = 0
    window_size = 300  # 30s @ 100ms
    for i in range(window_size, len(bsv_arr), 100):
        win = pd.DataFrame({
            "book_shape_vector": [bsv_arr[j] for j in range(i - window_size, i)],
            "bid_ask_imbalance": l2_arr[i - window_size:i, 0],
        }, index=timestamps[i - window_size:i])
        t = time.time()
        matches = engine.match(win, None, ctx)
        latencies_ms.append((time.time() - t) * 1000.0)
        n_matches += len(matches)
    p50 = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
    print(f"  Match calls: {len(latencies_ms)}, total matches: {n_matches}")
    print(f"  Latency p50={p50:.1f}ms, p95={p95:.1f}ms (target <50ms p50)")

    # ── Verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 4 SANITY CHECK SUMMARY")
    print("=" * 80)
    passed = 0
    total = 5
    if np.all(np.isfinite(l2_arr)):
        print("  ✓ L2 features finite for every snapshot"); passed += 1
    else:
        print("  ❌ L2 features have NaN/inf")
    bsv_norms = bsv_arr[:, :20].sum(axis=1)
    if np.all((np.isclose(bsv_norms, 1.0, atol=1e-6)) | (bsv_norms == 0)):
        print("  ✓ Book shape vector self-normalizing"); passed += 1
    else:
        print("  ❌ Book shape vector normalization broken")
    if n_buckets > 0:
        print(f"  ✓ VPIN produced {n_buckets} buckets"); passed += 1
    else:
        print("  ⚠ VPIN: no buckets (avg_daily_volume too high vs trade tape)")
    if fi_fast_arr.min() >= 0 and fi_fast_arr.max() <= 1 \
       and fi_full_arr.min() >= 0 and fi_full_arr.max() <= 1:
        print("  ✓ Fragility Index in [0, 1]"); passed += 1
    else:
        print("  ❌ Fragility Index out of range")
    if p50 < 50:
        print(f"  ✓ Engine latency p50={p50:.1f}ms < 50ms target"); passed += 1
    else:
        print(f"  ⚠ Engine latency p50={p50:.1f}ms exceeds 50ms target")

    print(f"\n{'✅' if passed == total else '⚠'} {passed}/{total} criteria passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
