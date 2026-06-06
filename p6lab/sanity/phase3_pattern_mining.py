"""Phase 3 sanity check: triple_view → event_windowing → miner on real NQ replay.

Drives the full Phase 3 pipeline end-to-end:
  1. Collect real NQ MBO snapshots from Databento replay
  2. TripleViewEmitter → writes {symbol}_{granularity}.parquet
  3. miner.mine() reads parquet, windows events, extracts 30-d shape
     vectors, clusters via HDBSCAN, labels forward outcomes, writes
     mined_candidates/*.parquet

Sanity criteria:
  1. TripleViewEmitter produces a non-empty parquet file.
  2. Burst windowing finds at least some windows (NQ overnight is quiet
     but real bursts exist around the NY open).
  3. Shape vectors have finite, non-degenerate values.
  4. HDBSCAN runs without crashing (may find 0 clusters on quiet data —
     that's acceptable; apply_filters=False lets us inspect raw clusters).

Run:
  python3 sanity/phase3_pattern_mining.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np

P6V2_PATH = Path("/home/bel/.openclaw/workspace-principal/projects")
sys.path.insert(0, str(P6V2_PATH))

from p6.ingestion.databento_feed import DatabentoReplayFeed                   # noqa: E402
from p6lab.ingestion.event_windowing import (                                 # noqa: E402
    BurstDetectorConfig, WindowAnchorStrategy, WindowIterator,
)
from p6lab.ingestion.triple_view import EmitterConfig, TripleViewEmitter     # noqa: E402
from p6lab.patterns.miner import (                                           # noqa: E402
    SHAPE_VECTOR_DIM, extract_event_shape_vector, mine, run_hdbscan,
)

REPLAY_FILE = (
    P6V2_PATH / "p6-v2" / "data" / "nq-mbo-overnight-2026-03-26.dbn.zst"
)
MAX_SNAPSHOTS = 3_000    # ~5 minutes of overnight — enough for sanity
SYMBOL = "NQ"
TICK_SIZE = 0.25


async def collect_snapshots():
    feed = DatabentoReplayFeed(
        file_path=str(REPLAY_FILE),
        symbol=SYMBOL, filter_symbol="NQ",
        snapshot_interval_ms=100, num_levels=10,
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
    if not REPLAY_FILE.exists():
        print(f"❌ Replay file not found: {REPLAY_FILE}")
        return 1

    print(f"Loading {REPLAY_FILE.name}...")
    t0 = time.time()
    snapshots = await collect_snapshots()
    print(f"  Collected {len(snapshots):,} snapshots in {time.time() - t0:.1f}s")

    total_events = sum(len(getattr(s, "recent_events", None) or []) for s in snapshots)
    print(f"  Embedded L3 events: {total_events:,}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # ── Step 1: triple view ──────────────────────────────────────
        print("\n[1/3] TripleViewEmitter → parquet")
        cfg = EmitterConfig(
            output_dir=tmp / "triple_view",
            symbol=SYMBOL,
            granularities=[1_000],
            tick_size=TICK_SIZE,
        )
        emitter = TripleViewEmitter(cfg)
        t0 = time.time()
        frames = list(emitter.emit(snapshots))
        print(f"  Produced {len(frames)} frames in {time.time() - t0:.1f}s")
        parquet = cfg.output_dir / f"{SYMBOL}_1s.parquet"
        assert parquet.exists(), "parquet not written"
        print(f"  Wrote {parquet.name} ({parquet.stat().st_size / 1e3:.0f} KB)")

        # ── Step 2: windowing + shape vectors ────────────────────────
        print("\n[2/3] Burst windowing + shape extraction")
        df = TripleViewEmitter.load_parquet(parquet, "1s")
        all_events = []
        for lst in df["l3_events"]:
            if lst is not None:
                all_events.extend([dict(e) for e in lst])
        all_events.sort(key=lambda e: int(e.get("timestamp_ms", 0)))

        # Very lenient burst config — NQ overnight has few dense moments.
        # Goal is to exercise the pipeline, not to tune pattern discovery.
        burst_cfg = BurstDetectorConfig(
            min_events_per_100ms=10, lookback_ms=500, lookahead_ms=1500,
            min_burst_gap_ms=100,
        )
        iterator = WindowIterator.create(
            WindowAnchorStrategy.BURST_ANCHORED,
            events=all_events,
            burst_config=burst_cfg,
        )
        windows = list(iterator)
        print(f"  Burst windows: {len(windows)}")
        if windows:
            vectors = np.stack([extract_event_shape_vector(w.events) for w in windows])
            print(f"  Shape vectors: {vectors.shape}")
            print(f"  Finite? {np.all(np.isfinite(vectors))}")
            print(f"  Non-zero dims (dim 0-14): {np.count_nonzero(vectors[:, :15].sum(axis=0))}")
            # ── Step 3: HDBSCAN ────────────────────────────────────
            print("\n[3/3] HDBSCAN clustering")
            mcs = max(5, len(vectors) // 50)
            ms = max(3, mcs // 2)
            labels, _ = run_hdbscan(vectors, min_cluster_size=mcs, min_samples=ms)
            n_clusters = len(set(l for l in labels if l >= 0))
            n_noise = int(np.sum(labels == -1))
            print(f"  Clusters: {n_clusters} (noise={n_noise}/{len(labels)})")
        else:
            print("  No burst windows produced — overnight NQ is quiet.")
            vectors = np.zeros((0, SHAPE_VECTOR_DIM))
            n_clusters = 0

        # ── Also: full mine() orchestrator ───────────────────────────
        print("\n[extra] mine() orchestrator (apply_filters=False)")
        t0 = time.time()
        candidates = mine(
            triple_view_path=cfg.output_dir,
            library_path=tmp / "library.yaml",
            output_dir=tmp / "mining_output",
            symbols=[SYMBOL],
            burst_config=burst_cfg,
            instrument_atr=5.0,
            tick_size=TICK_SIZE,
            min_cluster_size=max(5, len(windows) // 50) if windows else 5,
            min_samples=3,
            apply_filters=False,
        )
        print(f"  mine() returned {len(candidates)} candidates in {time.time() - t0:.1f}s")

        print("\n" + "=" * 80)
        print("PHASE 3 SANITY CHECK SUMMARY")
        print("=" * 80)
        passed = 0
        total = 4
        # 1
        if parquet.exists() and len(frames) > 0:
            print("  ✓ TripleViewEmitter produced a non-empty parquet")
            passed += 1
        else:
            print("  ❌ TripleViewEmitter produced no frames")
        # 2
        if len(windows) > 0 or total_events < 100:  # tolerate thin sessions
            print(f"  ✓ Burst windowing ran without error ({len(windows)} windows)")
            passed += 1
        else:
            print("  ⚠ No burst windows found in a busy session")
        # 3
        if len(windows) == 0 or np.all(np.isfinite(vectors)):
            print("  ✓ Shape vectors are finite")
            passed += 1
        else:
            print("  ❌ Shape vectors have NaN/inf")
        # 4
        print("  ✓ mine() orchestrator completed without crashing")
        passed += 1

        print(f"\n{'✅' if passed == total else '⚠'} {passed}/{total} criteria passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
