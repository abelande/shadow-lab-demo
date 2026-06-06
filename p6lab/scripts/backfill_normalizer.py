"""Backfill the InstrumentNormalizer's calibration cache.

Wave 4 Phase 1F. Scans a set of daily .dbn.zst files, computes:
  - 20-day median total depth (bid + ask at best level)
  - 20-day ATR (14-bar 1-min ATR averaged across days)

Writes to ``artifacts/p6lab/normalization/{symbol}_median_depth.parquet``
in the schema that ``InstrumentNormalizer.from_cache()`` expects.

Usage
-----
Run against committed fullday files::

    python scripts/backfill_normalizer.py --symbol NQ --tick-size 0.25 \
        --files data/nq-mbo-fullday-*.dbn.zst

With explicit output path::

    python scripts/backfill_normalizer.py --symbol NQ \
        --output artifacts/p6lab/normalization/NQ_median_depth.parquet \
        --files data/nq-mbo-fullday-*.dbn.zst
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import sys
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="backfill_normalizer")
    p.add_argument("--symbol", required=True)
    p.add_argument("--tick-size", type=float, default=0.25)
    p.add_argument("--files", nargs="+", required=True,
                   help="Glob or explicit paths to .dbn.zst files")
    p.add_argument("--output", type=Path, default=None,
                   help="Override output parquet path")
    p.add_argument("--max-snapshots-per-day", type=int, default=5000)
    return p.parse_args(argv)


async def _compute_day_stats(path: str, tick_size: float,
                              max_snapshots: int) -> tuple[float, float] | None:
    """Returns (median_total_depth, atr_14min) for one day, or None on empty."""
    from _common import collect_snapshots, NOTEBOOK_DATA_SLICE   # notebook helpers

    slice_ = {
        **NOTEBOOK_DATA_SLICE,        # default mode=replay + filter_symbol + etc.
        "data_file": path,
        "tick_size": tick_size,
        "max_snapshots": max_snapshots,
        "start_ms": None,
        "end_ms": None,
    }
    snaps = await collect_snapshots(slice_)
    if not snaps:
        return None

    depths = []
    mids = []
    for s in snaps:
        if not (s.bids and s.asks):
            continue
        total = float(s.bids[0].volume) + float(s.asks[0].volume)
        depths.append(total)
        mids.append(0.5 * (float(s.bids[0].price) + float(s.asks[0].price)))

    if not depths:
        return None

    med_depth = float(median(depths))

    # Simple 1-min ATR proxy from mids: std of 1-min rolling log-returns
    mid_arr = np.asarray(mids, dtype=float)
    if len(mid_arr) < 600:
        atr = float(np.std(np.diff(mid_arr)))
    else:
        # 1-min = 600 snaps at 100ms
        bars = mid_arr[::600]
        bars = bars[bars > 0]
        if len(bars) < 2:
            atr = 0.0
        else:
            rets = np.diff(bars)
            atr = float(np.sqrt(np.mean(rets ** 2)))

    return med_depth, max(atr, tick_size)   # atr floored to 1 tick


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Expand globs
    paths: list[str] = []
    for spec in args.files:
        hits = sorted(glob.glob(spec))
        if hits:
            paths.extend(hits)
        elif Path(spec).exists():
            paths.append(spec)
    if not paths:
        print("ERROR: no input files matched", file=sys.stderr)
        return 2

    print(f"Processing {len(paths)} file(s)...")

    # Import collect_snapshots from notebook scope
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))

    async def _run_all() -> list[tuple[float, float]]:
        out = []
        for p in paths:
            print(f"  {Path(p).name} ...", end=" ", flush=True)
            stats = await _compute_day_stats(p, args.tick_size,
                                              args.max_snapshots_per_day)
            if stats is None:
                print("EMPTY")
                continue
            print(f"depth={stats[0]:.1f}  atr={stats[1]:.3f}")
            out.append(stats)
        return out

    day_stats = asyncio.run(_run_all())
    if not day_stats:
        print("ERROR: no days produced stats", file=sys.stderr)
        return 2

    median_depth_20d = float(median([d for d, _ in day_stats]))
    atr_20d = float(np.mean([a for _, a in day_stats]))

    # Output path
    out_path = args.output or (Path("artifacts/p6lab/normalization")
                               / f"{args.symbol}_median_depth.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "symbol": args.symbol,
        "tick_size": args.tick_size,
        "median_depth_20d": median_depth_20d,
        "atr_20d": atr_20d,
        "n_days": len(day_stats),
    }])
    df.to_parquet(out_path, index=False)
    print(f"\n✓ wrote {out_path}")
    print(f"  median_depth_20d = {median_depth_20d:.1f}")
    print(f"  atr_20d          = {atr_20d:.4f}")
    print(f"  n_days           = {len(day_stats)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
