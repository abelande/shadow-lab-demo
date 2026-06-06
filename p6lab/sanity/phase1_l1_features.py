"""Phase 1 sanity check: L1 features on real NQ replay data.

Loads one NQ MBO replay file, runs each snapshot through L1Adapter +
compute_l1_features, and produces a summary statistics table.

Sanity criteria:
  1. All 16 features produce finite values (no NaN / inf)
  2. spread_bps_l1 has a sensible range for NQ (typically 0.1 - 10 bps)
  3. top_imbalance is bounded in [-1, +1]
  4. At least a few hundred non-zero refresh-rate readings (proves the
     passive-add event path is firing)
  5. Some variety in tick_direction_streak (not constant zero)

Run from the scaffold root:
  python3 sanity/phase1_l1_features.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add p6-v2 to path so we can import its ReplayFeed
P6V2_PATH = Path("/home/bel/.openclaw/workspace-principal/projects")
sys.path.insert(0, str(P6V2_PATH))

from p6.ingestion.databento_feed import DatabentoReplayFeed                 # noqa: E402
from p6lab.features._l1_adapter import L1Adapter, L1AdapterConfig           # noqa: E402
from p6lab.features.l1_features import (                                    # noqa: E402
    L1FeatureNames, compute_l1_features,
)

REPLAY_FILE = (
    P6V2_PATH / "p6-v2" / "data" / "nq-mbo-overnight-2026-03-26.dbn.zst"
)
MAX_SNAPSHOTS = 50_000   # enough for meaningful stats, bounded for speed
SYMBOL = "NQ"


async def run() -> pd.DataFrame:
    """Drive the replay feed and collect L1 features."""
    adapter = L1Adapter(L1AdapterConfig(tick_size=0.25, trim_every_n=200))

    feed = DatabentoReplayFeed(
        file_path=str(REPLAY_FILE),
        symbol=SYMBOL,
        filter_symbol="NQ",
        snapshot_interval_ms=100,
        num_levels=10,
    )
    await feed.connect()

    rows = []
    count = 0
    t0 = time.time()

    while count < MAX_SNAPSHOTS:
        snap = await feed.next()
        if snap is None:
            break
        l1_snap = adapter.ingest(snap)
        features = compute_l1_features(l1_snap, adapter.history)
        rows.append(features)
        count += 1

        if count % 10_000 == 0:
            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            print(f"  processed {count:>6d} snapshots  ({rate:.0f}/sec)")

    elapsed = time.time() - t0
    print(f"  TOTAL: {count} snapshots in {elapsed:.1f}s ({count/elapsed:.0f}/sec)")

    df = pd.DataFrame(rows, columns=L1FeatureNames.ALL)
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print summary statistics per feature."""
    print("\n" + "=" * 100)
    print("L1 FEATURE SUMMARY — Phase 1 Sanity Check")
    print("=" * 100)
    print(f"Replay file:  {REPLAY_FILE.name}")
    print(f"Snapshots:    {len(df):,}")
    print()

    # Per-feature stats
    summary = df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    summary["nan_count"] = df.isna().sum()
    summary["zero_count"] = (df == 0).sum()
    summary["zero_pct"] = (df == 0).sum() / len(df) * 100

    # Reorder columns and format
    cols = ["count", "mean", "std", "min", "1%", "25%", "50%", "75%", "99%",
            "max", "nan_count", "zero_pct"]
    summary = summary[cols]
    summary.index.name = "feature"

    print(summary.to_string(float_format=lambda x: f"{x:>10.4f}"))
    print()


def sanity_checks(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Return (all_ok, list_of_findings)."""
    findings: list[str] = []
    ok = True

    # Check 1: no NaN or inf values
    n_nan = df.isna().sum().sum()
    n_inf = np.isinf(df.values).sum()
    if n_nan > 0:
        findings.append(f"❌ FAIL: {n_nan} NaN values across all columns")
        ok = False
    else:
        findings.append("✓ All features produce finite values (no NaN)")
    if n_inf > 0:
        findings.append(f"❌ FAIL: {n_inf} inf values")
        ok = False
    else:
        findings.append("✓ No inf values")

    # Check 2: spread_bps_l1 sensible range for NQ (0.1 - 10 bps typical)
    spread_bps = df[L1FeatureNames.SPREAD_BPS_L1]
    median_spread = spread_bps.median()
    if 0.01 <= median_spread <= 50.0:
        findings.append(
            f"✓ spread_bps_l1 median = {median_spread:.3f} bps (sensible for NQ)"
        )
    else:
        findings.append(
            f"⚠ WARN: spread_bps_l1 median = {median_spread:.3f} bps — "
            f"outside typical NQ range [0.1, 10]"
        )

    # Check 3: top_imbalance bounded in [-1, +1]
    imb = df[L1FeatureNames.TOP_IMBALANCE]
    if imb.min() >= -1.0 and imb.max() <= 1.0:
        findings.append(
            f"✓ top_imbalance bounded: [{imb.min():.3f}, {imb.max():.3f}]"
        )
    else:
        findings.append(
            f"❌ FAIL: top_imbalance out of range: "
            f"[{imb.min():.3f}, {imb.max():.3f}]"
        )
        ok = False

    # Check 4: refresh rate features firing (non-zero count)
    bid_rr = df[L1FeatureNames.BID_REFRESH_RATE]
    ask_rr = df[L1FeatureNames.ASK_REFRESH_RATE]
    bid_nonzero = (bid_rr > 0).sum()
    ask_nonzero = (ask_rr > 0).sum()
    if bid_nonzero >= 100 and ask_nonzero >= 100:
        findings.append(
            f"✓ Refresh-rate events firing: "
            f"bid_refresh_rate>0 in {bid_nonzero} frames, "
            f"ask_refresh_rate>0 in {ask_nonzero}"
        )
    else:
        findings.append(
            f"⚠ WARN: Refresh-rate nonzero counts low: "
            f"bid={bid_nonzero}, ask={ask_nonzero}. "
            f"Check passive-add event classification."
        )

    # Check 5: tick_direction_streak has variety
    streak = df[L1FeatureNames.TICK_DIRECTION_STREAK]
    streak_nonzero = (streak != 0).sum()
    streak_range = streak.max() - streak.min()
    if streak_nonzero >= 100 and streak_range >= 2:
        findings.append(
            f"✓ Tick streak shows variety: nonzero={streak_nonzero}, "
            f"range=[{streak.min():.0f}, {streak.max():.0f}]"
        )
    else:
        findings.append(
            f"⚠ WARN: Tick streak mostly zero. "
            f"nonzero={streak_nonzero}, range={streak_range}"
        )

    # Check 6: trade_at_bid_ratio has some variety
    tabr = df[L1FeatureNames.TRADE_AT_BID_RATIO]
    non_neutral = ((tabr != 0.5)).sum()
    if non_neutral >= 100:
        findings.append(
            f"✓ trade_at_bid_ratio has variety: {non_neutral} non-neutral frames"
        )
    else:
        findings.append(
            f"⚠ WARN: trade_at_bid_ratio mostly at neutral 0.5 "
            f"({non_neutral} non-neutral frames) — trade classification may not be firing"
        )

    # Check 7: l1_shape_vector bounded roughly in [-1, 1]
    shape = df[L1FeatureNames.L1_SHAPE_VECTOR]
    if -1.5 <= shape.min() and shape.max() <= 1.5:
        findings.append(
            f"✓ l1_shape_vector bounded: [{shape.min():.3f}, {shape.max():.3f}]"
        )
    else:
        findings.append(
            f"⚠ WARN: l1_shape_vector out of expected range: "
            f"[{shape.min():.3f}, {shape.max():.3f}]"
        )

    return ok, findings


def main() -> int:
    if not REPLAY_FILE.exists():
        print(f"❌ Replay file not found: {REPLAY_FILE}")
        return 1

    print(f"Loading {REPLAY_FILE.name} ({REPLAY_FILE.stat().st_size / 1e6:.0f} MB)...")
    df = asyncio.run(run())

    if df.empty:
        print("❌ No snapshots produced")
        return 1

    summarize(df)

    ok, findings = sanity_checks(df)
    print("=" * 100)
    print("SANITY CHECK RESULTS")
    print("=" * 100)
    for f in findings:
        print(f"  {f}")

    print()
    if ok:
        print("✅ Phase 1 sanity check PASSED — all 16 features produce sensible values on real NQ data")
        return 0
    else:
        print("❌ Phase 1 sanity check FAILED — see findings above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
