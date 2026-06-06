"""
Burst-threshold tuning sweep for the pattern miner.

Sweeps ``BurstDetectorConfig.min_events_per_100ms`` across a set of
candidate thresholds, measuring burst count + HDBSCAN cluster count at
each value. The goal is to replace the arbitrary default (``20``) in
NB04 with a data-driven pick.

Runs at 50k snapshots (~83min of NQ overnight tape) against the file
pinned in ``notebooks/_common.py``. Writes a markdown report to
``artifacts/p6lab/mining/burst_tuning_report.md`` and prints a summary.

Run:
    cd p6lab
    python3 sanity/burst_tuning_sweep.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P6LAB_ROOT = HERE.parent
NB_DIR = P6LAB_ROOT / "notebooks"
for p in (str(P6LAB_ROOT / "src"), str(P6LAB_ROOT.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import _common  # noqa: E402
from p6lab.ingestion.event_windowing import (  # noqa: E402
    BurstDetectorConfig, WindowAnchorStrategy, WindowIterator,
)
from p6lab.patterns.miner import (  # noqa: E402
    SHAPE_VECTOR_DIM, extract_event_shape_vector, run_hdbscan,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

THRESHOLDS: tuple[int, ...] = (5, 10, 20, 50)
N_SNAPSHOTS = 50_000
OUTPUT_PATH = P6LAB_ROOT / "artifacts" / "p6lab" / "mining" / "burst_tuning_report.md"


async def _load_events() -> list:
    """Return MBO events as ``Order`` objects; windowing reads `.timestamp_ms`."""
    slice_override = {**_common.NOTEBOOK_DATA_SLICE, "max_snapshots": N_SNAPSHOTS}
    return await _common.collect_events(slice_override)


def _sweep(events: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for thresh in THRESHOLDS:
        cfg = BurstDetectorConfig(
            min_events_per_100ms=thresh,
            lookback_ms=500,
            lookahead_ms=1_500,
            min_burst_gap_ms=200,
        )
        windows = list(WindowIterator.create(
            WindowAnchorStrategy.BURST_ANCHORED,
            events=events,
            burst_config=cfg,
        ))
        n_windows = len(windows)

        if n_windows == 0:
            vectors = np.zeros((0, SHAPE_VECTOR_DIM))
            n_clusters = 0
        else:
            vectors = np.stack([extract_event_shape_vector(w.events) for w in windows])
            if n_windows < 5:
                n_clusters = 0
            else:
                labels, _ = run_hdbscan(
                    vectors,
                    min_cluster_size=max(5, n_windows // 20),
                    min_samples=3,
                )
                n_clusters = len(set(int(lab) for lab in labels if lab >= 0))

        row = {
            "threshold": thresh,
            "n_windows": n_windows,
            "n_clusters": n_clusters,
            "vectors_shape": str(vectors.shape),
        }
        log.info("thresh=%d  windows=%d  clusters=%d  vec=%s",
                 thresh, n_windows, n_clusters, vectors.shape)
        rows.append(row)
    return rows


def _write_report(rows: list[dict], n_events: int, data_file: str) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    recommend, rationale = _recommend(rows, n_events)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("# NB04 — Burst Threshold Tuning Report\n\n")
        fh.write(f"**Data file:** `{data_file}`\n")
        fh.write(f"**Snapshots:** {N_SNAPSHOTS}\n")
        fh.write(f"**MBO events:** {n_events}\n")
        fh.write(f"**Event rate:** {n_events / (N_SNAPSHOTS * 0.1):.1f} events/sec "
                 f"(≈{n_events / (N_SNAPSHOTS * 0.1) / 10:.1f} per 100ms)\n\n")
        fh.write("## Sweep results\n\n")
        fh.write("| min_events/100ms | # burst windows | windows/min | # HDBSCAN clusters |\n")
        fh.write("|---|---|---|---|\n")
        dur_min = N_SNAPSHOTS * 0.1 / 60.0
        for r in rows:
            fh.write(f"| {r['threshold']} | {r['n_windows']} | "
                     f"{r['n_windows'] / dur_min:.1f} | {r['n_clusters']} |\n")
        fh.write(f"\n## Recommendation\n\n**min_events_per_100ms = {recommend}**\n\n{rationale}\n")
    log.info("wrote %s — recommendation: min_events_per_100ms=%d", OUTPUT_PATH, recommend)


def _recommend(rows: list[dict], n_events: int) -> tuple[int, str]:
    """Pick the threshold whose window rate lands in [3, 60] bursts/minute —
    dense enough to discover patterns, sparse enough to avoid burst mania."""
    dur_min = N_SNAPSHOTS * 0.1 / 60.0
    ranked = sorted(rows, key=lambda r: r["threshold"])
    target_lo, target_hi = 3 * dur_min, 60 * dur_min
    in_band = [r for r in ranked if target_lo <= r["n_windows"] <= target_hi]
    if in_band:
        pick = in_band[len(in_band) // 2]
        reason = (f"Lands in the 3-60 bursts/min target band at "
                  f"{pick['n_windows'] / dur_min:.1f} bursts/min.")
    else:
        pick = min(ranked, key=lambda r: abs(r["n_windows"] - (target_lo + target_hi) / 2))
        reason = (f"No threshold hits the 3-60/min band exactly; closest is "
                  f"{pick['n_windows'] / dur_min:.1f}/min.")
    if all(r["n_clusters"] == 0 for r in rows):
        reason += ("\n\n**NOTE:** HDBSCAN produced 0 clusters at every threshold. "
                   "This is a separate issue — likely `min_cluster_size` is too "
                   "large relative to shape-vector diversity — and should be "
                   "investigated independently of burst tuning.")
    return int(pick["threshold"]), reason


async def main() -> None:
    events = await _load_events()
    log.info("loaded %d MBO events for sweep", len(events))
    rows = _sweep(events)
    _write_report(rows, len(events), _common.NOTEBOOK_DATA_SLICE["data_file"])


if __name__ == "__main__":
    asyncio.run(main())
