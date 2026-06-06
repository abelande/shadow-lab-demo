"""Promote a mined pattern candidate into library.yaml.

Wave 4 Phase 1D — closes the NB04 → library gap.

NB04 produces ``mined_candidates.parquet`` in a versioned artifacts dir.
Each row has cluster_id, centroid, member_count, hit_rate_5m, sharpe.
This CLI reads one row, constructs a ``PatternDefinition``, and calls
``PatternLibrary.add_pattern()`` atomically (filelock + atomic rename).

Usage
-----
Promote a specific cluster from the latest mining run::

    python -m p6lab.scripts.promote_pattern \
        --library artifacts/p6lab/pattern_library/library.yaml \
        --candidate artifacts/p6lab/mining/nb04_latest/mined_candidates.parquet \
        --cluster-id 0 \
        --name bid_heavy_stacking_burst \
        --status mined_approved

Promote to ACTIVE directly (skip mined_approved stage)::

    python -m p6lab.scripts.promote_pattern ... --status active

Dry-run (print what would be added, no write)::

    python -m p6lab.scripts.promote_pattern ... --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from p6lab.patterns.library import (
    OutcomeDistribution,
    PatternDefinition,
    PatternLibrary,
    PatternStatus,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="promote_pattern",
        description=(
            "Promote a mined pattern candidate into library.yaml. "
            "Reads a row from mined_candidates.parquet and constructs a "
            "PatternDefinition atomically."
        ),
    )
    p.add_argument("--library", required=True, type=Path,
                   help="Path to library.yaml (created if missing).")
    p.add_argument("--candidate", required=True, type=Path,
                   help="Path to mined_candidates.parquet from NB04.")
    p.add_argument("--cluster-id", required=True, type=int,
                   help="Which cluster_id in the parquet to promote.")
    p.add_argument("--name", required=True,
                   help="Pattern ID for library.yaml (unique key).")
    p.add_argument("--status", default="mined_approved",
                   choices=["candidate", "mined_approved", "active", "retired"],
                   help="Target status after promotion.")
    p.add_argument("--instrument", default="NQ",
                   help="Instrument symbol (default NQ).")
    p.add_argument("--horizon", default="5m",
                   help="Outcome distribution horizon key (default 5m).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be added without writing.")
    return p.parse_args(argv)


def _build_definition(row: pd.Series, name: str, status: PatternStatus,
                      instrument: str, horizon: str) -> PatternDefinition:
    """Turn a mined_candidates.parquet row into a PatternDefinition."""
    hit_rate = float(row.get("hit_rate_5m", row.get("hit_rate_up", 0.0)))
    n = int(row.get("member_count", row.get("n", 0)))
    sharpe = float(row.get("sharpe", row.get("sharpe_proxy", 0.0)))
    mean_ticks = float(row.get("mean_move_ticks", 0.0))

    outcome = OutcomeDistribution(
        mean_atr=abs(mean_ticks) / 4.0 if mean_ticks else 0.25,
        std=max(0.1, abs(sharpe) * 0.3),
        hit_rate=max(0.0, min(1.0, hit_rate)),
        n=max(n, 1),
    )

    return PatternDefinition(
        name=name,
        l3_signature=f"mined_cluster_{int(row['cluster_id'])}",
        l2_manifestation=f"centroid_at_cluster_{int(row['cluster_id'])}",
        l1_footprint="derived_from_mining",
        outcome_distribution={horizon: outcome},
        min_sample_size=200,
        regime_specific=False,
        instruments=[instrument],
        status=status,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.candidate.exists():
        print(f"ERROR: candidate parquet missing: {args.candidate}", file=sys.stderr)
        return 2
    df = pd.read_parquet(args.candidate)
    if df.empty:
        print("ERROR: candidate parquet is empty", file=sys.stderr)
        return 2

    rows = df[df["cluster_id"] == args.cluster_id]
    if rows.empty:
        print(f"ERROR: cluster_id={args.cluster_id} not found. "
              f"Available: {sorted(df['cluster_id'].unique().tolist())}",
              file=sys.stderr)
        return 2
    row = rows.iloc[0]

    status = PatternStatus[args.status.upper()]
    definition = _build_definition(row, args.name, status,
                                    args.instrument, args.horizon)

    print(f"Built pattern definition: {definition.model_dump_json(indent=2)}")
    if args.dry_run:
        print("--dry-run — not writing")
        return 0

    library = PatternLibrary(args.library)
    library.load()
    if args.name in library.get_active_patterns():
        print(f"Pattern {args.name!r} already active; use --status to promote/retire.")
        library.promote(args.name, status)
    else:
        library.add_pattern(args.name, definition)
    library.save()
    print(f"✓ wrote {args.library} with pattern {args.name!r} status={status.value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
