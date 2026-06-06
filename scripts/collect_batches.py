#!/usr/bin/env python3
"""Collect (download) Databento batch jobs already on the server.

Use after a partial / interrupted fetch_data.py run to pick up jobs that
were submitted but never downloaded.

Usage:
    python collect_batches.py                       # list all jobs, download 'done' ones
    python collect_batches.py --list-only           # list only, no download
    python collect_batches.py --state queued        # show queued jobs (no download)
    python collect_batches.py --since 2026-04-30    # filter by ts_received
    python collect_batches.py --data-dir /Volumes/DRIVE/p6-data
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import fetch_data
from fetch_data import (
    SYMBOL_MAP,
    _download_and_rename,
    _load_api_key,
    _target_path,
)

REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}
ACTIVE_STATES = ("received", "queued", "processing")


def _parse_iso(ts: str) -> datetime:
    ts = ts.rstrip("Z").split(".")[0]
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


def _normalize_symbols(raw) -> list[str]:
    """Databento may return job 'symbols' as a list or a comma-separated string."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return [str(s) for s in raw]


def _local_symbol(db_symbol: str) -> str | None:
    """Map 'NQ.c.0' → 'NQ'. Returns None if not in our SYMBOL_MAP."""
    return REVERSE_SYMBOL_MAP.get(db_symbol)


def main() -> None:
    p = argparse.ArgumentParser(description="Collect Databento batch jobs already submitted.")
    p.add_argument("--state", default="all",
                   choices=["received", "queued", "processing", "done",
                            "expired", "failed", "active", "all"],
                   help="Filter jobs by state. 'active' = received|queued|processing. (default: all)")
    p.add_argument("--since", default=None,
                   help="Only jobs with ts_received >= this date (YYYY-MM-DD UTC)")
    p.add_argument("--list-only", action="store_true",
                   help="Print matching jobs but do not download")
    p.add_argument("--data-dir", default=None,
                   help="Output directory (default: same as fetch_data.py)")
    args = p.parse_args()

    if args.data_dir:
        fetch_data.DATA_DIR = os.path.abspath(os.path.expanduser(args.data_dir))
    os.makedirs(fetch_data.DATA_DIR, exist_ok=True)

    import databento as db
    client = db.Historical(key=_load_api_key())

    jobs = client.batch.list_jobs()
    if args.state == "active":
        jobs = [j for j in jobs if j.get("state") in ACTIVE_STATES]
    elif args.state != "all":
        jobs = [j for j in jobs if j.get("state") == args.state]
    if args.since:
        cutoff = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        jobs = [j for j in jobs if _parse_iso(j.get("ts_received", "1970-01-01")) >= cutoff]

    if not jobs:
        print(f"No jobs found (state={args.state}, since={args.since}).")
        return

    print(f"Found {len(jobs)} job(s):")
    for j in jobs:
        syms = _normalize_symbols(j.get("symbols"))
        sym = syms[0] if syms else "?"
        print(f"  {j['id']:36s}  {j.get('state'):11s}  {sym:10s}  "
              f"{j.get('start')} -> {j.get('end')}")

    if args.list_only:
        return

    done = [j for j in jobs if j.get("state") == "done"]
    if not done:
        active = [j for j in jobs if j.get("state") in ACTIVE_STATES]
        if active:
            print(f"\n{len(active)} job(s) still {'/'.join(ACTIVE_STATES)} — not yet downloadable.")
        else:
            print("\nNo jobs in 'done' state to download.")
        return

    print(f"\nDownloading {len(done)} done job(s) to {fetch_data.DATA_DIR}...")
    for j in done:
        syms = _normalize_symbols(j.get("symbols"))
        if not syms:
            print(f"  [{j['id']}] no symbols field, skipping")
            continue
        db_sym = syms[0]
        local_sym = _local_symbol(db_sym)
        if not local_sym:
            print(f"  [{j['id']}] symbol '{db_sym}' not in SYMBOL_MAP, skipping")
            continue
        try:
            start = _parse_iso(j["start"])
            end = _parse_iso(j["end"])
        except (KeyError, ValueError) as exc:
            print(f"  [{local_sym}] could not parse start/end ({exc}), skipping")
            continue

        out_path = _target_path(local_sym, start, end)
        if os.path.exists(out_path):
            print(f"  [{local_sym}] already on disk: {out_path}")
            continue

        try:
            path = _download_and_rename(client, j["id"], local_sym, start, end)
            if path:
                size_mb = os.path.getsize(path) / 1_048_576
                print(f"  [{local_sym}] {size_mb:.1f} MB -> {path}")
            else:
                print(f"  [{local_sym}] download returned no .dbn.zst path")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{local_sym}] FAILED -- {exc}")


if __name__ == "__main__":
    main()
