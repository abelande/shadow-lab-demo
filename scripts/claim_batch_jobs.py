#!/usr/bin/env python3
"""Download already-completed Databento batch jobs into the local cache.

Scans `client.batch.list_jobs()` for state='done' MBO jobs on a single
continuous symbol, filters to those whose UTC start date falls in the
requested range, downloads them, and renames to the local filename
convention so `fetch_data.py` will skip them on a subsequent run.

Usage:
    python claim_batch_jobs.py --date 2026-03-23:2026-04-21 --data-dir /path/to/out
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "ingestion"))
from dbn_filename import make_dbn_filename, rename_batch_to_local  # noqa: E402

SYMBOL_MAP_REV = {
    "NQ.c.0": "NQ",
    "ES.c.0": "ES",
    "CL.c.0": "CL",
    "GC.c.0": "GC",
    "SI.c.0": "SI",
}


def _load_api_key() -> str:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith("DATABENTO_API_KEY="):
                return line.split("=", 1)[1].strip()
    key = os.environ.get("DATABENTO_API_KEY", "")
    if not key:
        sys.exit("ERROR: DATABENTO_API_KEY not found in .env or environment.")
    return key


def _parse_iso(ts: str) -> datetime:
    ts = ts.rstrip("Z").split(".")[0]
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD:YYYY-MM-DD range")
    p.add_argument("--data-dir", required=True, help="Output directory")
    p.add_argument("--symbol", nargs="+", default=None, metavar="SYM",
                   help="Filter to these symbols (default: all)")
    args = p.parse_args()

    parts = args.date.split(":")
    if len(parts) != 2:
        sys.exit("ERROR: --date must be YYYY-MM-DD:YYYY-MM-DD")
    range_start = datetime.strptime(parts[0], "%Y-%m-%d").date()
    range_end = datetime.strptime(parts[1], "%Y-%m-%d").date()

    symbol_filter: Optional[set[str]] = None
    if args.symbol:
        symbol_filter = {s.upper() for s in args.symbol}
        unknown = symbol_filter - set(SYMBOL_MAP_REV.values())
        if unknown:
            sys.exit(f"ERROR: unknown symbol(s): {', '.join(unknown)}")

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    os.makedirs(data_dir, exist_ok=True)

    import databento as db
    client = db.Historical(key=_load_api_key())

    claimed = 0
    skipped = 0
    failed: list[tuple[str, str, str]] = []  # (job_id, symbol, date)
    for j in client.batch.list_jobs():
        if j.get("state") != "done":
            continue
        if j.get("schema") != "mbo":
            continue
        syms_raw = j.get("symbols", "")
        syms = syms_raw.split(",") if isinstance(syms_raw, str) else list(syms_raw)
        if len(syms) != 1 or syms[0] not in SYMBOL_MAP_REV:
            continue

        sym = SYMBOL_MAP_REV[syms[0]]
        if symbol_filter is not None and sym not in symbol_filter:
            continue
        start = _parse_iso(j["start"])
        end = _parse_iso(j["end"])

        if (end - start) != timedelta(days=1):
            continue
        if start.date() < range_start or start.date() > range_end:
            continue

        target = os.path.join(data_dir, make_dbn_filename(sym, start, end))
        if os.path.exists(target):
            print(f"  skip (exists): {os.path.basename(target)}")
            skipped += 1
            continue

        print(f"  claim [{sym}] {start.date()} ← {j['id']}")
        try:
            paths = client.batch.download(job_id=j["id"], output_dir=data_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED — {exc}")
            failed.append((j["id"], sym, str(start.date())))
            continue
        if isinstance(paths, (str, os.PathLike)):
            paths = [str(paths)]
        else:
            paths = [str(p) for p in paths]
        for p_ in paths:
            if p_.endswith(".dbn.zst"):
                renamed = rename_batch_to_local(p_, sym, start, end)
                final_path = os.path.join(data_dir, os.path.basename(renamed))
                if os.path.abspath(renamed) != os.path.abspath(final_path):
                    os.rename(renamed, final_path)
                break
        claimed += 1

    print(f"\nClaimed {claimed} file(s), {skipped} already present, {len(failed)} failed.")
    if failed:
        print("\nFailed jobs (likely expired — fetch_data.py will resubmit these):")
        for jid, sym, date in failed:
            print(f"  [{sym}] {date} job={jid}")


if __name__ == "__main__":
    main()
