#!/usr/bin/env python3
"""Download Databento GLBX.MDP3 MBO data for morning report.

Two transport modes:

  stream  — client.timeseries.get_range (synchronous, good for small/intraday)
  batch   — client.batch.submit_job + download (async, much faster for bulk)

Usage:
    python fetch_data.py                                  # yesterday, NQ+ES, auto mode
    python fetch_data.py --symbol NQ                      # yesterday, NQ only
    python fetch_data.py --date 2026-03-23                # specific date
    python fetch_data.py --date today                     # today's rolling window
    python fetch_data.py --date today --symbol NQ ES CL
    python fetch_data.py --date 2026-04-10:2026-04-14 --symbol NQ ES   # date range
    python fetch_data.py --mode batch  --symbol NQ ES CL  # force batch
    python fetch_data.py --mode stream --symbol NQ        # force stream

Auto mode: uses 'batch' when >=2 full-day jobs requested, else 'stream'.

Output filenames always use the local convention:
    {symbol}-mbo-{YYYYMMDD}T{HHMM}Z-{YYYYMMDD}T{HHMM}Z.dbn.zst
Batch-delivered files are renamed after download.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "ingestion"))
from dbn_filename import make_dbn_filename, rename_batch_to_local  # noqa: E402

logger = logging.getLogger(__name__)

DATASET = "GLBX.MDP3"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SYMBOL_MAP = {
    "NQ": "NQ.c.0",
    "ES": "ES.c.0",
    "CL": "CL.c.0",
    "GC": "GC.c.0",
    "SI": "SI.c.0",
}

MIN_INTRADAY_HOURS = 1.0
BATCH_POLL_INTERVAL_SEC = 10
BATCH_POLL_TIMEOUT_SEC = 60 * 30  # 30 min hard cap
MAX_PARALLEL_DOWNLOADS = 8
BATCH_SUBMIT_PACING_SEC = 0.35  # gentle pacing between submits to avoid 429s
BATCH_SUBMIT_MAX_RETRIES = 6


# ───────────────────────────────── shared helpers ─────────────────────────────────


def _load_api_key() -> str:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DATABENTO_API_KEY="):
                    return line.split("=", 1)[1].strip()
    key = os.getenv("DATABENTO_API_KEY", "")
    if not key:
        sys.exit("ERROR: DATABENTO_API_KEY not found in .env or environment.")
    return key


def _get_mbo_rolling_end(client) -> datetime:
    info = client.metadata.get_dataset_range(dataset=DATASET)
    end_str = info["schema"]["mbo"]["end"]
    end_str = end_str.rstrip("Z").split(".")[0]
    return datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)


def resolve_window(date_str: str, client) -> tuple[datetime, datetime, bool]:
    """Return (start, end, is_full_day) UTC datetimes for a downloadable MBO window.

    is_full_day is True when the window spans a complete UTC day (00:00 → 00:00)
    — the batch API is only a clean fit in that case.
    """
    now = datetime.now(timezone.utc)

    if date_str == "yesterday":
        target = (now - timedelta(days=1)).date()
    elif date_str == "today":
        target = now.date()
    else:
        try:
            target = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(
                f"Invalid date format '{date_str}' — use YYYY-MM-DD, today, or yesterday."
            )

    today = now.date()
    if target > today:
        raise ValueError(f"{target} is in the future.")

    conditions = client.metadata.get_dataset_condition(
        dataset=DATASET,
        start_date=str(target),
        end_date=str(target),
    )
    if not conditions:
        raise ValueError(
            f"No data available for {target} (weekend, holiday, or outside dataset range)."
        )

    cond = conditions[0].get("condition", "unknown")
    if cond not in ("available", "degraded"):
        raise ValueError(f"Data for {target} has condition '{cond}' — not downloadable.")

    start = datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    is_full_day = True

    # Cap the window at the rolling MBO cutoff — the license may not cover
    # the trailing portion of any date that butts up against "now".
    rolling_end = _get_mbo_rolling_end(client)
    if end > rolling_end:
        available_hours = (rolling_end - start).total_seconds() / 3600
        print(f"  Rolling MBO cutoff: {rolling_end.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Capped window:      {available_hours:.1f}h (license boundary)")
        if available_hours < MIN_INTRADAY_HOURS:
            raise ValueError(
                f"Window only has {available_hours:.1f}h of licensed data "
                f"(minimum {MIN_INTRADAY_HOURS}h required)."
            )
        end = rolling_end
        is_full_day = False

    return start, end, is_full_day


def _display_date(date_str: str) -> str:
    now = datetime.now(timezone.utc)
    if date_str == "yesterday":
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    if date_str == "today":
        return now.strftime("%Y-%m-%d")
    return date_str


def _target_path(symbol: str, start: datetime, end: datetime) -> str:
    return os.path.join(DATA_DIR, make_dbn_filename(symbol, start, end))


# ───────────────────────────────── stream transport ─────────────────────────────────


def fetch_stream(
    symbol: str,
    start: datetime,
    end: datetime,
    client,
    skip_existing: bool,
) -> Optional[str]:
    """Download one (symbol, window) via timeseries.get_range. Returns output path."""
    out_path = _target_path(symbol, start, end)
    if skip_existing and os.path.exists(out_path):
        print(f"  [{symbol}] Already exists, skipping: {out_path}")
        return out_path

    print(f"  [{symbol}] stream → {out_path}")
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[SYMBOL_MAP[symbol]],
        schema="mbo",
        start=start.isoformat(),
        end=end.isoformat(),
        stype_in="continuous",
    )
    data.to_file(out_path)
    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"  [{symbol}] done, {size_mb:.1f} MB")
    return out_path


def run_stream(
    requests: list[tuple[str, datetime, datetime]],
    client,
    skip_existing: bool,
    max_workers: int = 4,
) -> list[str]:
    """Run many stream downloads in parallel (I/O-bound, thread-safe)."""
    if not requests:
        return []

    results: list[str] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(requests))) as pool:
        futures = {
            pool.submit(fetch_stream, sym, start, end, client, skip_existing): sym
            for sym, start, end in requests
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                path = fut.result()
                if path:
                    results.append(path)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{sym}] stream FAILED — {exc}")
    return results


# ───────────────────────────────── batch transport ─────────────────────────────────


_TRANSIENT_MARKERS = ("429", "Too Many Requests", "500", "502", "503", "504",
                      "gateway", "timed out", "timeout")


def _submit_batch_job(
    client,
    symbol: str,
    start: datetime,
    end: datetime,
) -> dict:
    """Submit one batch job. Retries on 429 and transient 5xx errors."""
    for attempt in range(BATCH_SUBMIT_MAX_RETRIES):
        try:
            return client.batch.submit_job(
                dataset=DATASET,
                symbols=[SYMBOL_MAP[symbol]],
                schema="mbo",
                start=start.isoformat(),
                end=end.isoformat(),
                stype_in="continuous",
                encoding="dbn",
                compression="zstd",
                split_duration="day",
                delivery="download",
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if not any(marker.lower() in msg.lower() for marker in _TRANSIENT_MARKERS):
                raise
            m = re.search(r"Retry in (\d+)s", msg)
            if m:
                delay = int(m.group(1))
                kind = "429 rate limit"
            else:
                delay = min(60, 2 ** attempt)
                kind = "transient error"
            print(f"    [{symbol}] {kind} — sleeping {delay}s (attempt {attempt + 1}/{BATCH_SUBMIT_MAX_RETRIES}): {msg[:120]}")
            time.sleep(delay + 1)  # +1 for clock skew
    raise RuntimeError(f"[{symbol}] batch submit failed after {BATCH_SUBMIT_MAX_RETRIES} retries")


def _wait_for_jobs(client, job_ids: set[str]) -> None:
    """Block until every job_id is in state 'done'. Raises on timeout/failure."""
    deadline = time.monotonic() + BATCH_POLL_TIMEOUT_SEC
    pending = set(job_ids)
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Batch jobs did not finish within {BATCH_POLL_TIMEOUT_SEC}s: {pending}")
        jobs = client.batch.list_jobs()
        by_id = {j["id"]: j for j in jobs if j["id"] in pending}
        for jid, rec in by_id.items():
            state = rec.get("state")
            if state == "done":
                pending.discard(jid)
            elif state in ("expired", "failed"):
                raise RuntimeError(f"Batch job {jid} ended in state '{state}'")
        if pending:
            print(f"  batch: {len(pending)} job(s) still running...")
            time.sleep(BATCH_POLL_INTERVAL_SEC)


def _download_and_rename(
    client,
    job_id: str,
    symbol: str,
    start: datetime,
    end: datetime,
) -> Optional[str]:
    """Download a finished job's files and rename the MBO artifact to local convention.

    The SDK places files in a per-job subdirectory under output_dir. We flatten
    the .dbn.zst up to DATA_DIR so fetch_data.py's skip_existing check matches;
    JSON sidecars stay in the subdirectory for provenance.
    """
    paths = client.batch.download(job_id=job_id, output_dir=DATA_DIR)
    if isinstance(paths, (str, os.PathLike)):
        paths = [str(paths)]
    else:
        paths = [str(p) for p in paths]

    for p in paths:
        base = os.path.basename(p)
        if base.endswith(".dbn.zst"):
            renamed = rename_batch_to_local(p, symbol, start, end)
            final_path = os.path.join(DATA_DIR, os.path.basename(renamed))
            if os.path.abspath(renamed) != os.path.abspath(final_path):
                os.rename(renamed, final_path)
            return final_path
    return None


def run_batch(
    requests: list[tuple[str, datetime, datetime]],
    client,
    skip_existing: bool,
) -> list[str]:
    """Submit all (symbol, window) jobs, wait, then download in parallel."""
    if not requests:
        return []

    # Filter already-present files before paying for jobs
    to_submit: list[tuple[str, datetime, datetime]] = []
    results: list[str] = []
    for sym, start, end in requests:
        out_path = _target_path(sym, start, end)
        if skip_existing and os.path.exists(out_path):
            print(f"  [{sym}] Already exists, skipping: {out_path}")
            results.append(out_path)
        else:
            to_submit.append((sym, start, end))

    if not to_submit:
        return results

    # 1. Submit
    print(f"  batch: submitting {len(to_submit)} job(s)...")
    submitted: list[tuple[str, str, datetime, datetime]] = []  # (job_id, sym, start, end)
    for sym, start, end in to_submit:
        job = _submit_batch_job(client, sym, start, end)
        jid = job["id"]
        submitted.append((jid, sym, start, end))
        print(f"    [{sym}] job {jid} queued")
        time.sleep(BATCH_SUBMIT_PACING_SEC)

    # 2. Poll
    _wait_for_jobs(client, {j[0] for j in submitted})

    # 3. Download in parallel
    print(f"  batch: downloading {len(submitted)} artifact(s)...")
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_DOWNLOADS, len(submitted))) as pool:
        futures = {
            pool.submit(_download_and_rename, client, jid, sym, start, end): sym
            for jid, sym, start, end in submitted
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                path = fut.result()
                if path:
                    size_mb = os.path.getsize(path) / 1_048_576
                    print(f"  [{sym}] done, {size_mb:.1f} MB → {path}")
                    results.append(path)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{sym}] batch download FAILED — {exc}")

    return results


# ───────────────────────────────── orchestration ─────────────────────────────────


def expand_dates(date_str: str) -> list[str]:
    """Expand a date string into a list of date strings.

    Accepts:
        "yesterday"                 → ["yesterday"]
        "today"                     → ["today"]
        "2026-04-14"                → ["2026-04-14"]
        "2026-04-10:2026-04-14"     → ["2026-04-10", "2026-04-11", ..., "2026-04-14"]
    """
    if ":" not in date_str:
        return [date_str]

    parts = date_str.split(":", 1)
    try:
        range_start = datetime.strptime(parts[0], "%Y-%m-%d").date()
        range_end = datetime.strptime(parts[1], "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            f"Invalid date range '{date_str}' — use YYYY-MM-DD:YYYY-MM-DD"
        )

    if range_start > range_end:
        raise ValueError(f"Range start {range_start} is after end {range_end}.")

    dates = []
    cursor = range_start
    while cursor <= range_end:
        dates.append(str(cursor))
        cursor += timedelta(days=1)
    return dates


def build_requests(
    symbols: list[str],
    date_str: str,
    client,
) -> tuple[list[tuple[str, datetime, datetime]], bool]:
    """Resolve windows for all symbols × all dates. Returns (requests, all_full_day)."""
    dates = expand_dates(date_str)
    requests: list[tuple[str, datetime, datetime]] = []
    all_full = True

    for d in dates:
        display = _display_date(d)
        for raw in symbols:
            sym = raw.upper()
            if sym not in SYMBOL_MAP:
                print(f"  Unknown symbol '{sym}'. Valid: {', '.join(SYMBOL_MAP)}")
                continue
            print(f"\n[{sym}] Checking window for {display}...")
            try:
                start, end, full = resolve_window(d, client)
            except ValueError as e:
                print(f"  [{sym}] INVALID WINDOW — {e}")
                continue
            print(f"  Window: {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} UTC")
            requests.append((sym, start, end))
            all_full = all_full and full

    return requests, all_full


def pick_mode(mode: str, requests: list, all_full_day: bool) -> str:
    """Resolve 'auto' to 'stream' or 'batch' based on request profile."""
    if mode != "auto":
        return mode
    # Batch is only a clean fit for full-day historical windows, and pays off at ≥2.
    if all_full_day and len(requests) >= 2:
        return "batch"
    return "stream"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Databento MBO data.")
    parser.add_argument("--date", default="yesterday",
                        help="Date(s) to fetch: YYYY-MM-DD, YYYY-MM-DD:YYYY-MM-DD range, "
                             "'today', or 'yesterday' (default: yesterday)")
    parser.add_argument("--symbol", nargs="+", default=["NQ", "ES"], metavar="SYM",
                        help=f"Symbols (default: NQ ES). Options: {' '.join(SYMBOL_MAP)}")
    parser.add_argument("--mode", choices=["auto", "stream", "batch"], default="auto",
                        help="Transport: stream (get_range), batch (submit_job), or auto.")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-download even if file already exists")
    parser.add_argument("--data-dir", default=None,
                        help="Output directory (default: ../data relative to this script)")
    args = parser.parse_args()

    global DATA_DIR
    if args.data_dir:
        DATA_DIR = os.path.abspath(os.path.expanduser(args.data_dir))

    import databento as db
    client = db.Historical(key=_load_api_key())
    os.makedirs(DATA_DIR, exist_ok=True)
    skip_existing = not args.no_skip

    requests, all_full = build_requests(args.symbol, args.date, client)
    if not requests:
        print("\nNo valid windows to download.")
        return

    mode = pick_mode(args.mode, requests, all_full)
    print(f"\nTransport: {mode} ({len(requests)} window(s))")

    if mode == "batch":
        downloaded = run_batch(requests, client, skip_existing)
    else:
        downloaded = run_stream(requests, client, skip_existing)

    if downloaded:
        print("\nReady to run morning report:")
        if len(downloaded) == 1:
            print(f"  python morning_report.py {downloaded[0]}")
        else:
            print(f"  python morning_report.py --files {' '.join(downloaded)}")


if __name__ == "__main__":
    main()
