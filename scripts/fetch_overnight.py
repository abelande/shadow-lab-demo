#!/usr/bin/env python3
"""Fetch overnight (1800 ET → 0900 ET) MBO data and run morning reports.

Downloads custom time-range files from Databento for overnight analysis.
Output filenames encode the UTC time range:
    {symbol}-mbo-{YYYYMMDD}T{HHMM}Z-{YYYYMMDD}T{HHMM}Z.dbn.zst
"""
import os
import sys
import asyncio
from datetime import datetime, timedelta, timezone

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.dirname(_project_root))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.dirname(_project_root))

# Ensure p6v2 symlink
_pkg_link = os.path.join(os.path.dirname(_project_root), "p6v2")
if not os.path.exists(_pkg_link):
    os.symlink(_project_root, _pkg_link)

DATA_DIR = os.path.join(_project_root, "data")
os.makedirs(DATA_DIR, exist_ok=True)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from ingestion.dbn_filename import make_dbn_filename

ET = ZoneInfo("America/New_York")


def load_key():
    env_path = os.path.join(_script_dir, ".env")
    if not os.path.exists(env_path):
        env_path = os.path.join(_project_root, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("DATABENTO_API_KEY="):
                    return line.strip().split("=", 1)[1]
    return os.getenv("DATABENTO_API_KEY", "")


def overnight_window_utc(morning_date: str):
    """Given a morning date (YYYY-MM-DD), return (start_utc, end_utc) for 1800 ET night before → 0900 ET morning."""
    from datetime import date as _d
    d = _d.fromisoformat(morning_date)
    prev = d - timedelta(days=1)
    
    start_et = datetime(prev.year, prev.month, prev.day, 18, 0, tzinfo=ET)
    end_et = datetime(d.year, d.month, d.day, 9, 0, tzinfo=ET)
    
    return start_et.astimezone(timezone.utc), end_et.astimezone(timezone.utc)


def fetch_overnight(client, symbol: str, morning_date: str, skip_existing: bool = True):
    sym_map = {"NQ": "NQ.c.0", "ES": "ES.c.0", "CL": "CL.c.0", "GC": "GC.c.0", "SI": "SI.c.0"}
    if symbol not in sym_map:
        print(f"Unknown symbol {symbol}")
        return None
    
    start_utc, end_utc = overnight_window_utc(morning_date)
    out_path = os.path.join(DATA_DIR, make_dbn_filename(symbol, start_utc, end_utc))
    
    if skip_existing and os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1_048_576
        print(f"  [{symbol}] Already exists ({size_mb:.1f} MB), skipping: {out_path}")
        return out_path
    
    print(f"\n[{symbol}] Fetching overnight for morning of {morning_date}")
    print(f"  Window: {start_utc.strftime('%Y-%m-%d %H:%M UTC')} → {end_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  (1800 ET {(start_utc - timedelta(hours=-5)).strftime('%m-%d')} → 0900 ET {morning_date})")
    
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[sym_map[symbol]],
        schema="mbo",
        start=start_utc.isoformat(),
        end=end_utc.isoformat(),
        stype_in="continuous",
    )
    data.to_file(out_path)
    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"  Done. {size_mb:.1f} MB → {out_path}")
    return out_path


async def run_morning_report(file_path: str, symbol: str):
    from morning_report import run_report
    result = await run_report(file_path, symbol_override=symbol)
    return result


async def main():
    import databento as db
    key = load_key()
    if not key:
        sys.exit("No DATABENTO_API_KEY found")
    
    client = db.Historical(key=key)
    symbol = "NQ"
    dates = ["2026-03-25", "2026-03-26"]
    
    # Fetch data
    paths = []
    for d in dates:
        path = fetch_overnight(client, symbol, d)
        if path:
            paths.append((d, path))
    
    if not paths:
        print("No data fetched.")
        return
    
    # Run morning reports
    print("\n" + "=" * 63)
    print("  RUNNING MORNING REPORTS")
    print("=" * 63)
    
    results = []
    for date, path in paths:
        print(f"\n{'─' * 63}")
        print(f"  Processing overnight → morning {date}")
        print(f"{'─' * 63}\n")
        result = await run_morning_report(path, symbol)
        results.append(result)
    
    # Comparative summary if multiple
    if len(results) >= 2:
        from morning_report import _comparative_summary, SessionResult
        comp = _comparative_summary(results)
        if comp:
            print(comp)


if __name__ == "__main__":
    asyncio.run(main())
