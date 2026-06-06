"""Utilities for encoding/decoding time ranges in .dbn.zst filenames.

Two naming conventions are supported:

1. Local (timeseries.get_range downloads) — encodes the exact UTC time range:
       {symbol}-mbo-{YYYYMMDD}T{HHMM}Z-{YYYYMMDD}T{HHMM}Z.dbn.zst
   Examples:
       nq-mbo-20260327T0000Z-20260328T0000Z.dbn.zst   (full day)
       nq-mbo-20260324T2300Z-20260325T1400Z.dbn.zst   (overnight)

2. Batch API (client.batch.download) — Databento's server-side naming:
       {dataset}-{YYYYMMDD}.{schema}.dbn.zst
       {dataset}-{YYYYMMDD}-{SYMBOL}.{schema}.dbn.zst
   Examples:
       glbx-mdp3-20260327.mbo.dbn.zst
       glbx-mdp3-20260327-NQ.mbo.dbn.zst
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

_LOCAL_RE = re.compile(
    r"^([a-zA-Z]+)-mbo-"
    r"(\d{8})T(\d{4})Z"
    r"-"
    r"(\d{8})T(\d{4})Z"
    r"\.dbn\.zst$"
)

# Batch forms: "{dataset}-{YYYYMMDD}.{schema}.dbn.zst"
# or           "{dataset}-{YYYYMMDD}-{SYMBOL}.{schema}.dbn.zst"
# Dataset segments use letters/digits separated by hyphens (e.g. glbx-mdp3, xnas-itch).
_BATCH_RE = re.compile(
    r"^(?P<dataset>[a-z0-9]+(?:-[a-z0-9]+)*)"
    r"-(?P<date>\d{8})"
    r"(?:-(?P<symbol>[A-Za-z0-9.]+))?"
    r"\.(?P<schema>[a-z0-9_]+)"
    r"\.dbn\.zst$"
)


def make_dbn_filename(symbol: str, start: datetime, end: datetime) -> str:
    """Return a local-convention .dbn.zst filename for a UTC time range.

    Args:
        symbol: Instrument root (e.g. "NQ", "ES").
        start:  Inclusive start (must be UTC-aware).
        end:    Exclusive end (must be UTC-aware).
    """
    s = start.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ")
    e = end.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ")
    return f"{symbol.lower()}-mbo-{s}-{e}.dbn.zst"


def parse_dbn_filename(path: str) -> Optional[tuple[str, datetime, datetime]]:
    """Extract (symbol, start_utc, end_utc) from either naming convention.

    Returns None if the filename does not match any supported format.
    For batch-style filenames without an embedded symbol, symbol is "".
    Batch filenames imply a full UTC day [00:00, next-day 00:00).
    """
    name = os.path.basename(path)

    m = _LOCAL_RE.match(name)
    if m:
        symbol = m.group(1).upper()
        start = datetime.strptime(f"{m.group(2)}{m.group(3)}", "%Y%m%d%H%M").replace(
            tzinfo=timezone.utc
        )
        end = datetime.strptime(f"{m.group(4)}{m.group(5)}", "%Y%m%d%H%M").replace(
            tzinfo=timezone.utc
        )
        return symbol, start, end

    m = _BATCH_RE.match(name)
    if m:
        symbol = (m.group("symbol") or "").upper()
        start = datetime.strptime(m.group("date"), "%Y%m%d").replace(tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        return symbol, start, end

    return None


def is_batch_filename(path: str) -> bool:
    """True if the filename matches Databento's batch-delivery convention."""
    return _BATCH_RE.match(os.path.basename(path)) is not None


def rename_batch_to_local(
    path: str,
    symbol: str,
    start: datetime,
    end: datetime,
) -> str:
    """Rename a batch-delivered file in-place to the local convention.

    Returns the new path. No-op (returns original path) if the file already
    matches the local convention.
    """
    if _LOCAL_RE.match(os.path.basename(path)):
        return path
    new_name = make_dbn_filename(symbol, start, end)
    new_path = os.path.join(os.path.dirname(path), new_name)
    os.rename(path, new_path)
    return new_path
