#!/usr/bin/env python3
"""
Morning Prep Report — FAST (Polars vectorized)

Same report format as morning_report.py but uses BatchScanner's to_ndarray()
Rust path (0.3s load) instead of the 38+ minute streaming DatabentoReplayFeed.

Spoof detection reimplemented as vectorized Polars joins/groupby operations.

Usage:
    python morning_report_fast.py data/nq-mbo-20260412T2200Z-20260413T1613Z.dbn.zst
    python morning_report_fast.py data/nq-mbo-2026-03-27.dbn.zst --rth-only
    python morning_report_fast.py --files data/nq-mbo-2026-03-24.dbn.zst data/nq-mbo-2026-03-27.dbn.zst
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_project_root)
_pkg_link = os.path.join(_parent, "p6v2")
if not os.path.exists(_pkg_link):
    os.symlink(_project_root, _pkg_link)
sys.path.insert(0, _parent)

from p6v2.ingestion.batch_scanner import BatchScanner

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

_ET = ZoneInfo("America/New_York")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ── Instrument Config (same as morning_report.py) ─────────────────

@dataclass
class InstrumentConfig:
    symbol: str
    tick_size: float
    point_value: float
    bucket_size: float
    zone_bucket: float
    persistence_pct: float
    persistence_min: int
    iceberg_vol_threshold: float
    stop_run_levels: int
    session_tz: str = "America/New_York"


_INSTRUMENT_CONFIGS: Dict[str, InstrumentConfig] = {
    "NQ": InstrumentConfig(
        symbol="NQ", tick_size=0.25, point_value=20.0, bucket_size=0.25,
        zone_bucket=5.0, persistence_pct=0.01, persistence_min=50,
        iceberg_vol_threshold=10.0, stop_run_levels=5,
    ),
    "ES": InstrumentConfig(
        symbol="ES", tick_size=0.25, point_value=50.0, bucket_size=0.25,
        zone_bucket=2.0, persistence_pct=0.01, persistence_min=50,
        iceberg_vol_threshold=30.0, stop_run_levels=3,
    ),
    "CL": InstrumentConfig(
        symbol="CL", tick_size=0.01, point_value=1000.0, bucket_size=0.01,
        zone_bucket=0.25, persistence_pct=0.01, persistence_min=50,
        iceberg_vol_threshold=20.0, stop_run_levels=5,
    ),
    "GC": InstrumentConfig(
        symbol="GC", tick_size=0.10, point_value=100.0, bucket_size=0.10,
        zone_bucket=2.0, persistence_pct=0.01, persistence_min=50,
        iceberg_vol_threshold=15.0, stop_run_levels=4,
    ),
    "SI": InstrumentConfig(
        symbol="SI", tick_size=0.005, point_value=5000.0, bucket_size=0.005,
        zone_bucket=0.10, persistence_pct=0.02, persistence_min=30,
        iceberg_vol_threshold=5.0, stop_run_levels=4,
    ),
}


def _detect_instrument(file_path: str, symbol_override: Optional[str] = None) -> InstrumentConfig:
    if symbol_override:
        key = symbol_override.upper()
        if key in _INSTRUMENT_CONFIGS:
            return _INSTRUMENT_CONFIGS[key]
        root = "".join(c for c in key if c.isalpha())
        if root in _INSTRUMENT_CONFIGS:
            return _INSTRUMENT_CONFIGS[root]
        raise ValueError(f"Unknown symbol '{symbol_override}'. Available: {list(_INSTRUMENT_CONFIGS.keys())}")
    name = os.path.basename(file_path).upper()
    for sym, cfg in _INSTRUMENT_CONFIGS.items():
        if name.startswith(sym):
            return cfg
    if name.startswith("GLBX"):
        raise ValueError(f"Full-exchange file. Use --symbol (available: {list(_INSTRUMENT_CONFIGS.keys())})")
    return _INSTRUMENT_CONFIGS["NQ"]


# ── Time helpers ───────────────────────────────────────────────────

def _ns_to_ms(ns: int) -> int:
    return ns // 1_000_000

def _ms_to_et(ts_ms: int) -> str:
    if ts_ms == 0:
        return "??:??:??"
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).astimezone(_ET)
    return dt.strftime("%H:%M:%S")

def _ms_to_et_date(ts_ms: int) -> str:
    if ts_ms == 0:
        return ""
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).astimezone(_ET)
    return dt.strftime("%A %B %d %Y")


# ── Spoof Detection — Vectorized Polars ────────────────────────────

def detect_pull_before_touch(df: pl.DataFrame, cfg: InstrumentConfig) -> pl.DataFrame:
    """Pull-Before-Touch: ADD then CANCEL of same order within 150ms, size>=3.

    Returns DataFrame with columns: price_f, side, confidence, first_ts_ms, last_ts_ms, event_count
    """
    adds = df.filter(pl.col("action") == "A").select([
        pl.col("order_id"),
        pl.col("ts_ns").alias("add_ts"),
        pl.col("price_f").alias("add_price"),
        pl.col("size").alias("add_size"),
        pl.col("side"),
    ])

    cancels = df.filter(pl.col("action") == "C").select([
        pl.col("order_id"),
        pl.col("ts_ns").alias("cancel_ts"),
    ])

    # Join adds with their cancels
    joined = adds.join(cancels, on="order_id", how="inner")

    # Filter: cancel within 150ms of add, size >= 3
    pulls = joined.filter(
        ((pl.col("cancel_ts") - pl.col("add_ts")) > 0) &
        ((pl.col("cancel_ts") - pl.col("add_ts")) < 150_000_000) &  # 150ms in ns
        (pl.col("add_size") >= 3)
    )

    if pulls.is_empty():
        return pl.DataFrame(schema={
            "price_f": pl.Float64, "side": pl.Utf8, "spoof_type": pl.Utf8,
            "confidence": pl.Float64, "first_ts_ms": pl.Int64,
            "last_ts_ms": pl.Int64, "event_count": pl.UInt32,
        })

    # Bucket by price
    bucket = cfg.bucket_size
    result = (
        pulls.with_columns([
            (pl.col("add_price") / bucket).round(0).cast(pl.Float64).mul(bucket).alias("price_bucket"),
            (pl.col("add_ts") // 1_000_000).cast(pl.Int64).alias("ts_ms"),
        ])
        .group_by(["price_bucket", "side"])
        .agg([
            pl.len().alias("event_count"),
            pl.col("ts_ms").min().alias("first_ts_ms"),
            pl.col("ts_ms").max().alias("last_ts_ms"),
            pl.col("add_size").max().alias("max_size"),
        ])
        .filter(pl.col("event_count") >= 2)  # Need 2+ pulls at same price
        .with_columns([
            pl.col("price_bucket").alias("price_f"),
            pl.lit("pull_before_touch").alias("spoof_type"),
            # Confidence: more events + bigger sizes = higher
            (pl.col("event_count").cast(pl.Float64) / 20.0).clip(0.2, 0.95).alias("confidence"),
        ])
        .select(["price_f", "side", "spoof_type", "confidence", "first_ts_ms", "last_ts_ms", "event_count"])
    )
    return result


def detect_layering(df: pl.DataFrame, cfg: InstrumentConfig) -> pl.DataFrame:
    """Layering: 3+ ADDs on same side with similar size within 300ms at consecutive prices.

    Returns DataFrame with columns: price_f, side, confidence, first_ts_ms, last_ts_ms, event_count
    """
    adds = df.filter(
        (pl.col("action") == "A") & (pl.col("size") >= 2)
    ).select([
        pl.col("ts_ns"),
        pl.col("price_f"),
        pl.col("size"),
        pl.col("side"),
        (pl.col("ts_ns") // 1_000_000).cast(pl.Int64).alias("ts_ms"),
    ])

    if adds.is_empty():
        return pl.DataFrame(schema={
            "price_f": pl.Float64, "side": pl.Utf8, "spoof_type": pl.Utf8,
            "confidence": pl.Float64, "first_ts_ms": pl.Int64,
            "last_ts_ms": pl.Int64, "event_count": pl.UInt32,
        })

    # Bucket into 300ms windows and group by (side, size_bucket, time_window)
    tick = cfg.tick_size
    result = (
        adds.with_columns([
            (pl.col("ts_ns") // 300_000_000).alias("time_window"),  # 300ms windows
            (pl.col("size") // 1).alias("size_bucket"),  # exact size match
            (pl.col("price_f") / tick).round(0).alias("price_ticks"),
        ])
        .group_by(["side", "size_bucket", "time_window"])
        .agg([
            pl.len().alias("n_orders"),
            pl.col("price_ticks").n_unique().alias("n_levels"),
            pl.col("price_f").mean().alias("avg_price"),
            pl.col("ts_ms").min().alias("first_ts_ms"),
            pl.col("ts_ms").max().alias("last_ts_ms"),
        ])
        .filter(
            (pl.col("n_orders") >= 3) & (pl.col("n_levels") >= 3)  # 3+ orders at 3+ price levels
        )
    )

    if result.is_empty():
        return pl.DataFrame(schema={
            "price_f": pl.Float64, "side": pl.Utf8, "spoof_type": pl.Utf8,
            "confidence": pl.Float64, "first_ts_ms": pl.Int64,
            "last_ts_ms": pl.Int64, "event_count": pl.UInt32,
        })

    bucket = cfg.bucket_size
    result = (
        result.with_columns([
            (pl.col("avg_price") / bucket).round(0).cast(pl.Float64).mul(bucket).alias("price_f"),
            pl.lit("layering").alias("spoof_type"),
            (pl.col("n_orders").cast(pl.Float64) / 10.0).clip(0.3, 0.95).alias("confidence"),
            pl.col("n_orders").cast(pl.UInt32).alias("event_count"),
        ])
        .group_by(["price_f", "side"])
        .agg([
            pl.col("spoof_type").first(),
            pl.col("confidence").max(),
            pl.col("first_ts_ms").min(),
            pl.col("last_ts_ms").max(),
            pl.col("event_count").sum(),
        ])
        .select(["price_f", "side", "spoof_type", "confidence", "first_ts_ms", "last_ts_ms", "event_count"])
    )
    return result


def detect_icebergs(df: pl.DataFrame, cfg: InstrumentConfig) -> pl.DataFrame:
    """Iceberg: Repeated fills at same (side, price) where visible sizes are small but total large.

    Returns DataFrame with columns: price_f, side, total_volume, total_refills,
    first_ts_ms, last_ts_ms, detections, max_confidence
    """
    fills = df.filter(pl.col("action").is_in(["F", "T"]))

    if fills.is_empty():
        return pl.DataFrame(schema={
            "price_f": pl.Float64, "side": pl.Utf8, "total_volume": pl.Float64,
            "total_refills": pl.UInt32, "first_ts_ms": pl.Int64,
            "last_ts_ms": pl.Int64, "detections": pl.UInt32, "max_confidence": pl.Float64,
        })

    # Group fills by (price, side)
    bucket = cfg.bucket_size
    result = (
        fills.with_columns([
            (pl.col("price_f") / bucket).round(0).cast(pl.Float64).mul(bucket).alias("price_bucket"),
            (pl.col("ts_ns") // 1_000_000).cast(pl.Int64).alias("ts_ms"),
        ])
        .group_by(["price_bucket", "side"])
        .agg([
            pl.len().alias("fill_count"),
            pl.col("size").sum().cast(pl.Float64).alias("total_volume"),
            pl.col("size").max().alias("max_visible_size"),
            pl.col("size").mean().alias("avg_visible_size"),
            pl.col("ts_ms").min().alias("first_ts_ms"),
            pl.col("ts_ms").max().alias("last_ts_ms"),
        ])
        # Iceberg pattern: many fills, small visible sizes, large total
        .filter(
            (pl.col("fill_count") >= 4) &
            (pl.col("avg_visible_size") < 8) &
            (pl.col("total_volume") >= 30)
        )
        .with_columns([
            pl.col("price_bucket").alias("price_f"),
            pl.col("fill_count").cast(pl.UInt32).alias("total_refills"),
            pl.col("fill_count").cast(pl.UInt32).alias("detections"),
            # Confidence based on volume and fill count
            ((pl.col("total_volume") / 100.0) * (pl.col("fill_count").cast(pl.Float64) / 10.0))
            .clip(0.3, 0.95).alias("max_confidence"),
        ])
        .select(["price_f", "side", "total_volume", "total_refills",
                  "first_ts_ms", "last_ts_ms", "detections", "max_confidence"])
    )
    return result


def detect_phantom_walls(df: pl.DataFrame, cfg: InstrumentConfig) -> pl.DataFrame:
    """Phantom Wall: Large ADD (>=50) canceled within 100-500ms.

    Returns DataFrame with columns: price_f, side, confidence, first_ts_ms, last_ts_ms, event_count
    """
    large_adds = df.filter(
        (pl.col("action") == "A") & (pl.col("size") >= 50)
    ).select([
        pl.col("order_id"),
        pl.col("ts_ns").alias("add_ts"),
        pl.col("price_f").alias("add_price"),
        pl.col("size").alias("add_size"),
        pl.col("side"),
    ])

    cancels = df.filter(pl.col("action") == "C").select([
        pl.col("order_id"),
        pl.col("ts_ns").alias("cancel_ts"),
    ])

    if large_adds.is_empty():
        return pl.DataFrame(schema={
            "price_f": pl.Float64, "side": pl.Utf8, "spoof_type": pl.Utf8,
            "confidence": pl.Float64, "first_ts_ms": pl.Int64,
            "last_ts_ms": pl.Int64, "event_count": pl.UInt32,
        })

    joined = large_adds.join(cancels, on="order_id", how="inner")

    phantoms = joined.filter(
        ((pl.col("cancel_ts") - pl.col("add_ts")) >= 100_000_000) &   # >= 100ms
        ((pl.col("cancel_ts") - pl.col("add_ts")) <= 500_000_000)     # <= 500ms
    )

    if phantoms.is_empty():
        return pl.DataFrame(schema={
            "price_f": pl.Float64, "side": pl.Utf8, "spoof_type": pl.Utf8,
            "confidence": pl.Float64, "first_ts_ms": pl.Int64,
            "last_ts_ms": pl.Int64, "event_count": pl.UInt32,
        })

    bucket = cfg.bucket_size
    result = (
        phantoms.with_columns([
            (pl.col("add_price") / bucket).round(0).cast(pl.Float64).mul(bucket).alias("price_bucket"),
            (pl.col("add_ts") // 1_000_000).cast(pl.Int64).alias("ts_ms"),
        ])
        .group_by(["price_bucket", "side"])
        .agg([
            pl.len().alias("event_count"),
            pl.col("ts_ms").min().alias("first_ts_ms"),
            pl.col("ts_ms").max().alias("last_ts_ms"),
            pl.col("add_size").max().alias("max_size"),
        ])
        .with_columns([
            pl.col("price_bucket").alias("price_f"),
            pl.lit("phantom_wall").alias("spoof_type"),
            (pl.col("event_count").cast(pl.Float64) / 5.0).clip(0.3, 0.95).alias("confidence"),
            pl.col("event_count").cast(pl.UInt32),
        ])
        .select(["price_f", "side", "spoof_type", "confidence", "first_ts_ms", "last_ts_ms", "event_count"])
    )
    return result


# ── Fragility Analysis — Vectorized ────────────────────────────────

def compute_fragility(df: pl.DataFrame, cfg: InstrumentConfig) -> pl.DataFrame:
    """Compute per-price-level fragility from order flow.

    Fragility = high volume concentration in few orders (top 1 order holds >50% of level volume).
    We approximate by looking at the distribution of ADD sizes per price level per side.

    Returns DataFrame with columns: price_f, side, total_snapshots, fragile_count,
    solid_count, moderate_count, avg_volume, avg_order_count, overlapping_spoof (False default)
    """
    # Only look at adds (they define book state)
    adds = df.filter(
        (pl.col("action") == "A") & (pl.col("size") > 0)
    ).select(["price_f", "side", "size", "ts_ns"])

    if adds.is_empty():
        return pl.DataFrame(schema={
            "price_f": pl.Float64, "side": pl.Utf8, "total_snapshots": pl.UInt32,
            "fragile_count": pl.UInt32, "solid_count": pl.UInt32,
            "moderate_count": pl.UInt32, "avg_volume": pl.Float64,
            "avg_order_count": pl.Float64, "overlapping_spoof": pl.Boolean,
        })

    bucket = cfg.bucket_size

    # Bucket into 500ms windows (matching the original 500ms snapshot interval)
    windowed = adds.with_columns([
        (pl.col("price_f") / bucket).round(0).cast(pl.Float64).mul(bucket).alias("price_bucket"),
        (pl.col("ts_ns") // 500_000_000).alias("time_window"),
    ])

    # Per (price, side, time_window): count orders, sum volume, compute max_order_share
    level_stats = (
        windowed.group_by(["price_bucket", "side", "time_window"])
        .agg([
            pl.len().alias("order_count"),
            pl.col("size").sum().cast(pl.Float64).alias("total_vol"),
            pl.col("size").max().cast(pl.Float64).alias("max_order_size"),
        ])
        .with_columns([
            # Concentration: fraction of volume from the largest order
            (pl.col("max_order_size") / pl.col("total_vol").clip(lower_bound=1.0)).alias("concentration"),
        ])
        .with_columns([
            # Classify fragility
            pl.when(
                (pl.col("concentration") > 0.5) | (pl.col("order_count") <= 2)
            ).then(pl.lit("FRAGILE"))
            .when(
                (pl.col("concentration") < 0.3) & (pl.col("order_count") >= 5)
            ).then(pl.lit("SOLID"))
            .otherwise(pl.lit("MODERATE"))
            .alias("state"),
        ])
    )

    # Aggregate across time windows per (price, side)
    result = (
        level_stats.group_by(["price_bucket", "side"])
        .agg([
            pl.len().alias("total_snapshots"),
            (pl.col("state") == "FRAGILE").sum().cast(pl.UInt32).alias("fragile_count"),
            (pl.col("state") == "SOLID").sum().cast(pl.UInt32).alias("solid_count"),
            (pl.col("state") == "MODERATE").sum().cast(pl.UInt32).alias("moderate_count"),
            pl.col("total_vol").mean().alias("avg_volume"),
            pl.col("order_count").mean().cast(pl.Float64).alias("avg_order_count"),
        ])
        .with_columns([
            pl.col("price_bucket").alias("price_f"),
            pl.col("total_snapshots").cast(pl.UInt32),
            pl.lit(False).alias("overlapping_spoof"),
        ])
        .select(["price_f", "side", "total_snapshots", "fragile_count", "solid_count",
                  "moderate_count", "avg_volume", "avg_order_count", "overlapping_spoof"])
    )
    return result


# ── Authenticity Zones ─────────────────────────────────────────────

def compute_authenticity_zones(
    df: pl.DataFrame,
    spoof_events: pl.DataFrame,  # combined spoof detections
    cfg: InstrumentConfig,
) -> Tuple[pl.DataFrame, float, int, int]:
    """Compute authenticity per (price_zone, 30min_window).

    Returns: (zones_df, avg_auth, low_auth_count, total_snapshots)
    """
    # Get fills for mid-price approximation
    fills = df.filter(pl.col("action").is_in(["F", "T"]))

    if fills.is_empty() or spoof_events.is_empty():
        return pl.DataFrame(), 1.0, 0, 0

    # Time-based authenticity: more spoof events in a window = lower authenticity
    zone_bucket = cfg.zone_bucket

    # Build 30-min windows of spoof density
    if "first_ts_ms" not in spoof_events.columns:
        return pl.DataFrame(), 1.0, 0, 0

    spoof_density = (
        spoof_events.with_columns([
            (pl.col("first_ts_ms") // (30 * 60 * 1000)).alias("time_window"),
            (pl.col("price_f") / zone_bucket).round(0).cast(pl.Float64).mul(zone_bucket).alias("zone"),
        ])
        .group_by(["zone", "time_window"])
        .agg([
            pl.col("event_count").sum().alias("spoof_count"),
            pl.col("confidence").max().alias("max_conf"),
        ])
    )

    if spoof_density.is_empty():
        return pl.DataFrame(), 1.0, 0, 0

    # Authenticity = 1 - (spoof_density / baseline)
    max_count = spoof_density["spoof_count"].max()
    if max_count == 0:
        return pl.DataFrame(), 1.0, 0, 0

    zones = spoof_density.with_columns([
        (1.0 - (pl.col("spoof_count").cast(pl.Float64) / max(float(max_count), 1.0)) * pl.col("max_conf"))
        .clip(0.0, 1.0).alias("mean_auth"),
        pl.col("spoof_count").alias("snap_count"),
    ]).filter(pl.col("snap_count") >= 5)

    avg_auth = float(zones["mean_auth"].mean()) if not zones.is_empty() else 1.0
    low_count = int(zones.filter(pl.col("mean_auth") < 0.4).height)
    total = int(zones.height)

    return zones, avg_auth, low_count, total


# ── Report Builder ─────────────────────────────────────────────────

class FastMorningReportBuilder:
    def __init__(self, cfg: InstrumentConfig):
        self.cfg = cfg

    def run(
        self,
        file_path: str,
        rth_only: bool = False,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ) -> str:
        t_total = time.monotonic()

        # ── Load data via BatchScanner ──
        print(f"Loading {os.path.basename(file_path)} via to_ndarray()...")
        t0 = time.monotonic()
        scanner = BatchScanner()
        scanner.load(file_path, symbol_filter=self.cfg.symbol)
        df = scanner._events_df
        print(f"  Loaded {len(df):,} events in {time.monotonic() - t0:.2f}s")

        # ── Time filtering ──
        if time_start:
            start_ns = scanner._parse_time_ns(time_start)
            df = df.filter(pl.col("ts_ns") >= start_ns)
        if time_end:
            end_ns = scanner._parse_time_ns(time_end, end_of_day=True)
            df = df.filter(pl.col("ts_ns") <= end_ns)

        # ── RTH filter ──
        rth_target_date = None
        if rth_only:
            m = re.search(r'(\d{8})T\d{4}Z', os.path.basename(file_path))
            if m:
                from datetime import date as _date_cls
                rth_target_date = _date_cls(
                    int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8])
                )
            else:
                m = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(file_path))
                if m:
                    from datetime import date as _date_cls
                    rth_target_date = _date_cls.fromisoformat(m.group(1))

            # Convert ts_ns to ET datetime components for filtering
            df = df.with_columns([
                # Convert ns to seconds for datetime
                (pl.col("ts_ns") // 1_000_000_000).cast(pl.Int64).alias("_epoch_s"),
            ])

            # Use Python-based filtering for timezone conversion
            epoch_s = df["_epoch_s"].to_numpy()
            et_hours = np.zeros(len(epoch_s), dtype=np.int32)
            et_minutes = np.zeros(len(epoch_s), dtype=np.int32)
            et_weekdays = np.zeros(len(epoch_s), dtype=np.int32)
            et_dates = np.zeros(len(epoch_s), dtype="datetime64[D]")

            # Vectorized: approximate ET offset (-4 or -5 from UTC)
            # For precision, check DST. But for speed, use -4 during DST, -5 otherwise.
            # Most trading sessions will be within one DST regime.
            sample_dt = datetime.fromtimestamp(float(epoch_s[0]), tz=timezone.utc).astimezone(_ET)
            et_offset_s = int(sample_dt.utcoffset().total_seconds())

            et_epoch = epoch_s + et_offset_s
            et_hours = ((et_epoch % 86400) // 3600).astype(np.int32)
            et_minutes = ((et_epoch % 3600) // 60).astype(np.int32)
            et_time_mins = et_hours * 60 + et_minutes

            # Weekday: 0=Monday ... 6=Sunday
            # days since epoch (Jan 1 1970 was Thursday = 3)
            et_days = et_epoch // 86400
            et_weekdays = ((et_days + 3) % 7).astype(np.int32)  # 0=Mon

            rth_start = 9 * 60 + 30   # 9:30 AM
            rth_end = 16 * 60          # 4:00 PM

            mask = (
                (et_weekdays < 5) &           # weekdays only
                (et_time_mins >= rth_start) &  # after 9:30 AM
                (et_time_mins < rth_end)       # before 4:00 PM
            )

            if rth_target_date is not None:
                # Also filter to specific date
                target_epoch_start = int(datetime(
                    rth_target_date.year, rth_target_date.month, rth_target_date.day,
                    tzinfo=_ET
                ).timestamp())
                target_epoch_end = target_epoch_start + 86400
                # et_epoch is already in ET seconds
                date_mask = (
                    (epoch_s >= (target_epoch_start - et_offset_s)) &
                    (epoch_s < (target_epoch_end - et_offset_s))
                )
                mask = mask & date_mask

            mask_series = pl.Series("_rth_mask", mask)
            pre_count = len(df)
            df = df.filter(mask_series)
            df = df.drop("_epoch_s")
            print(f"  RTH filter: {len(df):,} kept, {pre_count - len(df):,} filtered out")

        if df.is_empty():
            return "⚠️  No events matched the filters."

        print(f"  Working with {len(df):,} events")

        # ── Extract session metrics ──
        ts_ms_min = int(df["ts_ms"].min())
        ts_ms_max = int(df["ts_ms"].max())

        fills_df = df.filter(pl.col("action").is_in(["F", "T"]))
        if not fills_df.is_empty():
            prices = fills_df["price_f"].to_list()
            open_p = prices[0]
            close_p = prices[-1]
            high_p = fills_df["price_f"].max()
            low_p = fills_df["price_f"].min()
        else:
            open_p = close_p = high_p = low_p = 0.0

        # Approximate "snapshot count" for compatibility with persistence thresholds
        session_duration_s = (ts_ms_max - ts_ms_min) / 1000.0
        snapshot_count = int(session_duration_s / 0.5)  # 500ms intervals

        # ── Run detectors ──
        print("  Running spoof detection...")
        t0 = time.monotonic()

        pull_results = detect_pull_before_touch(df, self.cfg)
        print(f"    Pull-Before-Touch: {len(pull_results)} zones ({time.monotonic() - t0:.2f}s)")

        t0 = time.monotonic()
        layer_results = detect_layering(df, self.cfg)
        print(f"    Layering: {len(layer_results)} zones ({time.monotonic() - t0:.2f}s)")

        t0 = time.monotonic()
        phantom_results = detect_phantom_walls(df, self.cfg)
        print(f"    Phantom Walls: {len(phantom_results)} zones ({time.monotonic() - t0:.2f}s)")

        t0 = time.monotonic()
        iceberg_results = detect_icebergs(df, self.cfg)
        print(f"    Icebergs: {len(iceberg_results)} zones ({time.monotonic() - t0:.2f}s)")

        # Combine spoof results
        spoof_dfs = []
        for sdf in [pull_results, layer_results, phantom_results]:
            if not sdf.is_empty():
                spoof_dfs.append(sdf)

        if spoof_dfs:
            all_spoofs = pl.concat(spoof_dfs)
        else:
            all_spoofs = pl.DataFrame(schema={
                "price_f": pl.Float64, "side": pl.Utf8, "spoof_type": pl.Utf8,
                "confidence": pl.Float64, "first_ts_ms": pl.Int64,
                "last_ts_ms": pl.Int64, "event_count": pl.UInt32,
            })

        print("  Running fragility analysis...")
        t0 = time.monotonic()
        fragility_results = compute_fragility(df, self.cfg)
        print(f"    Fragility: {len(fragility_results)} levels ({time.monotonic() - t0:.2f}s)")

        print("  Computing authenticity zones...")
        t0 = time.monotonic()
        auth_zones, avg_auth, low_auth_count, auth_total = compute_authenticity_zones(
            df, all_spoofs, self.cfg
        )
        print(f"    Authenticity: avg {avg_auth:.0%} ({time.monotonic() - t0:.2f}s)")

        # ── Generate Report ──
        report = self._format_report(
            file_path=file_path,
            rth_target_date=rth_target_date,
            ts_ms_min=ts_ms_min,
            ts_ms_max=ts_ms_max,
            open_p=open_p, close_p=close_p, high_p=high_p, low_p=low_p,
            snapshot_count=snapshot_count,
            all_spoofs=all_spoofs,
            iceberg_results=iceberg_results,
            fragility_results=fragility_results,
            auth_zones=auth_zones,
            avg_auth=avg_auth,
            low_auth_count=low_auth_count,
            auth_total=auth_total,
        )

        elapsed = time.monotonic() - t_total
        print(f"\n  Total time: {elapsed:.1f}s")

        return report

    def _format_report(
        self,
        file_path: str,
        rth_target_date,
        ts_ms_min: int, ts_ms_max: int,
        open_p: float, close_p: float, high_p: float, low_p: float,
        snapshot_count: int,
        all_spoofs: pl.DataFrame,
        iceberg_results: pl.DataFrame,
        fragility_results: pl.DataFrame,
        auth_zones: pl.DataFrame,
        avg_auth: float,
        low_auth_count: int,
        auth_total: int,
    ) -> str:
        cfg = self.cfg
        lines = []

        # ── Header ──
        sym = cfg.symbol
        if rth_target_date:
            date_str = rth_target_date.strftime("%A %B %d %Y")
        else:
            date_str = _ms_to_et_date(ts_ms_min) if ts_ms_min else "Unknown Date"
        start_t = _ms_to_et(ts_ms_min)
        end_t = _ms_to_et(ts_ms_max)
        change = close_p - open_p

        lines.append("═" * 63)
        lines.append(f"  {sym} MORNING PREP — {date_str}")
        lines.append(f"  L3 Forensics: Spoof Detection + Fragility Analysis")
        lines.append("═" * 63)
        if high_p > 0:
            lines.append(f"  Session:   {low_p:.2f} → {high_p:.2f}  (Δ {change:+.2f})")
            lines.append(f"  Open: {open_p:.2f}  Close: {close_p:.2f}")
        lines.append(f"  Time:      {start_t} → {end_t} ET")
        lines.append(f"  Snapshots: {snapshot_count:,} @ 500ms intervals")
        lines.append("")

        # ── Spoof Activity ──
        _type_emoji = {
            "pull_before_touch": "⚡",
            "layering": "📚",
            "phantom_wall": "👻",
            "stuffing": "📦",
        }

        def _spoof_section(side_code: str, label: str) -> None:
            lines.append(f"🎭 SPOOF ACTIVITY — {label}")
            side_spoofs = all_spoofs.filter(pl.col("side") == side_code)
            if side_spoofs.is_empty():
                lines.append("  None detected.")
            else:
                side_spoofs = side_spoofs.sort("event_count", descending=True)
                for row in side_spoofs.head(15).iter_rows(named=True):
                    emoji = _type_emoji.get(row["spoof_type"], "❓")
                    conf_pct = int(row["confidence"] * 100)
                    t0 = _ms_to_et(row["first_ts_ms"])
                    t1 = _ms_to_et(row["last_ts_ms"])
                    warn = "  ⚠️ HIGH" if conf_pct > 60 else ""
                    lines.append(
                        f"  {emoji} {row['price_f']:.2f}  {row['spoof_type']} × {row['event_count']}"
                        f"  conf {conf_pct}%  {t0}–{t1}{warn}"
                    )
            lines.append("")

        _spoof_section("B", "BID SIDE (fake support)")
        _spoof_section("A", "ASK SIDE (fake resistance)")

        # ── Iceberg Accumulation ──
        lines.append("─" * 63)
        lines.append("🧊 ICEBERG ACCUMULATION ZONES")
        lines.append("─" * 63)

        if iceberg_results.is_empty():
            lines.append("  No icebergs detected.")
        else:
            icebergs_sorted = iceberg_results.sort("total_volume", descending=True)
            for row in icebergs_sorted.head(15).iter_rows(named=True):
                side_label = "BID (buying)" if row["side"] == "B" else "ASK (selling)"
                dur_s = (row["last_ts_ms"] - row["first_ts_ms"]) / 1000
                lines.append(
                    f"  ★ {row['price_f']:.2f} — {side_label}  "
                    f"{row['total_refills']} refills  ~{row['total_volume']:.0f} contracts  "
                    f"conf {row['max_confidence']:.0%}  active {dur_s:.0f}s  "
                    f"({row['detections']} snaps)"
                )
        lines.append("")

        # ── Fragility Ladder ──
        threshold = max(int(snapshot_count * cfg.persistence_pct), cfg.persistence_min)

        persistent = fragility_results.filter(
            pl.col("total_snapshots") >= threshold
        )

        def _dominant_state(row):
            m = max(row["fragile_count"], row["solid_count"], row["moderate_count"])
            if m == row["fragile_count"]:
                return "FRAGILE"
            elif m == row["solid_count"]:
                return "SOLID"
            return "MODERATE"

        _state_emoji = {"FRAGILE": "🔴", "SOLID": "🟢", "MODERATE": "🟡"}

        lines.append("─" * 63)
        lines.append("🏗️  FRAGILITY LADDER (persistent levels, nearest to close)")
        lines.append("─" * 63)
        lines.append("")

        # Separate bid/ask
        bid_frag = persistent.filter(pl.col("side") == "B").sort("price_f", descending=True)
        ask_frag = persistent.filter(pl.col("side") == "A").sort("price_f")

        # Mark spoof overlap
        spoof_prices = set()
        if not all_spoofs.is_empty():
            spoof_prices = set(all_spoofs["price_f"].to_list())

        # ASK side (above close)
        ask_near = ask_frag.filter(pl.col("price_f") >= close_p).head(10)
        if not ask_near.is_empty():
            lines.append("  ASK ────────────────────────────────────────────")
            for row in ask_near.iter_rows(named=True):
                state = _dominant_state(row)
                e = _state_emoji.get(state, "⚪")
                frag_pct = row["fragile_count"] / max(row["total_snapshots"], 1) * 100
                spoof_tag = "  ← SPOOFED" if row["price_f"] in spoof_prices else ""
                lines.append(
                    f"  {e} {row['price_f']:.2f}  {state}  "
                    f"{frag_pct:.0f}% fragile  "
                    f"avg {row['avg_volume']:.0f} vol / {row['avg_order_count']:.0f} orders"
                    f"{spoof_tag}"
                )

        if close_p > 0:
            lines.append(f"  ─── {close_p:.2f} ── CLOSE ──────────────────────────")

        # BID side (below close)
        bid_near = bid_frag.filter(pl.col("price_f") <= close_p).head(10)
        if not bid_near.is_empty():
            for row in bid_near.iter_rows(named=True):
                state = _dominant_state(row)
                e = _state_emoji.get(state, "⚪")
                frag_pct = row["fragile_count"] / max(row["total_snapshots"], 1) * 100
                spoof_tag = "  ← SPOOFED" if row["price_f"] in spoof_prices else ""
                lines.append(
                    f"  {e} {row['price_f']:.2f}  {state}  "
                    f"{frag_pct:.0f}% fragile  "
                    f"avg {row['avg_volume']:.0f} vol / {row['avg_order_count']:.0f} orders"
                    f"{spoof_tag}"
                )
            lines.append("  BID ────────────────────────────────────────────")
        lines.append("")

        # ── Actionable Levels ──
        lines.append("─" * 63)
        lines.append("🎯 ACTIONABLE LEVELS")
        lines.append("─" * 63)
        lines.append("")

        high_conf: List[str] = []
        watch: List[str] = []

        # Icebergs
        if not iceberg_results.is_empty():
            bid_ice = iceberg_results.filter(
                (pl.col("side") == "B") & (pl.col("total_volume") >= cfg.iceberg_vol_threshold)
            ).sort("total_volume", descending=True).head(5)
            ask_ice = iceberg_results.filter(
                (pl.col("side") == "A") & (pl.col("total_volume") >= cfg.iceberg_vol_threshold)
            ).sort("total_volume", descending=True).head(5)

            for row in bid_ice.iter_rows(named=True):
                entry = f"  📈 {row['price_f']:.2f}  Institutional BUYING  (~{row['total_volume']:.0f} contracts, {row['total_refills']} refills)"
                (high_conf if row["max_confidence"] > 0.6 else watch).append(entry)
            for row in ask_ice.iter_rows(named=True):
                entry = f"  📉 {row['price_f']:.2f}  Institutional SELLING  (~{row['total_volume']:.0f} contracts, {row['total_refills']} refills)"
                (high_conf if row["max_confidence"] > 0.6 else watch).append(entry)

        # Cross-layer: spoofed + fragile
        if not persistent.is_empty():
            for row in persistent.iter_rows(named=True):
                state = _dominant_state(row)
                is_spoofed = row["price_f"] in spoof_prices
                frag_pct = row["fragile_count"] / max(row["total_snapshots"], 1) * 100

                if is_spoofed and state == "FRAGILE":
                    tag = "support" if row["side"] == "B" else "resistance"
                    entry = f"  ⚠️  {row['price_f']:.2f}  FRAGILE {tag} + SPOOFED  (fragile {frag_pct:.0f}% of session)"
                    high_conf.append(entry)
                elif state == "FRAGILE" and abs(row["price_f"] - close_p) <= 20:
                    tag = "support" if row["side"] == "B" else "resistance"
                    entry = f"  ⚠️  {row['price_f']:.2f}  Fragile {tag}  (fragile {frag_pct:.0f}% of session)"
                    watch.append(entry)

        # High-conf spoof zones
        if not all_spoofs.is_empty():
            hot_spoofs = all_spoofs.filter(
                (pl.col("confidence") > 0.6) & (pl.col("event_count") >= 5)
            )
            for row in hot_spoofs.iter_rows(named=True):
                side_label = "BID" if row["side"] == "B" else "ASK"
                entry = (
                    f"  🎭 {row['price_f']:.2f}  {row['spoof_type']} on {side_label}  "
                    f"(conf {int(row['confidence'] * 100)}%, × {row['event_count']})"
                )
                high_conf.append(entry)

        lines.append("  HIGH CONFIDENCE:")
        if high_conf:
            for e in high_conf[:10]:
                lines.append(e)
        else:
            lines.append("  None.")
        lines.append("")
        lines.append("  WATCH:")
        if watch:
            for e in watch[:10]:
                lines.append(e)
        else:
            lines.append("  None.")
        lines.append("")

        # ── Authenticity Zones ──
        lines.append("─" * 63)
        lines.append("🔍 AUTHENTICITY — LOW ZONES (< 60%)")
        lines.append("─" * 63)

        if auth_zones is not None and not auth_zones.is_empty():
            low_zones = auth_zones.filter(pl.col("mean_auth") < 0.60).sort("mean_auth")
            if low_zones.is_empty():
                lines.append("  All zones ≥ 60% authentic.")
            else:
                for row in low_zones.head(10).iter_rows(named=True):
                    win_start_ms = int(row["time_window"]) * 30 * 60 * 1000
                    t_str = _ms_to_et(win_start_ms)
                    lines.append(
                        f"  📍 {row['zone']:.2f}  auth {row['mean_auth']:.0%}  "
                        f"({row['snap_count']} snaps)  window ~{t_str} ET"
                    )
        else:
            lines.append("  All zones ≥ 60% authentic.")

        lines.append("")
        lines.append(f"  Session avg: {avg_auth:.0%}  |  Low (<40%) snaps: {low_auth_count} ({low_auth_count / max(auth_total, 1) * 100:.1f}%)")
        lines.append("")

        lines.append("═" * 63)
        lines.append("  Mark these levels on your chart before the session.")
        lines.append("═" * 63)

        return "\n".join(lines)


# ── Multi-file comparison ──────────────────────────────────────────

def comparative_summary(results: List[Tuple[str, str]]) -> str:
    """Generate comparative summary from multiple report runs.

    results: list of (file_path, report_text) tuples
    """
    if len(results) < 2:
        return ""
    # For now, just concatenate. A full implementation would parse and cross-reference.
    lines = []
    lines.append("")
    lines.append("═" * 63)
    lines.append("  COMPARATIVE SUMMARY")
    lines.append("═" * 63)
    lines.append(f"  {len(results)} sessions analyzed. Cross-reference actionable levels above.")
    lines.append("═" * 63)
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Morning Prep Report — FAST (Polars vectorized)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("file", nargs="?", help="Path to .dbn.zst file")
    group.add_argument("--files", nargs="+", help="Multiple .dbn.zst files for comparative analysis")
    parser.add_argument("--symbol", help="Instrument (ES, NQ, CL, GC, SI). Required for full-exchange files.")
    parser.add_argument("--rth-only", action="store_true", help="Only analyze RTH (9:30-4:00 ET)")
    parser.add_argument("--date", help="Extract single date from multi-day file (YYYY-MM-DD)")
    parser.add_argument("--dates", nargs="+", help="Extract multiple dates for comparison (YYYY-MM-DD ...)")
    parser.add_argument("--start", dest="time_start", help="Start time filter (YYYY-MM-DDTHH:MM or YYYY-MM-DD)")
    parser.add_argument("--end", dest="time_end", help="End time filter (YYYY-MM-DDTHH:MM or YYYY-MM-DD)")

    args = parser.parse_args()
    sym = args.symbol

    files = args.files or [args.file]
    results = []

    for fp in files:
        cfg = _detect_instrument(fp, symbol_override=sym)
        builder = FastMorningReportBuilder(cfg)

        time_start = args.time_start or (args.date if args.date else None)
        time_end = args.time_end or (args.date if args.date else None)

        report = builder.run(
            fp, rth_only=args.rth_only,
            time_start=time_start, time_end=time_end,
        )
        print("\n")
        print(report)
        results.append((fp, report))

        # Save report
        report_path = fp.replace(".dbn.zst", "-fast-report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

    if len(results) >= 2:
        comp = comparative_summary(results)
        print(comp)

    # Handle --dates
    if args.dates and args.file:
        for d in args.dates:
            cfg = _detect_instrument(args.file, symbol_override=sym)
            builder = FastMorningReportBuilder(cfg)
            report = builder.run(args.file, rth_only=args.rth_only, time_start=d, time_end=d)
            print("\n")
            print(report)


if __name__ == "__main__":
    main()
