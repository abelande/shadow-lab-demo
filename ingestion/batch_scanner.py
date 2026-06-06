"""Bulk scan of .dbn.zst files using Polars for fast OHLCV + stats extraction.

This is a separate path from the streaming pipeline. Use it for:
  - Pre-building candle arrays for ReplayEngine
  - Session statistics
  - Trade-level data for tick charts

The file is loaded once via load(), then candles/trades/stats are computed
from the cached DataFrame without re-reading the file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)


def _resolve_instrument_id(store, symbol: Optional[str]) -> Optional[int]:
    """Resolve a symbol name to an instrument_id from file metadata."""
    if symbol is None:
        return None

    mappings = getattr(store.metadata, "mappings", {})
    if not mappings:
        return None

    sym = symbol.upper().split(".")[0]

    if sym in mappings:
        return int(mappings[sym][0]["symbol"])

    for key in mappings:
        if key.startswith(sym + ".") or key.startswith(sym.lower() + "."):
            return int(mappings[key][0]["symbol"])

    for key, map_list in mappings.items():
        if sym in key.upper() and map_list:
            return int(map_list[0]["symbol"])

    return None


class BatchScanner:
    """Load a .dbn.zst file once, then query candles/trades/stats from memory.

    Usage::

        scanner = BatchScanner()
        scanner.load("data/nq-mbo-2026-03-27.dbn.zst", symbol_filter="NQ")

        candles = scanner.build_ohlcv(timeframe_seconds=15)
        trades  = scanner.get_trades()
        stats   = scanner.session_stats()
        ticks   = scanner.trades_for_candle(candle_time, 15)
    """

    def __init__(self) -> None:
        self._all_df: Optional[pl.DataFrame] = None
        self._fills_df: Optional[pl.DataFrame] = None
        self._trades_df: Optional[pl.DataFrame] = None
        self._loaded = False
        self._path: Optional[Path] = None
        self._symbol: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # Parquet cache directory (sibling to data files)
    _CACHE_DIR_NAME = ".cache"

    def _cache_path(self, path: Path, symbol_filter: Optional[str]) -> Path:
        """Return the parquet cache path for a given source file + symbol."""
        cache_dir = path.parent / self._CACHE_DIR_NAME
        suffix = f"_{symbol_filter}" if symbol_filter else ""
        return cache_dir / f"{path.stem}{suffix}.parquet"

    def load(self, path: str | Path, symbol_filter: Optional[str] = None) -> None:
        """Load a .dbn.zst file via to_ndarray() with parquet caching.

        First load: to_ndarray() (Rust → NumPy, ~0.3s) → Polars → write parquet cache.
        Subsequent loads: pl.read_parquet() (~0.14s).
        """
        import databento as db
        import time

        path = Path(path)
        cache = self._cache_path(path, symbol_filter)
        logger.info("BatchScanner: loading %s (symbol=%s)...", path.name, symbol_filter)

        t0 = time.monotonic()

        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            # Cache hit — read parquet directly (Rust-native, ~0.14s)
            df = pl.read_parquet(cache)
            logger.info("BatchScanner: cache hit %s (%.2fs, %d rows)", cache.name, time.monotonic() - t0, len(df))
        else:
            # Cache miss — load via to_ndarray() (Rust → NumPy, ~0.3s)
            store = db.DBNStore.from_file(str(path))

            # Resolve instrument ID before loading array
            inst_id = _resolve_instrument_id(store, symbol_filter)

            arr = store.to_ndarray()
            logger.info("BatchScanner: to_ndarray() %.2fs (%d records)", time.monotonic() - t0, len(arr))

            # Convert structured NumPy array → Polars with proper types.
            # Raw ndarray has: ts_event as uint64 nanoseconds, price as int64
            # fixed-point (÷1e9), action/side as bytes (b'A', b'B', etc.)
            df = pl.DataFrame({
                "instrument_id": pl.Series(arr["instrument_id"], dtype=pl.UInt32),
                "ts_event": pl.Series(arr["ts_event"], dtype=pl.UInt64),
                "order_id": pl.Series(arr["order_id"], dtype=pl.UInt64),
                "price": pl.Series(arr["price"], dtype=pl.Int64),
                "size": pl.Series(arr["size"], dtype=pl.UInt32),
                "action": pl.Series(arr["action"].astype(str), dtype=pl.Utf8),
                "side": pl.Series(arr["side"].astype(str), dtype=pl.Utf8),
                "flags": pl.Series(arr["flags"], dtype=pl.UInt8),
                "ts_recv": pl.Series(arr["ts_recv"], dtype=pl.UInt64),
            })

            # Filter by instrument
            if inst_id is not None:
                df = df.filter(pl.col("instrument_id") == inst_id)

            # Write parquet cache for next time
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(cache)
            logger.info("BatchScanner: cached to %s (%.2fs total)", cache.name, time.monotonic() - t0)

        self._all_df = df
        self._path = path
        self._symbol = symbol_filter

        # Pre-compute query-ready DataFrame with human-friendly columns.
        # Raw columns: ts_event (uint64 ns), price (int64 fixed-point ÷1e9).
        self._events_df = df.with_columns([
            (pl.col("price").cast(pl.Float64) / 1_000_000_000).alias("price_f"),
            (pl.col("ts_event") // 1_000_000_000).cast(pl.Int64).alias("ts_sec"),
            pl.col("ts_event").cast(pl.Int64).alias("ts_ns"),
            (pl.col("ts_event") // 1_000_000).cast(pl.Int64).alias("ts_ms"),
        ]).sort("ts_ns")

        # Pre-compute fills DataFrame (used by candles, trades, stats)
        fills = df.filter(pl.col("action").is_in(["F", "T"]))
        fills = fills.filter(pl.col("price") > 0)
        fills = fills.with_columns([
            (pl.col("price").cast(pl.Float64) / 1_000_000_000).alias("price_f"),
            (pl.col("ts_event") // 1_000_000_000).cast(pl.Int64).alias("ts_sec"),
            pl.col("ts_event").cast(pl.Int64).alias("ts_ns"),
        ])
        self._fills_df = fills

        # Pre-compute trades DataFrame for tick chart
        self._trades_df = fills.select([
            "ts_ns", "ts_sec", "price_f", "size", "side",
        ]).rename({"price_f": "price"}).sort("ts_ns")

        self._loaded = True
        logger.info(
            "BatchScanner: ready — %d records, %d fills from %s (%.2fs)",
            len(df), len(fills), path.name, time.monotonic() - t0,
        )

    def build_ohlcv(self, timeframe_seconds: int = 15) -> List[dict]:
        """Build OHLCV candles from pre-loaded fill data.

        Returns list of { time, open, high, low, close, volume } dicts
        sorted by time ascending. Time is Unix seconds.
        """
        if not self._loaded or self._fills_df is None:
            raise RuntimeError("Call load() before build_ohlcv()")

        df = self._fills_df
        if df.is_empty():
            return []

        tf = timeframe_seconds
        df = df.with_columns([
            ((pl.col("ts_sec") // tf) * tf).alias("bucket"),
        ])

        candles = (
            df.group_by("bucket")
            .agg([
                pl.col("price_f").first().alias("open"),
                pl.col("price_f").max().alias("high"),
                pl.col("price_f").min().alias("low"),
                pl.col("price_f").last().alias("close"),
                pl.col("size").sum().cast(pl.Float64).alias("volume"),
            ])
            .sort("bucket")
        )

        result = []
        for row in candles.iter_rows(named=True):
            result.append({
                "time": int(row["bucket"]),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            })

        logger.info("BatchScanner: %d candles at %ds", len(result), timeframe_seconds)
        return result

    def get_trades(self) -> pl.DataFrame:
        """Return pre-computed trades DataFrame.

        Columns: [ts_ns, ts_sec, price, size, side], sorted by ts_ns.
        """
        if not self._loaded or self._trades_df is None:
            raise RuntimeError("Call load() before get_trades()")
        return self._trades_df

    def trades_for_candle(
        self,
        candle_time_sec: int,
        timeframe_seconds: int,
    ) -> List[dict]:
        """Extract trades within a candle's time window for tick-chart rendering.

        Returns list of { ts, price } dicts.
        """
        if not self._loaded or self._trades_df is None:
            raise RuntimeError("Call load() before trades_for_candle()")

        start = candle_time_sec
        end = candle_time_sec + timeframe_seconds

        window = self._trades_df.filter(
            (pl.col("ts_sec") >= start) & (pl.col("ts_sec") < end)
        )

        result = []
        for row in window.iter_rows(named=True):
            result.append({
                "ts": row["ts_ns"] / 1_000_000_000,
                "price": row["price"],
            })
        return result

    def session_stats(self) -> Dict:
        """Return session-level stats from pre-loaded data."""
        if not self._loaded or self._all_df is None:
            raise RuntimeError("Call load() before session_stats()")

        df = self._all_df
        fills = self._fills_df

        action_counts = df.group_by("action").len()
        counts = {row["action"]: row["len"] for row in action_counts.iter_rows(named=True)}

        df_ts = df.with_columns([
            (pl.col("ts_event").dt.epoch("ns")).alias("ts_ns"),
        ])
        ts_min = df_ts["ts_ns"].min()
        ts_max = df_ts["ts_ns"].max()

        stats = {
            "total_records": len(df),
            "fill_count": len(fills),
            "add_count": counts.get("A", 0),
            "cancel_count": counts.get("C", 0),
            "modify_count": counts.get("M", 0),
            "trade_count": counts.get("F", 0) + counts.get("T", 0),
            "time_start_ns": int(ts_min) if ts_min is not None else None,
            "time_end_ns": int(ts_max) if ts_max is not None else None,
        }

        if fills is not None and not fills.is_empty():
            stats["price_open"] = float(fills["price_f"].head(1)[0])
            stats["price_close"] = float(fills["price_f"].tail(1)[0])
            stats["price_high"] = float(fills["price_f"].max())
            stats["price_low"] = float(fills["price_f"].min())
            stats["total_volume"] = float(fills["size"].sum())

        return stats

    # ═══════════════════════════════════════════════════════════════
    # Event Query / Search / Extract
    # ═══════════════════════════════════════════════════════════════

    _ACTION_LABELS = {"A": "ADD", "C": "CANCEL", "M": "MODIFY", "F": "FILL", "T": "FILL"}
    _SIDE_LABELS = {"B": "BID", "A": "ASK"}

    def query_events(
        self,
        *,
        actions: Optional[List[str]] = None,
        side: Optional[str] = None,
        order_id: Optional[str] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        size_min: Optional[float] = None,
        size_max: Optional[float] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> Dict:
        """Query MBO events with flexible filters.

        Args:
            actions: Filter by action types. Accepts raw codes ("A","C","M","F","T")
                     or labels ("ADD","CANCEL","MODIFY","FILL").
            side: Filter by side — "BID"/"B" or "ASK"/"A".
            order_id: Filter by exact order_id.
            price_min/price_max: Price range filter (human-readable, e.g. 19998.00).
            size_min/size_max: Order size range filter.
            time_start/time_end: ISO datetime strings (UTC) for time range.
            limit: Max rows returned (default 1000, max 50000).
            offset: Skip first N matching rows.

        Returns:
            Dict with "total" (matching count), "returned" (rows in this page),
            "offset", "events" (list of event dicts).
        """
        if not self._loaded or self._events_df is None:
            raise RuntimeError("Call load() before query_events()")

        df = self._events_df
        limit = min(limit, 50000)

        # ── Action filter ──
        if actions:
            # Normalize labels to raw codes
            label_to_code = {v: k for k, v in self._ACTION_LABELS.items()}
            codes = set()
            for a in actions:
                upper = a.upper()
                if upper in label_to_code:
                    codes.add(label_to_code[upper])
                elif upper in self._ACTION_LABELS:
                    codes.add(upper)
            if codes:
                df = df.filter(pl.col("action").is_in(list(codes)))

        # ── Side filter ──
        if side:
            side_upper = side.upper()
            if side_upper in ("BID", "B"):
                df = df.filter(pl.col("side") == "B")
            elif side_upper in ("ASK", "A"):
                df = df.filter(pl.col("side") == "A")

        # ── Order ID filter ──
        if order_id is not None:
            oid = int(order_id) if order_id.isdigit() else order_id
            df = df.filter(pl.col("order_id") == oid)

        # ── Price range ──
        if price_min is not None:
            df = df.filter(pl.col("price_f") >= price_min)
        if price_max is not None:
            df = df.filter(pl.col("price_f") <= price_max)

        # ── Size range ──
        if size_min is not None:
            df = df.filter(pl.col("size") >= size_min)
        if size_max is not None:
            df = df.filter(pl.col("size") <= size_max)

        # ── Time range ──
        if time_start:
            start_ns = self._parse_time_ns(time_start)
            df = df.filter(pl.col("ts_ns") >= start_ns)
        if time_end:
            end_ns = self._parse_time_ns(time_end, end_of_day=True)
            df = df.filter(pl.col("ts_ns") <= end_ns)

        total = len(df)
        page = df.slice(offset, limit)

        events = []
        for row in page.iter_rows(named=True):
            action_raw = row.get("action", "")
            side_raw = row.get("side", "")
            events.append({
                "timestamp_ms": row.get("ts_ms"),
                "timestamp_ns": row.get("ts_ns"),
                "action": self._ACTION_LABELS.get(action_raw, action_raw),
                "action_raw": action_raw,
                "order_id": str(row.get("order_id", "")),
                "side": self._SIDE_LABELS.get(side_raw, side_raw),
                "side_raw": side_raw,
                "price": row.get("price_f"),
                "size": row.get("size"),
            })

        return {
            "total": total,
            "returned": len(events),
            "offset": offset,
            "limit": limit,
            "events": events,
        }

    def order_lifecycle(self, order_id: str) -> Dict:
        """Trace every event for a single order_id — its full lifecycle.

        Returns the sequence: ADD → MODIFY* → FILL/CANCEL, with timestamps
        and price/size at each step.
        """
        if not self._loaded or self._events_df is None:
            raise RuntimeError("Call load() before order_lifecycle()")

        oid = int(order_id) if order_id.isdigit() else order_id
        df = self._events_df.filter(pl.col("order_id") == oid).sort("ts_ns")

        if df.is_empty():
            return {"order_id": order_id, "events": [], "summary": "not found"}

        events = []
        for row in df.iter_rows(named=True):
            action_raw = row.get("action", "")
            side_raw = row.get("side", "")
            events.append({
                "timestamp_ms": row.get("ts_ms"),
                "action": self._ACTION_LABELS.get(action_raw, action_raw),
                "side": self._SIDE_LABELS.get(side_raw, side_raw),
                "price": row.get("price_f"),
                "size": row.get("size"),
            })

        first = events[0]
        last = events[-1]
        duration_ms = (last["timestamp_ms"] or 0) - (first["timestamp_ms"] or 0)

        return {
            "order_id": order_id,
            "event_count": len(events),
            "lifecycle": f"{first['action']} → {last['action']}",
            "duration_ms": duration_ms,
            "side": first["side"],
            "initial_price": first["price"],
            "initial_size": first["size"],
            "final_action": last["action"],
            "final_price": last["price"],
            "final_size": last["size"],
            "events": events,
        }

    def aggregate_by_action(
        self,
        *,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ) -> Dict:
        """Aggregate event counts and volume by action type within a time range.

        Useful for questions like "how many cancels vs fills in this window?"
        """
        if not self._loaded or self._events_df is None:
            raise RuntimeError("Call load() before aggregate_by_action()")

        df = self._events_df

        if time_start:
            start_ns = self._parse_time_ns(time_start)
            df = df.filter(pl.col("ts_ns") >= start_ns)
        if time_end:
            end_ns = self._parse_time_ns(time_end, end_of_day=True)
            df = df.filter(pl.col("ts_ns") <= end_ns)

        agg = (
            df.group_by("action")
            .agg([
                pl.len().alias("count"),
                pl.col("size").sum().alias("total_size"),
                pl.col("size").mean().alias("avg_size"),
                pl.col("price_f").mean().alias("avg_price"),
            ])
            .sort("count", descending=True)
        )

        result = {}
        for row in agg.iter_rows(named=True):
            label = self._ACTION_LABELS.get(row["action"], row["action"])
            result[label] = {
                "count": row["count"],
                "total_size": float(row["total_size"]) if row["total_size"] else 0,
                "avg_size": round(float(row["avg_size"]), 2) if row["avg_size"] else 0,
                "avg_price": round(float(row["avg_price"]), 4) if row["avg_price"] else None,
            }

        return {
            "total_events": len(df),
            "time_start": time_start,
            "time_end": time_end,
            "by_action": result,
        }

    def extract_range(
        self,
        time_start: str,
        time_end: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """Extract a time range of raw events for replay or export.

        If output_path is provided, writes a Parquet file (fast, compact,
        re-loadable by Polars/Pandas). Otherwise returns metadata + preview.
        """
        if not self._loaded or self._events_df is None:
            raise RuntimeError("Call load() before extract_range()")

        start_ns = self._parse_time_ns(time_start)
        end_ns = self._parse_time_ns(time_end, end_of_day=True)

        extracted = self._events_df.filter(
            (pl.col("ts_ns") >= start_ns) & (pl.col("ts_ns") <= end_ns)
        )

        info = {
            "time_start": time_start,
            "time_end": time_end,
            "total_events": len(extracted),
            "source_file": str(self._path) if self._path else None,
            "symbol": self._symbol,
        }

        if extracted.is_empty():
            info["warning"] = "No events found in the specified range."
            return info

        # Action breakdown
        action_counts = extracted.group_by("action").len()
        info["action_breakdown"] = {
            self._ACTION_LABELS.get(row["action"], row["action"]): row["len"]
            for row in action_counts.iter_rows(named=True)
        }

        if output_path:
            extracted.write_parquet(output_path)
            info["output_path"] = output_path
            info["output_format"] = "parquet"
            logger.info(
                "Extracted %d events (%s → %s) to %s",
                len(extracted), time_start, time_end, output_path,
            )

        return info

    @staticmethod
    def _parse_time_ns(time_str: str, end_of_day: bool = False) -> int:
        """Parse an ISO datetime string to nanoseconds since epoch (UTC)."""
        from datetime import datetime, timezone
        s = time_str
        if len(s) == 10:  # bare date YYYY-MM-DD
            s = f"{s}T23:59:59" if end_of_day else f"{s}T00:00:00"
        if "+" not in s and "Z" not in s:
            s = s + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
