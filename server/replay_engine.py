"""ReplayEngine: pre-processes MBO data into candle + level arrays for human-speed replay.

Loads a .dbn.zst file via DatabentoReplayFeed, processes all snapshots through
the pipeline and LevelTracker, and stores candle/level arrays for random access.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from ..models import (
    ForceVector,
    GameState,
    InstrumentVisualConfig,
    LevelState,
    Order,
    OrderAction,
    ReplayCandle,
    ReplayFrame,
    Side,
    SpoofEvent,
    TapeSummary,
)
from ..level_tracker import LevelTracker
from ..pipeline import OrderBookMetaPipeline

logger = logging.getLogger(__name__)

# Supported timeframes in seconds
SUPPORTED_TIMEFRAMES = [1, 5, 15, 60, 300]

# Default snapshot interval used during pre-processing
_SNAPSHOT_INTERVAL_MS = 100


def _empty_tape_summary() -> TapeSummary:
    return TapeSummary(
        buy_volume=0.0,
        sell_volume=0.0,
        delta=0.0,
        largest_fill_price=None,
        largest_fill_size=0.0,
        largest_fill_side=None,
        iceberg_hits=0,
        cancel_count_bid=0,
        cancel_count_ask=0,
    )


def _accumulate_tape(summary: TapeSummary, snapshot_trades: List[Order], snapshot_events: List[Order]) -> TapeSummary:
    """Accumulate trades and events into a TapeSummary."""
    buy_vol = summary.buy_volume
    sell_vol = summary.sell_volume
    largest_size = summary.largest_fill_size
    largest_price = summary.largest_fill_price
    largest_side = summary.largest_fill_side
    iceberg_hits = summary.iceberg_hits
    cancel_bid = summary.cancel_count_bid
    cancel_ask = summary.cancel_count_ask

    for trade in snapshot_trades:
        if trade.action != OrderAction.FILL:
            continue
        vol = trade.size
        if trade.side == Side.ASK:
            # ASK fill = buy aggressor
            buy_vol += vol
        else:
            sell_vol += vol
        if vol > largest_size:
            largest_size = vol
            largest_price = trade.price
            largest_side = "BID" if trade.side == Side.BID else "ASK"

    for event in snapshot_events:
        if event.action == OrderAction.CANCEL:
            if event.side == Side.BID:
                cancel_bid += 1
            else:
                cancel_ask += 1

    delta = buy_vol - sell_vol
    return TapeSummary(
        buy_volume=buy_vol,
        sell_volume=sell_vol,
        delta=delta,
        largest_fill_price=largest_price,
        largest_fill_size=largest_size,
        largest_fill_side=largest_side,
        iceberg_hits=iceberg_hits,
        cancel_count_bid=cancel_bid,
        cancel_count_ask=cancel_ask,
    )


def _build_candles(snapshot_records: List[dict], timeframe_s: int) -> List[ReplayCandle]:
    """Aggregate snapshot records into OHLCV candles at a given timeframe.

    Track 3E fix: The mid-price fallback was inside `if current is None` which
    meant it only fired once (before the first trade). Now it always fires for
    trade-less snapshots to carry price continuity across empty candle windows.
    """
    if not snapshot_records:
        return []

    candles: List[ReplayCandle] = []
    current: Optional[dict] = None  # { time, open, high, low, close, volume }

    for rec in snapshot_records:
        ts_ms = rec["timestamp_ms"]
        ts_s = ts_ms // 1000
        bucket_s = (ts_s // timeframe_s) * timeframe_s
        had_trades = False

        for trade in rec.get("trades", []):
            price = trade.price
            vol = trade.size
            if current is None or bucket_s > current["time"]:
                if current is not None:
                    candles.append(ReplayCandle(
                        time=current["time"],
                        open=current["open"],
                        high=current["high"],
                        low=current["low"],
                        close=current["close"],
                        volume=current["volume"],
                    ))
                current = {"time": bucket_s, "open": price, "high": price, "low": price, "close": price, "volume": 0.0}
            else:
                if price > current["high"]:
                    current["high"] = price
                if price < current["low"]:
                    current["low"] = price
                current["close"] = price
            current["volume"] += vol
            had_trades = True

        # Mid-price fallback for snapshots with no trades.
        # Ensures candles continue to form even during quiet periods.
        if not had_trades:
            best_bid = rec.get("best_bid")
            best_ask = rec.get("best_ask")
            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2.0
                if current is None or bucket_s > current["time"]:
                    if current is not None:
                        candles.append(ReplayCandle(
                            time=current["time"],
                            open=current["open"],
                            high=current["high"],
                            low=current["low"],
                            close=current["close"],
                            volume=current["volume"],
                        ))
                    current = {"time": bucket_s, "open": mid, "high": mid, "low": mid, "close": mid, "volume": 0.0}
                else:
                    # Update close within current bucket even for mid-price
                    if mid > current["high"]:
                        current["high"] = mid
                    if mid < current["low"]:
                        current["low"] = mid
                    current["close"] = mid

    if current is not None:
        candles.append(ReplayCandle(
            time=current["time"],
            open=current["open"],
            high=current["high"],
            low=current["low"],
            close=current["close"],
            volume=current["volume"],
        ))

    return candles


def _find_snapshot_idx_for_candle_time(
    snapshot_records: List[dict], candle_time_s: int, timeframe_s: int
) -> int:
    """Find the last snapshot index whose timestamp is within the candle's time window."""
    candle_end_ms = (candle_time_s + timeframe_s) * 1000
    idx = 0
    for i, rec in enumerate(snapshot_records):
        if rec["timestamp_ms"] < candle_end_ms:
            idx = i
    return idx


class ReplayEngine:
    """Pre-processes a .dbn.zst file into candle + level arrays for replay.

    Usage::

        engine = ReplayEngine()
        await engine.load("path/to/data.dbn.zst", symbol="NQ")

        # Get candles at 15s timeframe
        candles = engine.get_candles(15)

        # Get replay frame at candle index 42
        frame = engine.get_frame_at(42, timeframe_s=15)

        # Step through replay
        engine.seek(0)
        engine.play(speed=2.0)
        frame = engine.step_forward()
    """

    def __init__(self) -> None:
        self._snapshot_records: List[dict] = []
        self._candle_cache: Dict[int, List[ReplayCandle]] = {}
        self._frame_cache: Dict[Tuple[int, int], ReplayFrame] = {}  # (candle_idx, tf_s) -> frame

        self._current_tf_s: int = 15
        self._current_candle_idx: int = 0
        self._is_playing: bool = False
        self._speed: float = 1.0
        self._loaded: bool = False
        self._symbol: str = ""
        self._progress: float = 0.0  # 0.0-1.0 loading progress (aggregate)

        # Three-phase load state (Phase B refactor).
        # Phase 1 — chart: BatchScanner → OHLCV candles for all timeframes.
        # Phase 2 — ticks: per-candle tick index for fast playback lookups.
        # Phase 3 — levels: streaming pipeline + LevelTracker for per-snapshot level state.
        self._total_records: int = 0  # Set by BatchScanner Phase 1 for progress %
        self._records_scanned: int = 0  # Running count from streaming feed (Phase 3)
        self._chart_ready: bool = False
        self._ticks_ready: bool = False
        self._levels_ready: bool = False
        self._chart_progress: float = 0.0
        self._tick_progress: float = 0.0
        self._levels_progress: float = 0.0
        # Tick index: {timeframe_s: {bucket_time_sec: [tick_dict, ...]}}
        self._tick_index: Dict[int, Dict[int, List[dict]]] = {}
        # Hold BatchScanner reference between phases so Phase 2 can read the in-memory trades DF.
        self._scanner = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def symbol(self) -> str:
        return self._symbol

    async def load(
        self,
        file_path: str,
        symbol: str = "ES",
        snapshot_interval_ms: int = _SNAPSHOT_INTERVAL_MS,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        rth_only: bool = False,
    ) -> None:
        """Two-phase load of a .dbn.zst file (Track 3B).

        Phase 1: Fast candle build via BatchScanner (Polars).
          - Builds OHLCV candles for all timeframes
          - Sets chart_ready=True so the frontend can start scrubbing
          - Records total row count for Phase 2 progress tracking

        Phase 2: Level state streaming via DatabentoReplayFeed.
          - Streams through the full pipeline + LevelTracker
          - Only computes per-snapshot level state (candles already built)
          - Progress = feed.records_scanned / total_records from Phase 1
          - Sets state_ready=True when complete
        """
        from ..ingestion.databento_feed import DatabentoReplayFeed
        from ..ingestion.batch_scanner import BatchScanner

        logger.info("ReplayEngine: loading %s for %s (three-phase)", file_path, symbol)
        self._loaded = False
        self._chart_ready = False
        self._ticks_ready = False
        self._levels_ready = False
        self._chart_progress = 0.0
        self._tick_progress = 0.0
        self._levels_progress = 0.0
        self._symbol = symbol
        self._snapshot_records = []
        self._candle_cache = {}
        self._frame_cache = {}
        self._tick_index = {}
        self._scanner = None
        self._progress = 0.0
        self._total_records = 0
        self._records_scanned = 0

        # ── Phase 1: Fast candle build via BatchScanner ──────────────
        try:
            scanner = BatchScanner()
            import asyncio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: scanner.load(file_path, symbol_filter=symbol),
            )

            # Apply time range filter if specified
            if (time_start or time_end) and scanner._all_df is not None:
                import polars as pl
                df = scanner._all_df
                if time_start:
                    start_ns = scanner._parse_time_ns(time_start, end_of_day=False)
                    if "ts_ns" not in df.columns:
                        df = df.with_columns((pl.col("ts_event").dt.epoch("ns")).alias("ts_ns"))
                    df = df.filter(pl.col("ts_ns") >= start_ns)
                if time_end:
                    end_ns = scanner._parse_time_ns(time_end, end_of_day=True)
                    if "ts_ns" not in df.columns:
                        df = df.with_columns((pl.col("ts_event").dt.epoch("ns")).alias("ts_ns"))
                    df = df.filter(pl.col("ts_ns") <= end_ns)
                scanner._all_df = df
                # Rebuild fills from filtered data
                fills = df.filter(pl.col("action").is_in(["F", "T"]))
                fills = fills.filter(pl.col("price") > 0)
                fills = fills.with_columns([
                    pl.col("price").cast(pl.Float64).alias("price_f"),
                    (pl.col("ts_event").dt.epoch("s")).alias("ts_sec"),
                    (pl.col("ts_event").dt.epoch("ns")).alias("ts_ns"),
                ])
                scanner._fills_df = fills

            # Record total records for Phase 2 progress
            self._total_records = len(scanner._all_df) if scanner._all_df is not None else 0

            # Build candles for all supported timeframes
            for tf in SUPPORTED_TIMEFRAMES:
                candles_raw = scanner.build_ohlcv(timeframe_seconds=tf)
                self._candle_cache[tf] = [
                    ReplayCandle(
                        time=c["time"], open=c["open"], high=c["high"],
                        low=c["low"], close=c["close"], volume=c["volume"],
                    )
                    for c in candles_raw
                ]
                logger.info("ReplayEngine Phase 1: %d candles at %ds", len(self._candle_cache[tf]), tf)

            self._chart_ready = True
            self._chart_progress = 1.0
            self._scanner = scanner
            self._progress = 0.33
            logger.info("ReplayEngine: Phase 1 complete — chart ready (%d total records)", self._total_records)

        except Exception as exc:
            logger.error("ReplayEngine Phase 1 (BatchScanner) failed: %s", exc)
            self._chart_ready = False

        # ── Phase 2: Tick indexing (run in executor to avoid blocking event loop) ──
        if self._scanner is not None:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._build_tick_index, self._scanner)
                self._progress = 0.50
                logger.info("ReplayEngine: Phase 2 complete — ticks indexed for %d timeframes", len(self._tick_index))
            except Exception as exc:
                logger.error("ReplayEngine Phase 2 (tick indexing) failed: %s", exc)
                self._ticks_ready = False

        # ── Phase 3: Level state streaming ───────────────────────────
        cfg = InstrumentVisualConfig.for_symbol(symbol)
        pipeline = OrderBookMetaPipeline()
        tracker = LevelTracker(cfg)

        # Compute time range for RTH if needed
        _time_start = time_start
        _time_end = time_end
        if rth_only and not time_start and not time_end:
            import re, os
            date_match = re.search(r'(\d{4})-?(\d{2})-?(\d{2})', os.path.basename(file_path))
            if date_match:
                y, m, d = date_match.groups()
                _time_start = f"{y}-{m}-{d}T13:30:00"
                _time_end = f"{y}-{m}-{d}T20:00:00"

        feed = DatabentoReplayFeed(
            file_path=file_path,
            symbol=symbol,
            filter_symbol=symbol,
            snapshot_interval_ms=snapshot_interval_ms,
            time_start=_time_start,
            time_end=_time_end,
        )
        await feed.connect()

        snapshot_count = 0

        while True:
            snapshot = await feed.next()
            if snapshot is None:
                break

            try:
                frame = pipeline.run(snapshot)
                spoof_events = frame.authenticity.spoof_events if frame.authenticity else []
                auth_score = frame.authenticity.authenticity_score if frame.authenticity else 1.0
                level_states = tracker.update(snapshot, spoof_events=spoof_events, authenticity_score=auth_score)
            except Exception as exc:
                logger.warning("Pipeline error at %d: %s", snapshot.timestamp_ms, exc)
                level_states = []
                frame = None
                spoof_events = []

            rec = {
                "timestamp_ms": snapshot.timestamp_ms,
                "trades": list(snapshot.recent_trades or []),
                "events": list(snapshot.recent_events or []),
                "best_bid": snapshot.best_bid,
                "best_ask": snapshot.best_ask,
                "level_states": level_states,
                "game_state": frame.game_state if frame else None,
                "force_vector": frame.force_vector if frame else None,
                "spoof_events": spoof_events,
            }
            self._snapshot_records.append(rec)

            snapshot_count += 1
            self._records_scanned = feed.records_scanned
            if self._total_records > 0:
                self._levels_progress = min(0.99, feed.records_scanned / self._total_records)

            # Yield to event loop periodically so status polls and WebSocket
            # broadcasts can proceed during the long Phase 3 streaming.
            if snapshot_count % 50 == 0:
                await asyncio.sleep(0)
                self._progress = 0.50 + 0.50 * self._levels_progress

        self._levels_progress = 1.0
        self._progress = 1.0
        self._levels_ready = True
        self._loaded = True
        logger.info("ReplayEngine: Phase 3 complete — %d snapshots for %s", len(self._snapshot_records), symbol)

        if not self._chart_ready:
            logger.info("ReplayEngine: Building candles from snapshot records (Phase 1/2 fallback)")
            for tf in SUPPORTED_TIMEFRAMES:
                self._candle_cache[tf] = _build_candles(self._snapshot_records, tf)
                logger.info("ReplayEngine: %d candles at %ds timeframe", len(self._candle_cache[tf]), tf)
            self._chart_ready = True

    def _build_tick_index(self, scanner) -> None:
        """Phase 2: Build per-candle tick index from the in-memory trades DataFrame.

        Groups trades by candle bucket for each supported timeframe, producing a
        lookup dict for fast per-candle tick retrieval during playback.
        """
        import polars as pl

        self._tick_progress = 0.0
        self._tick_index = {}

        trades_df = scanner._trades_df
        if trades_df is None or trades_df.is_empty():
            self._tick_index = {tf: {} for tf in SUPPORTED_TIMEFRAMES}
            self._ticks_ready = True
            self._tick_progress = 1.0
            return

        total_steps = len(SUPPORTED_TIMEFRAMES)
        for step, tf in enumerate(SUPPORTED_TIMEFRAMES):
            bucketed = trades_df.with_columns(
                ((pl.col("ts_sec") // tf) * tf).cast(pl.Int64).alias("bucket"),
            )
            grouped = bucketed.group_by("bucket", maintain_order=True).agg(pl.all())

            index: Dict[int, List[dict]] = {}
            for row in grouped.iter_rows(named=True):
                bucket = int(row["bucket"])
                index[bucket] = [
                    {"ts": int(t), "price": float(p), "size": float(sz), "side": sd}
                    for t, p, sz, sd in zip(
                        row["ts_ns"], row["price"], row["size"], row["side"],
                    )
                ]

            self._tick_index[tf] = index
            self._tick_progress = (step + 1) / total_steps
            logger.debug("ReplayEngine Phase 2: indexed %d candles at %ds", len(index), tf)

        self._ticks_ready = True
        self._tick_progress = 1.0

    def get_ticks_for_candle(self, candle_time_sec: int, timeframe_s: int = 15) -> List[dict]:
        """Return pre-indexed ticks for a given candle bucket."""
        tf_index = self._tick_index.get(timeframe_s)
        if tf_index is None:
            return []
        return tf_index.get(candle_time_sec, [])

    def reset(self) -> None:
        """Hard reset: unload everything. Caller must LOAD again to use the engine."""
        self._loaded = False
        self._chart_ready = False
        self._ticks_ready = False
        self._levels_ready = False
        self._chart_progress = 0.0
        self._tick_progress = 0.0
        self._levels_progress = 0.0
        self._progress = 0.0
        self._is_playing = False
        self._speed = 1.0
        self._current_candle_idx = 0
        self._snapshot_records = []
        self._candle_cache = {}
        self._frame_cache = {}
        self._tick_index = {}
        self._scanner = None
        self._total_records = 0
        self._records_scanned = 0
        self._symbol = ""
        logger.info("ReplayEngine: reset — all data unloaded")

    def get_candles(self, timeframe_s: int = 15) -> List[ReplayCandle]:
        """Return the candle array for a given timeframe."""
        if timeframe_s not in self._candle_cache:
            self._candle_cache[timeframe_s] = _build_candles(self._snapshot_records, timeframe_s)
        return self._candle_cache.get(timeframe_s, [])

    def get_frame_at(self, candle_idx: int, timeframe_s: int = 15) -> Optional[ReplayFrame]:
        """Return the ReplayFrame for a given candle index and timeframe."""
        cache_key = (candle_idx, timeframe_s)
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        candles = self.get_candles(timeframe_s)
        if not candles or candle_idx < 0 or candle_idx >= len(candles):
            return None

        candle = candles[candle_idx]

        # Find snapshot records within this candle's time window
        candle_start_ms = candle.time * 1000
        candle_end_ms = candle_start_ms + timeframe_s * 1000

        recs_in_candle = [
            r for r in self._snapshot_records
            if candle_start_ms <= r["timestamp_ms"] < candle_end_ms
        ]

        # Level states from the last snapshot in this candle
        level_states: List[LevelState] = []
        game_state = None
        force_vector = None
        spoof_events: List[SpoofEvent] = []

        if recs_in_candle:
            last = recs_in_candle[-1]
            level_states = last["level_states"]
            game_state = last["game_state"]
            force_vector = last["force_vector"]
            spoof_events = last["spoof_events"]
        elif self._snapshot_records:
            # Use most recent snapshot before candle_start
            candidates = [r for r in self._snapshot_records if r["timestamp_ms"] < candle_start_ms]
            if candidates:
                last = candidates[-1]
                level_states = last["level_states"]
                game_state = last["game_state"]
                force_vector = last["force_vector"]
                spoof_events = last["spoof_events"]

        # Build tape summary
        tape_summary = _empty_tape_summary()
        for rec in recs_in_candle:
            tape_summary = _accumulate_tape(tape_summary, rec.get("trades", []), rec.get("events", []))

        frame = ReplayFrame(
            candle=candle,
            levels=level_states,
            game_state=game_state,
            force_vector=force_vector,
            spoof_events=spoof_events,
            tape_summary=tape_summary,
            timestamp_ms=candle_end_ms - 1,
        )
        self._frame_cache[cache_key] = frame
        return frame

    def total_candles(self, timeframe_s: int = 15) -> int:
        """Return total number of candles for a timeframe."""
        return len(self.get_candles(timeframe_s))

    def seek(self, candle_idx: int) -> None:
        """Jump to a candle index."""
        candles = self.get_candles(self._current_tf_s)
        self._current_candle_idx = max(0, min(candle_idx, len(candles) - 1))

    def seek_by_timestamp(self, timestamp_ms: int) -> None:
        """Jump to the candle closest to a given timestamp."""
        candles = self.get_candles(self._current_tf_s)
        target_s = timestamp_ms // 1000
        best_idx = 0
        for i, c in enumerate(candles):
            if c.time <= target_s:
                best_idx = i
        self._current_candle_idx = best_idx

    def step_forward(self) -> Optional[ReplayFrame]:
        """Advance one candle and return the resulting frame."""
        candles = self.get_candles(self._current_tf_s)
        if self._current_candle_idx < len(candles) - 1:
            self._current_candle_idx += 1
        return self.get_frame_at(self._current_candle_idx, self._current_tf_s)

    def step_backward(self) -> Optional[ReplayFrame]:
        """Go back one candle and return the resulting frame."""
        if self._current_candle_idx > 0:
            self._current_candle_idx -= 1
        return self.get_frame_at(self._current_candle_idx, self._current_tf_s)

    def current_frame(self) -> Optional[ReplayFrame]:
        """Return the frame at the current position."""
        return self.get_frame_at(self._current_candle_idx, self._current_tf_s)

    def set_timeframe(self, timeframe_s: int) -> None:
        """Switch timeframe, preserving approximate time position."""
        if timeframe_s == self._current_tf_s:
            return

        old_candles = self.get_candles(self._current_tf_s)
        current_time_s: Optional[int] = None
        if old_candles and self._current_candle_idx < len(old_candles):
            current_time_s = old_candles[self._current_candle_idx].time

        self._current_tf_s = timeframe_s
        self._frame_cache.clear()  # Invalidate frame cache for new timeframe

        if current_time_s is not None:
            new_candles = self.get_candles(timeframe_s)
            best_idx = 0
            for i, c in enumerate(new_candles):
                if c.time <= current_time_s:
                    best_idx = i
            self._current_candle_idx = best_idx

    def play(self, speed: float = 1.0) -> None:
        self._is_playing = True
        self._speed = speed

    def pause(self) -> None:
        self._is_playing = False

    def get_status(self) -> dict:
        """Return current playback status with three-phase progress fields."""
        candles = self.get_candles(self._current_tf_s)
        current_ts_ms: Optional[int] = None
        if candles and self._current_candle_idx < len(candles):
            current_ts_ms = candles[self._current_candle_idx].time * 1000

        return {
            "loaded": self._loaded,
            "symbol": self._symbol,
            "timeframe_s": self._current_tf_s,
            "current_candle_idx": self._current_candle_idx,
            "total_candles": len(candles),
            "is_playing": self._is_playing,
            "speed": self._speed,
            "current_timestamp_ms": current_ts_ms,
            "progress_pct": (self._current_candle_idx / max(1, len(candles) - 1)) * 100,
            # Three-phase progress
            "chart_ready": self._chart_ready,
            "ticks_ready": self._ticks_ready,
            "levels_ready": self._levels_ready,
            "load_progress_chart": self._chart_progress,
            "load_progress_ticks": self._tick_progress,
            "load_progress_levels": self._levels_progress,
            "load_progress": self._progress,
            "records_scanned": self._records_scanned,
            # Backward-compat aliases
            "state_ready": self._levels_ready,
            "load_progress_state": self._levels_progress,
        }
