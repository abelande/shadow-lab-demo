"""FastAPI router for the new ReplayEngine-backed replay system."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from .replay_engine import ReplayEngine, SUPPORTED_TIMEFRAMES
from .websocket import ws_manager, serialize_frame

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/replay", tags=["replay"])

# Singleton replay engine instance
_engine = ReplayEngine()
_play_task: Optional[asyncio.Task] = None

# BatchScanner for fast candle/trade loading (separate from level processing)
_scanner = None
_scanner_candles: dict = {}   # timeframe_s -> list of candle dicts
_scanner_loading = False


# ── Request / Response Models ──────────────────────────────────────

class LoadRequest(BaseModel):
    file_path: str = Field(..., description="Path to .dbn.zst file")
    symbol: str = Field(default="ES", description="Instrument symbol")
    snapshot_interval_ms: int = Field(default=100, ge=10, le=1000)
    time_start: Optional[str] = Field(default=None, description="Time range start (ISO UTC)")
    time_end: Optional[str] = Field(default=None, description="Time range end (ISO UTC)")
    rth_only: bool = Field(default=False, description="Filter to RTH only")


class PlayRequest(BaseModel):
    speed: float = Field(default=1.0, ge=0.1, le=100.0, description="Playback speed multiplier")


class SeekRequest(BaseModel):
    candle_idx: Optional[int] = Field(default=None, ge=0)
    timestamp_ms: Optional[int] = Field(default=None)


class TimeframeRequest(BaseModel):
    timeframe_s: int = Field(..., description="Timeframe in seconds (1,5,15,60,300)")


# ── Background Play Loop ───────────────────────────────────────────

async def _play_loop(speed: float) -> None:
    """Advance replay one candle at a time, broadcasting frames via WebSocket."""
    global _engine
    try:
        while _engine.is_playing:
            frame = _engine.step_forward()
            if frame is None:
                _engine.pause()
                break

            # Serialize and broadcast
            payload = _build_replay_payload(frame)
            await ws_manager.broadcast(json.dumps(payload))

            # Wait based on timeframe and speed
            tf_s = _engine.get_status()["timeframe_s"]
            delay = tf_s / max(0.01, speed)
            # Cap delay: never wait more than 30s or less than 0.05s
            delay = max(0.05, min(30.0, delay))
            await asyncio.sleep(delay)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("Replay play loop error: %s", exc)
        _engine.pause()


def _build_replay_payload(frame) -> dict:
    """Build the WebSocket payload dict for a replay frame."""
    from ..models import LevelLifecycle, Side

    def _ser_level(lvl) -> dict:
        return {
            "price": lvl.price,
            "side": lvl.side.value if hasattr(lvl.side, "value") else lvl.side,
            "volume": lvl.volume,
            "peak_volume": lvl.peak_volume,
            "order_count": lvl.order_count,
            "lifecycle": lvl.lifecycle.value if hasattr(lvl.lifecycle, "value") else lvl.lifecycle,
            "first_seen_ms": lvl.first_seen_ms,
            "last_seen_ms": lvl.last_seen_ms,
            "age_ms": lvl.age_ms,
            "significance": round(lvl.significance, 4),
            "authenticity": round(lvl.authenticity, 4),
            "spoof_type": lvl.spoof_type.value if lvl.spoof_type else None,
            "iceberg_suspected": lvl.iceberg_suspected,
            "fill_count": lvl.fill_count,
            "refill_count": lvl.refill_count,
        }

    def _ser_candle(c) -> dict:
        return {
            "time": c.time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }

    def _ser_tape(t) -> Optional[dict]:
        if t is None:
            return None
        return {
            "buy_volume": t.buy_volume,
            "sell_volume": t.sell_volume,
            "delta": t.delta,
            "largest_fill_price": t.largest_fill_price,
            "largest_fill_size": t.largest_fill_size,
            "largest_fill_side": t.largest_fill_side,
            "iceberg_hits": t.iceberg_hits,
            "cancel_count_bid": t.cancel_count_bid,
            "cancel_count_ask": t.cancel_count_ask,
        }

    def _ser_game_state(gs) -> Optional[dict]:
        if gs is None:
            return None
        return {
            "state": gs.state.value if hasattr(gs.state, "value") else str(gs.state),
            "pressure": gs.pressure,
            "streak_length": gs.streak_length,
        }

    def _ser_force(fv) -> Optional[dict]:
        if fv is None:
            return None
        return {
            "total_force": fv.total_force,
            "institutional_score": fv.institutional_score,
        }

    def _ser_spoof(se) -> dict:
        return {
            "spoof_type": se.spoof_type.value if hasattr(se.spoof_type, "value") else str(se.spoof_type),
            "price": se.price,
            "side": se.side.value if hasattr(se.side, "value") else str(se.side),
            "confidence": se.confidence,
            "timestamp_ms": se.timestamp_ms,
        }

    return {
        "type": "replay_frame",
        "candle": _ser_candle(frame.candle),
        "levels": [_ser_level(lvl) for lvl in frame.levels],
        "game_state": _ser_game_state(frame.game_state),
        "force_vector": _ser_force(frame.force_vector),
        "spoof_events": [_ser_spoof(se) for se in (frame.spoof_events or [])],
        "tape_summary": _ser_tape(frame.tape_summary),
        "timestamp_ms": frame.timestamp_ms,
    }


# ── API Endpoints ──────────────────────────────────────────────────

@router.post("/load")
async def load_file(request: LoadRequest, background_tasks: BackgroundTasks) -> dict:
    """Load a .dbn.zst file and pre-process into candles + levels.

    This may take several seconds for large files. Poll /api/replay/status
    for load_progress to track completion.
    """
    import os
    if not os.path.isfile(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    # Stop any active playback
    global _play_task
    if _play_task and not _play_task.done():
        _play_task.cancel()
    _engine.pause()

    # Run the three-phase load as an asyncio task (not BackgroundTasks) so
    # the event loop stays free for status polls and WebSocket broadcasts
    # while the CPU-bound BatchScanner and streaming pipeline run.
    async def _do_load():
        try:
            await _engine.load(
                file_path=request.file_path,
                symbol=request.symbol,
                snapshot_interval_ms=request.snapshot_interval_ms,
                time_start=request.time_start,
                time_end=request.time_end,
                rth_only=request.rth_only,
            )
            logger.info("ReplayEngine load complete for %s", request.file_path)
        except Exception as exc:
            logger.error("ReplayEngine load failed: %s", exc)

    asyncio.create_task(_do_load())

    return {"status": "loading", "file_path": request.file_path, "symbol": request.symbol}


@router.post("/play")
async def play(request: PlayRequest) -> dict:
    """Start or resume playback at the given speed multiplier."""
    if not _engine.is_loaded:
        raise HTTPException(status_code=400, detail="No file loaded. Call /api/replay/load first.")

    global _play_task
    if _play_task and not _play_task.done():
        _play_task.cancel()

    _engine.play(speed=request.speed)
    _play_task = asyncio.create_task(_play_loop(request.speed))
    return {"status": "playing", "speed": request.speed}


@router.post("/pause")
async def pause() -> dict:
    """Pause playback."""
    global _play_task
    _engine.pause()
    if _play_task and not _play_task.done():
        _play_task.cancel()
    return {"status": "paused"}


@router.post("/seek")
async def seek(request: SeekRequest) -> dict:
    """Jump to a candle index or timestamp."""
    if not _engine.is_loaded:
        raise HTTPException(status_code=400, detail="No file loaded.")

    if request.candle_idx is not None:
        _engine.seek(request.candle_idx)
    elif request.timestamp_ms is not None:
        _engine.seek_by_timestamp(request.timestamp_ms)
    else:
        raise HTTPException(status_code=400, detail="Provide candle_idx or timestamp_ms.")

    frame = _engine.current_frame()
    payload = _build_replay_payload(frame) if frame else {}
    await ws_manager.broadcast(json.dumps(payload))

    return {"status": "seeked", **_engine.get_status()}


@router.post("/step")
async def step_forward() -> dict:
    """Advance one candle and broadcast the frame."""
    if not _engine.is_loaded:
        raise HTTPException(status_code=400, detail="No file loaded.")

    frame = _engine.step_forward()
    if frame:
        payload = _build_replay_payload(frame)
        await ws_manager.broadcast(json.dumps(payload))

    return {"status": "stepped", **_engine.get_status()}


@router.post("/step-back")
async def step_backward() -> dict:
    """Go back one candle and broadcast the frame."""
    if not _engine.is_loaded:
        raise HTTPException(status_code=400, detail="No file loaded.")

    frame = _engine.step_backward()
    if frame:
        payload = _build_replay_payload(frame)
        await ws_manager.broadcast(json.dumps(payload))

    return {"status": "stepped-back", **_engine.get_status()}


@router.post("/reset")
async def reset_engine() -> dict:
    """Hard reset: unload all data and stop playback.

    After reset, the engine is in its initial state — the frontend must
    call /api/replay/load again before any playback or scrubbing.
    """
    global _play_task, _scanner, _scanner_candles, _scanner_loading
    if _play_task and not _play_task.done():
        _play_task.cancel()
    _engine.reset()
    _scanner = None
    _scanner_candles = {}
    _scanner_loading = False
    return {"status": "reset"}


@router.get("/status")
async def get_status() -> dict:
    """Return current replay state with three-phase progress.

    Phase 1 — chart:  chart_ready / load_progress_chart (0.0–1.0)
    Phase 2 — ticks:  ticks_ready / load_progress_ticks (0.0–1.0)
    Phase 3 — levels: levels_ready / load_progress_levels (0.0–1.0)

    Backward-compat aliases: state_ready, load_progress_state.
    """
    return _engine.get_status()


@router.get("/candles")
async def get_candles(timeframe_s: int = 15) -> dict:
    """Bulk fetch the full candle array for initial chart load.

    Track 3B: Returns candles as soon as chart_ready is True (Phase 1),
    even before level state streaming (Phase 2) completes.
    """
    if not _engine._chart_ready and not _engine.is_loaded:
        raise HTTPException(status_code=400, detail="No file loaded.")

    if timeframe_s not in SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported timeframe. Use one of: {SUPPORTED_TIMEFRAMES}",
        )

    candles = _engine.get_candles(timeframe_s)
    return {
        "timeframe_s": timeframe_s,
        "count": len(candles),
        "candles": [
            {"time": c.time, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles
        ],
    }


@router.get("/ticks")
async def get_ticks(candle_time: int, timeframe_s: int = 15) -> dict:
    """Return pre-indexed ticks for a candle from the Phase 2 tick index."""
    if not _engine._ticks_ready:
        raise HTTPException(status_code=400, detail="Tick index not ready. Wait for Phase 2.")
    ticks = _engine.get_ticks_for_candle(candle_time, timeframe_s)
    return {"candle_time": candle_time, "timeframe_s": timeframe_s, "count": len(ticks), "ticks": ticks}


class FastLoadRequest(BaseModel):
    file_path: str = Field(..., description="Path to .dbn.zst file")
    symbol: str = Field(default="NQ", description="Symbol filter (NQ, ES, CL, etc.)")
    timeframe_s: int = Field(default=15, description="Default candle timeframe in seconds")


@router.post("/fast-load")
async def fast_load(request: FastLoadRequest, background_tasks: BackgroundTasks) -> dict:
    """Fast-load a .dbn.zst file using BatchScanner (Polars).

    Builds candles and trade index in ~45s for a full session.
    Returns immediately — poll /api/replay/fast-load/status for progress.
    After loading, use /api/replay/fast-load/candles and /api/replay/fast-load/ticks.
    """
    import os
    if not os.path.isfile(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    global _scanner, _scanner_candles, _scanner_loading
    _scanner_loading = True
    _scanner_candles = {}

    async def _do_fast_load():
        global _scanner, _scanner_candles, _scanner_loading
        try:
            from ..ingestion.batch_scanner import BatchScanner
            scanner = BatchScanner()

            # Run in executor since to_df() is CPU-bound blocking
            import asyncio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: scanner.load(request.file_path, symbol_filter=request.symbol),
            )

            _scanner = scanner
            # Pre-build candles at requested timeframe
            candles = scanner.build_ohlcv(timeframe_seconds=request.timeframe_s)
            _scanner_candles[request.timeframe_s] = candles
            _scanner_loading = False
            logger.info("Fast-load complete: %d candles at %ds", len(candles), request.timeframe_s)
        except Exception as exc:
            logger.error("Fast-load failed: %s", exc)
            _scanner_loading = False

    background_tasks.add_task(_do_fast_load)
    return {"status": "loading", "file_path": request.file_path, "symbol": request.symbol}


@router.get("/fast-load/status")
async def fast_load_status() -> dict:
    """Poll fast-load progress."""
    global _scanner, _scanner_loading
    if _scanner_loading:
        return {"status": "loading"}
    if _scanner is not None and _scanner.is_loaded:
        candle_counts = {str(k): len(v) for k, v in _scanner_candles.items()}
        trades_count = len(_scanner.get_trades()) if _scanner.is_loaded else 0
        return {"status": "ready", "candle_counts": candle_counts, "trades_count": trades_count}
    return {"status": "idle"}


@router.get("/fast-load/candles")
async def fast_load_candles(timeframe_s: int = 15) -> dict:
    """Get candles from the fast-loaded data. Builds on-demand if timeframe not cached."""
    global _scanner, _scanner_candles
    if _scanner is None or not _scanner.is_loaded:
        raise HTTPException(status_code=400, detail="No fast-load data. Call /api/replay/fast-load first.")

    if timeframe_s not in _scanner_candles:
        _scanner_candles[timeframe_s] = _scanner.build_ohlcv(timeframe_seconds=timeframe_s)

    candles = _scanner_candles[timeframe_s]
    return {
        "timeframe_s": timeframe_s,
        "count": len(candles),
        "candles": candles,
    }


@router.get("/fast-load/ticks")
async def fast_load_ticks(candle_time: int, timeframe_s: int = 15) -> dict:
    """Get individual trade ticks within a specific candle's time window.

    Used for tick-chart rendering during replay playback.
    """
    global _scanner
    if _scanner is None or not _scanner.is_loaded:
        raise HTTPException(status_code=400, detail="No fast-load data.")

    ticks = _scanner.trades_for_candle(candle_time, timeframe_s)
    return {
        "candle_time": candle_time,
        "timeframe_s": timeframe_s,
        "count": len(ticks),
        "ticks": ticks,
    }


@router.get("/fast-load/stats")
async def fast_load_stats() -> dict:
    """Get session statistics from fast-loaded data."""
    global _scanner
    if _scanner is None or not _scanner.is_loaded:
        raise HTTPException(status_code=400, detail="No fast-load data.")
    return _scanner.session_stats()


# ── Event Query / Search / Extract Endpoints ─────────────────────


class EventQueryRequest(BaseModel):
    actions: Optional[list[str]] = Field(default=None, description="Filter by action: ADD, CANCEL, MODIFY, FILL")
    side: Optional[str] = Field(default=None, description="Filter by side: BID or ASK")
    order_id: Optional[str] = Field(default=None, description="Exact order_id lookup")
    price_min: Optional[float] = Field(default=None, description="Min price (e.g. 19998.00)")
    price_max: Optional[float] = Field(default=None, description="Max price")
    size_min: Optional[float] = Field(default=None, description="Min order size")
    size_max: Optional[float] = Field(default=None, description="Max order size")
    time_start: Optional[str] = Field(default=None, description="Start time (ISO UTC, e.g. 2026-03-27T14:30:00)")
    time_end: Optional[str] = Field(default=None, description="End time (ISO UTC)")
    limit: int = Field(default=1000, ge=1, le=50000, description="Max rows returned")
    offset: int = Field(default=0, ge=0, description="Skip first N matches")


class ExtractRequest(BaseModel):
    time_start: str = Field(..., description="Start time (ISO UTC)")
    time_end: str = Field(..., description="End time (ISO UTC)")
    output_path: Optional[str] = Field(default=None, description="Path to write .parquet file (optional)")


class AggregateRequest(BaseModel):
    time_start: Optional[str] = Field(default=None, description="Start time (ISO UTC)")
    time_end: Optional[str] = Field(default=None, description="End time (ISO UTC)")


@router.post("/events/query")
async def query_events(request: EventQueryRequest) -> dict:
    """Search and filter MBO events from the loaded data.

    Supports filtering by action type, side, order_id, price/size ranges,
    and time ranges. Results are paginated via limit/offset.

    Examples:
      - All cancels: {"actions": ["CANCEL"]}
      - Large fills on the bid: {"actions": ["FILL"], "side": "BID", "size_min": 50}
      - Events at a price level: {"price_min": 19998.0, "price_max": 19998.0}
      - Custom time window: {"time_start": "2026-03-27T14:30:00", "time_end": "2026-03-27T15:00:00"}
    """
    global _scanner
    if _scanner is None or not _scanner.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded. Call /api/replay/fast-load first.")

    return _scanner.query_events(
        actions=request.actions,
        side=request.side,
        order_id=request.order_id,
        price_min=request.price_min,
        price_max=request.price_max,
        size_min=request.size_min,
        size_max=request.size_max,
        time_start=request.time_start,
        time_end=request.time_end,
        limit=request.limit,
        offset=request.offset,
    )


@router.get("/events/order/{order_id}")
async def order_lifecycle(order_id: str) -> dict:
    """Trace the full lifecycle of a single order.

    Returns every event (ADD → MODIFY → FILL/CANCEL) with timestamps,
    prices, and sizes at each step.
    """
    global _scanner
    if _scanner is None or not _scanner.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded. Call /api/replay/fast-load first.")

    return _scanner.order_lifecycle(order_id)


@router.post("/events/aggregate")
async def aggregate_events(request: AggregateRequest) -> dict:
    """Aggregate event counts and volume by action type.

    Answers: "how many ADDs vs CANCELs vs FILLs in this time window?"
    """
    global _scanner
    if _scanner is None or not _scanner.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded. Call /api/replay/fast-load first.")

    return _scanner.aggregate_by_action(
        time_start=request.time_start,
        time_end=request.time_end,
    )


@router.post("/events/extract")
async def extract_range(request: ExtractRequest) -> dict:
    """Extract a time range of raw events for offline replay or analysis.

    Optionally writes a Parquet file that can be re-loaded by Polars/Pandas.
    Without output_path, returns metadata and action breakdown only.
    """
    global _scanner
    if _scanner is None or not _scanner.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded. Call /api/replay/fast-load first.")

    return _scanner.extract_range(
        time_start=request.time_start,
        time_end=request.time_end,
        output_path=request.output_path,
    )


@router.post("/timeframe")
async def set_timeframe(request: TimeframeRequest) -> dict:
    """Switch timeframe. Re-aggregates candles; levels are time-based and persist."""
    if not _engine.is_loaded:
        raise HTTPException(status_code=400, detail="No file loaded.")

    if request.timeframe_s not in SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported timeframe. Use one of: {SUPPORTED_TIMEFRAMES}",
        )

    _engine.set_timeframe(request.timeframe_s)
    candles = _engine.get_candles(request.timeframe_s)

    return {
        "status": "timeframe_set",
        "timeframe_s": request.timeframe_s,
        "total_candles": len(candles),
        **_engine.get_status(),
    }
