"""REST API for backtesting / historical replay."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .config import config
from .engine_runner import engine_runner
from .websocket import ws_manager, serialize_frame

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/backtest", tags=["backtest"])


# ── Request/Response Models ────────────────────────────────────────

class BacktestStartRequest(BaseModel):
    """Parameters to start a historical replay."""
    file_path: str = Field(..., description="Path to snapshot data file (JSON lines)")
    speed_multiplier: float = Field(default=1.0, ge=0.1, le=100.0, description="Replay speed")
    start_time: Optional[int] = Field(default=None, description="Start timestamp_ms (inclusive)")
    end_time: Optional[int] = Field(default=None, description="End timestamp_ms (inclusive)")


class BacktestSeekRequest(BaseModel):
    """Parameters for seeking to a specific position."""
    timestamp_ms: Optional[int] = Field(default=None, description="Seek to this timestamp")
    frame_index: Optional[int] = Field(default=None, ge=0, description="Seek to this frame index")


class BacktestStatusResponse(BaseModel):
    """Current backtest status."""
    mode: str
    total_frames: int
    current_frame: int
    current_timestamp_ms: Optional[int]
    elapsed_seconds: float
    speed_multiplier: float
    progress_pct: float


class BacktestResultsResponse(BaseModel):
    """Performance metrics from backtest."""
    total_frames: int
    elapsed_seconds: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    total_pnl: float
    win_count: int
    loss_count: int
    avg_confidence: float


# ── Backtest Engine ────────────────────────────────────────────────

class BacktestRunner:
    """Manages historical replay of snapshot data through the pipeline."""

    def __init__(self) -> None:
        self._snapshots: list = []
        self._current_index: int = 0
        self._mode: str = "idle"  # idle, running, paused, finished
        self._task: Optional[asyncio.Task] = None
        self._speed: float = 1.0
        self._start_time: float = 0.0
        self._elapsed: float = 0.0
        # Performance tracking
        self._signals: list[dict] = []
        self._pnl: float = 0.0
        self._last_price: Optional[float] = None

    @property
    def mode(self) -> str:
        return self._mode

    def _load_snapshots(self, file_path: str, start_time: Optional[int], end_time: Optional[int]) -> None:
        """Load snapshot data from a JSON lines file."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Snapshot file not found: {file_path}")

        self._snapshots = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                ts = data.get("timestamp_ms", 0)
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                self._snapshots.append(data)

        if not self._snapshots:
            raise ValueError("No snapshots found in the specified range")

        logger.info("Loaded %d snapshots from %s", len(self._snapshots), file_path)

    async def start(self, request: BacktestStartRequest) -> dict:
        """Start a backtest replay."""
        if self._mode == "running":
            raise ValueError("Backtest already running — stop first")

        self._load_snapshots(request.file_path, request.start_time, request.end_time)
        self._speed = request.speed_multiplier
        self._current_index = 0
        self._start_time = time.monotonic()
        self._elapsed = 0.0
        self._signals = []
        self._pnl = 0.0
        self._last_price = None
        self._mode = "running"
        config.mode = "replay"

        engine_runner.init_pipeline()
        self._task = asyncio.get_event_loop().create_task(self._replay_loop())

        return {"status": "started", "total_frames": len(self._snapshots)}

    async def _replay_loop(self) -> None:
        """Main replay loop."""
        try:
            while self._current_index < len(self._snapshots) and self._mode == "running":
                await self._process_frame(self._current_index)
                self._current_index += 1

                # Timing: respect speed multiplier
                interval = (1.0 / config.frame_rate_limit) / self._speed
                await asyncio.sleep(max(0.01, interval))

            if self._mode == "running":
                self._mode = "finished"
                config.mode = "paused"
                logger.info("Backtest finished — %d frames", len(self._snapshots))

        except asyncio.CancelledError:
            logger.info("Backtest cancelled")
        except Exception:
            logger.exception("Backtest error")
            self._mode = "idle"

    async def _process_frame(self, index: int) -> None:
        """Process a single snapshot frame."""
        from ..models import OrderBookSnapshot, OrderBookLevel, Side

        data = self._snapshots[index]

        def _build_level(d: dict) -> object:
            return OrderBookLevel(
                price=d.get("price", 0),
                side=Side(d.get("side", "BID")),
                volume=d.get("volume", 0),
                order_count=d.get("order_count", 0),
            )

        snapshot = OrderBookSnapshot(
            timestamp_ms=data.get("timestamp_ms", 0),
            symbol=data.get("symbol", config.instrument),
            bids=[_build_level(b) for b in data.get("bids", [])],
            asks=[_build_level(a) for a in data.get("asks", [])],
        )

        # Run pipeline
        frame = engine_runner._pipeline.run(snapshot)

        # Track signal for performance
        self._track_signal(frame, snapshot)

        # Broadcast
        msg = serialize_frame(frame)
        await ws_manager.broadcast(msg)

        self._elapsed = time.monotonic() - self._start_time

    def _track_signal(self, frame: object, snapshot: object) -> None:
        """Track signals for performance metrics."""
        mid = snapshot.mid_price
        if mid is None:
            return

        signal = {
            "direction": frame.direction,
            "confidence": frame.confidence,
            "price": mid,
            "timestamp_ms": frame.timestamp_ms,
        }
        self._signals.append(signal)

        # Simple PnL tracking: if we had a direction last tick, calculate return
        if self._last_price is not None and len(self._signals) >= 2:
            prev = self._signals[-2]
            price_change = mid - self._last_price
            self._pnl += price_change * prev["direction"]

        self._last_price = mid

    def pause(self) -> dict:
        """Pause the replay."""
        if self._mode != "running":
            raise ValueError(f"Cannot pause — mode is {self._mode}")
        self._mode = "paused"
        return {"status": "paused", "frame": self._current_index}

    def resume(self) -> dict:
        """Resume the replay."""
        if self._mode != "paused":
            raise ValueError(f"Cannot resume — mode is {self._mode}")
        self._mode = "running"
        self._task = asyncio.get_event_loop().create_task(self._replay_loop())
        return {"status": "resumed", "frame": self._current_index}

    async def stop(self) -> dict:
        """Stop and reset the replay."""
        self._mode = "idle"
        config.mode = "paused"
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        result = {"status": "stopped", "frames_processed": self._current_index}
        self._current_index = 0
        self._snapshots = []
        return result

    async def step(self) -> dict:
        """Advance exactly one frame."""
        if self._mode not in ("paused", "idle", "finished"):
            raise ValueError(f"Cannot step — mode is {self._mode}")
        if self._current_index >= len(self._snapshots):
            raise ValueError("No more frames to step through")

        engine_runner.init_pipeline()
        await self._process_frame(self._current_index)
        result = {"status": "stepped", "frame": self._current_index}
        self._current_index += 1
        return result

    def seek(self, request: BacktestSeekRequest) -> dict:
        """Jump to a specific position."""
        if not self._snapshots:
            raise ValueError("No snapshots loaded — start a backtest first")

        if request.frame_index is not None:
            if request.frame_index >= len(self._snapshots):
                raise ValueError(f"Frame index {request.frame_index} out of range (0-{len(self._snapshots)-1})")
            self._current_index = request.frame_index
        elif request.timestamp_ms is not None:
            # Find closest frame
            best_idx = 0
            best_diff = float("inf")
            for i, snap in enumerate(self._snapshots):
                diff = abs(snap.get("timestamp_ms", 0) - request.timestamp_ms)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            self._current_index = best_idx
        else:
            raise ValueError("Provide either timestamp_ms or frame_index")

        return {"status": "seeked", "frame": self._current_index}

    def get_status(self) -> BacktestStatusResponse:
        """Current replay status."""
        total = len(self._snapshots)
        progress = (self._current_index / total * 100) if total > 0 else 0
        current_ts = None
        if self._snapshots and self._current_index < total:
            current_ts = self._snapshots[self._current_index].get("timestamp_ms")

        return BacktestStatusResponse(
            mode=self._mode,
            total_frames=total,
            current_frame=self._current_index,
            current_timestamp_ms=current_ts,
            elapsed_seconds=round(self._elapsed, 2),
            speed_multiplier=self._speed,
            progress_pct=round(progress, 2),
        )

    def get_results(self) -> BacktestResultsResponse:
        """Compute performance metrics from signals."""
        if not self._signals:
            return BacktestResultsResponse(
                total_frames=0, elapsed_seconds=0, sharpe_ratio=0,
                max_drawdown=0, hit_rate=0, total_pnl=0,
                win_count=0, loss_count=0, avg_confidence=0,
            )

        # Calculate returns
        returns: list[float] = []
        for i in range(1, len(self._signals)):
            prev = self._signals[i - 1]
            curr = self._signals[i]
            price_change = curr["price"] - prev["price"]
            ret = price_change * prev["direction"]
            returns.append(ret)

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        # Sharpe ratio (annualized, assuming ~252 trading days, ~6.5 hrs/day)
        import statistics
        mean_ret = statistics.mean(returns) if returns else 0
        std_ret = statistics.stdev(returns) if len(returns) > 1 else 1
        sharpe = (mean_ret / std_ret) * (len(returns) ** 0.5) if std_ret > 0 else 0

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in returns:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        avg_conf = statistics.mean([s["confidence"] for s in self._signals])

        return BacktestResultsResponse(
            total_frames=len(self._signals),
            elapsed_seconds=round(self._elapsed, 2),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            hit_rate=round(len(wins) / len(returns), 4) if returns else 0,
            total_pnl=round(self._pnl, 4),
            win_count=len(wins),
            loss_count=len(losses),
            avg_confidence=round(avg_conf, 4),
        )


# Singleton
backtest_runner = BacktestRunner()


# ── Routes ─────────────────────────────────────────────────────────

@router.post("/start")
async def backtest_start(request: BacktestStartRequest) -> dict:
    """Start a historical replay backtest."""
    try:
        return await backtest_runner.start(request)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/pause")
async def backtest_pause() -> dict:
    """Pause the running backtest."""
    try:
        return backtest_runner.pause()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/resume")
async def backtest_resume() -> dict:
    """Resume a paused backtest."""
    try:
        return backtest_runner.resume()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stop")
async def backtest_stop() -> dict:
    """Stop and reset the backtest."""
    return await backtest_runner.stop()


@router.post("/step")
async def backtest_step() -> dict:
    """Advance exactly one frame in the backtest."""
    try:
        return await backtest_runner.step()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/seek")
async def backtest_seek(request: BacktestSeekRequest) -> dict:
    """Seek to a specific timestamp or frame index."""
    try:
        return backtest_runner.seek(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status")
async def backtest_status() -> BacktestStatusResponse:
    """Get current backtest replay status."""
    return backtest_runner.get_status()


@router.get("/results")
async def backtest_results() -> BacktestResultsResponse:
    """Get performance metrics from the backtest."""
    return backtest_runner.get_results()
