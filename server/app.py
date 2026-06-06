"""FastAPI application — main entry point for the Staircase Terminal server."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

# Load .env for secrets (DATABENTO_API_KEY etc.) before any imports that need them.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except ImportError:
    pass


def _ns_to_iso(ns_val) -> Optional[str]:
    """Convert a nanosecond epoch int (or datetime) to ISO-8601 UTC string."""
    if ns_val is None:
        return None
    if isinstance(ns_val, int):
        return datetime.fromtimestamp(ns_val / 1_000_000_000, tz=timezone.utc).isoformat()
    if isinstance(ns_val, datetime):
        return ns_val.isoformat()
    # Already a string — return as-is
    s = str(ns_val)
    if s.isdigit():
        return datetime.fromtimestamp(int(s) / 1_000_000_000, tz=timezone.utc).isoformat()
    return s

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

from .config import config, ServerConfig
from .websocket import ws_manager
from .engine_runner import engine_runner
from .backtest_api import router as backtest_router
from .webhook_bridge import router as webhook_router
from .replay_api import router as replay_router

logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop background tasks with the app lifecycle."""
    logger.info("Staircase Terminal starting on %s:%d", config.host, config.port)

    # Start heartbeat loop
    heartbeat_task = asyncio.create_task(ws_manager.heartbeat_loop())

    # Start engine runner
    engine_runner.start()

    yield

    # Shutdown
    heartbeat_task.cancel()
    await engine_runner.stop()
    logger.info("Staircase Terminal stopped")


# ── App ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Staircase Terminal",
    description="Order Book Meta Model — real-time depth analysis & signal engine",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(backtest_router)
app.include_router(webhook_router)
app.include_router(replay_router)


# ── Config Models ──────────────────────────────────────────────────

class ConfigUpdateRequest(BaseModel):
    """Partial config update request."""
    instrument: Optional[str] = None
    mode: Optional[str] = None
    frame_rate_limit: Optional[float] = Field(default=None, ge=1.0, le=60.0)
    ws_max_clients: Optional[int] = Field(default=None, ge=1, le=500)
    webhook_url: Optional[str] = None
    webhook_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    webhook_rate_limit_seconds: Optional[float] = Field(default=None, ge=1.0, le=3600.0)
    risk_max_position: Optional[float] = Field(default=None, ge=0.0)
    risk_abstain_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ── REST Endpoints ─────────────────────────────────────────────────

@app.get("/api/status")
async def get_status() -> dict:
    """Return current engine status."""
    return engine_runner.get_status()


@app.get("/api/config")
async def get_config() -> dict:
    """Return current server configuration."""
    return config.model_dump()


@app.post("/api/config")
async def update_config(update: ConfigUpdateRequest) -> dict:
    """Update server configuration (partial update)."""
    updates = update.model_dump(exclude_none=True)
    if not updates:
        return {"status": "no changes", "config": config.model_dump()}

    # Validate mode
    if "mode" in updates and updates["mode"] not in ("live", "replay", "paused"):
        return JSONResponse(
            status_code=400,
            content={"detail": "mode must be one of: live, replay, paused"},
        )

    for key, value in updates.items():
        if hasattr(config, key):
            setattr(config, key, value)

    logger.info("Config updated: %s", updates)
    return {"status": "updated", "config": config.model_dump()}


# ── Feed Control Endpoints ─────────────────────────────────────────

# ── Preflight Lock Endpoint ────────────────────────────────────────

class PreflightLockRequest(BaseModel):
    """Validate and lock a single preflight step before feed start."""
    step: str = Field(..., description="Step name: mode, datafile, symbol, session, speed, interval, timeframe")
    value: str = Field(..., description="Selected value for this step")
    context: dict = Field(default_factory=dict, description="Accumulated locked values from prior steps")


@app.post("/api/preflight/lock")
async def preflight_lock(request: PreflightLockRequest) -> dict:
    """Validate a single preflight step. Returns { ok, resolved, error }."""
    step = request.step
    value = request.value
    ctx = request.context

    if step == "mode":
        if value not in ("replay", "live"):
            return {"ok": False, "resolved": value, "error": "Invalid mode"}
        return {"ok": True, "resolved": value, "error": None}

    if step == "datafile":
        # Check file exists
        if not value or not os.path.isfile(value):
            return {"ok": False, "resolved": value, "error": "File not found"}
        # Return metadata for the file
        try:
            import databento as db
            store = db.DBNStore.from_file(value)
            start = _ns_to_iso(getattr(store.metadata, "start", None))
            end = _ns_to_iso(getattr(store.metadata, "end", None))
            return {"ok": True, "resolved": value, "error": None, "meta": {"start": start, "end": end}}
        except Exception as e:
            return {"ok": True, "resolved": value, "error": None, "meta": {}}

    if step == "symbol":
        valid_symbols = ["ES.c.0", "NQ.c.0", "CL.c.0", "GC.c.0", "YM.c.0"]
        if value not in valid_symbols:
            return {"ok": False, "resolved": value, "error": f"Invalid symbol: {value}"}
        return {"ok": True, "resolved": value, "error": None}

    if step == "session":
        valid_sessions = ["full", "rth", "overnight", "custom"]
        if value not in valid_sessions:
            return {"ok": False, "resolved": value, "error": f"Invalid session: {value}"}
        return {"ok": True, "resolved": value, "error": None}

    if step == "speed":
        try:
            speed = int(value)
            if speed < 1 or speed > 30:
                return {"ok": False, "resolved": value, "error": "Speed must be 1-30"}
            return {"ok": True, "resolved": value, "error": None}
        except ValueError:
            return {"ok": False, "resolved": value, "error": "Invalid speed value"}

    if step == "interval":
        valid_intervals = ["500", "1000", "2000", "5000"]
        if value not in valid_intervals:
            return {"ok": False, "resolved": value, "error": f"Invalid interval: {value}"}
        return {"ok": True, "resolved": value, "error": None}

    if step == "timeframe":
        valid_timeframes = ["1000", "5000", "15000", "30000", "60000", "300000"]
        if value not in valid_timeframes:
            return {"ok": False, "resolved": value, "error": f"Invalid timeframe: {value}"}
        return {"ok": True, "resolved": value, "error": None}

    return {"ok": False, "resolved": value, "error": f"Unknown step: {step}"}


class LiveFeedRequest(BaseModel):
    """Start a live Databento feed.

    level="L1" — MBP-1 schema (best bid/ask + trades). Available on
    standard subscriptions. Enables: tape, price chart, BBO DOM, cup flip.

    level="L3" — MBO schema (individual orders). Requires MBO live
    subscription. Enables all 5 engine layers.
    """
    symbol: str = Field(default="ES.c.0", description="Instrument symbol")
    dataset: str = Field(default="GLBX.MDP3", description="Databento dataset")
    snapshot_interval_ms: int = Field(default=100, ge=10, le=5000)
    level: str = Field(default="L1", description="Data level: L1 (MBP-1) or L3 (MBO)")


class ReplayFeedRequest(BaseModel):
    """Start a replay from a .dbn.zst file.

    Supports both single-instrument files (es-mbo-*.dbn.zst) and
    full-exchange files (glbx-mdp3-*.dbn.zst) via filter_symbol.
    """
    file_path: str = Field(..., description="Path to .dbn.zst file")
    symbol: str = Field(default="ES", description="Symbol label (ES, NQ, CL, GC, SI)")
    filter_symbol: Optional[str] = Field(default=None, description="Override symbol filter for multi-instrument files")
    time_start: Optional[str] = Field(default=None, description="Start time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    time_end: Optional[str] = Field(default=None, description="End time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    rth_only: bool = Field(default=False, description="Only process RTH hours (09:30-16:00 ET)")
    snapshot_interval_ms: int = Field(default=100, ge=10, le=5000)


@app.post("/api/feed/live")
async def start_live_feed(request: LiveFeedRequest) -> dict:
    """Start live Databento feed (L1 or L3)."""
    if request.level not in ("L1", "L3"):
        raise HTTPException(status_code=400, detail="level must be L1 or L3")
    try:
        return await engine_runner.start_live_feed(
            symbol=request.symbol,
            dataset=request.dataset,
            snapshot_interval_ms=request.snapshot_interval_ms,
            level=request.level,
        )
    except (ValueError, ImportError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/feed/replay")
async def start_replay_feed(request: ReplayFeedRequest) -> dict:
    """Start replay from a Databento .dbn.zst file.

    DEPRECATED for replay mode — use /api/replay/load + /api/replay/play instead.
    The three-phase loader (chart → ticks → levels) provides a better experience.
    This endpoint is retained for backward compatibility and the live-feed path.
    """
    try:
        return await engine_runner.start_replay_feed(
            file_path=request.file_path,
            symbol=request.symbol,
            filter_symbol=request.filter_symbol,
            time_start=request.time_start,
            time_end=request.time_end,
            rth_only=request.rth_only,
            snapshot_interval_ms=request.snapshot_interval_ms,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/feed/stop")
async def stop_feed() -> dict:
    """Stop the current feed."""
    await engine_runner._stop_feed()
    config.mode = "paused"
    return {"status": "stopped"}


# ── Data Files API ─────────────────────────────────────────────────

@app.get("/api/data/files")
async def list_data_files() -> list:
    """List available .dbn.zst replay files with metadata.

    For single-instrument files (es-mbo-*.dbn.zst), returns one entry.
    For full-exchange files (glbx-mdp3-*.dbn.zst), returns one entry per
    supported symbol so the UI can select symbol independently.
    Reads only metadata (no to_df()) so it's safe for large files.
    """
    import glob
    _SUPPORTED = ["ES", "NQ", "CL", "GC", "SI"]
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    files = []

    for path in sorted(glob.glob(os.path.join(data_dir, "*.dbn.zst"))):
        fname = os.path.basename(path)
        size_mb = round(os.path.getsize(path) / 1024 / 1024, 1)

        try:
            import databento as db
            store = db.DBNStore.from_file(path)
            # Read metadata only — never call to_df()
            requested_symbols = getattr(store.metadata, "symbols", [])
            start = _ns_to_iso(getattr(store.metadata, "start", None))
            end   = _ns_to_iso(getattr(store.metadata, "end", None))
            schema = str(store.metadata.schema) if hasattr(store.metadata, "schema") else "mbo"

            is_multi = len(requested_symbols) > 1 or fname.startswith("glbx")

            if is_multi:
                # Expand into one entry per supported symbol
                # Infer available symbols from requested list (e.g. ["NQ.FUT","ES.FUT",...])
                available = []
                for rsym in requested_symbols:
                    root = rsym.split(".")[0]
                    if root in _SUPPORTED:
                        available.append(root)
                if not available:
                    available = _SUPPORTED  # fallback: show all
                for sym in available:
                    files.append({
                        "file": fname,
                        "path": path,
                        "size_mb": size_mb,
                        "symbol": sym,
                        "filter_symbol": sym,
                        "start": start,
                        "end": end,
                        "schema": schema,
                        "multi_instrument": True,
                    })
            else:
                # Single-instrument file: infer symbol from filename
                sym = fname.split("-")[0].upper() if "-" in fname else fname.split("_")[0].upper()
                files.append({
                    "file": fname,
                    "path": path,
                    "size_mb": size_mb,
                    "symbol": sym,
                    "filter_symbol": sym,
                    "start": start,
                    "end": end,
                    "schema": schema,
                    "multi_instrument": False,
                })
        except Exception as e:
            logger.warning("Could not read metadata for %s: %s", fname, e)
            sym = fname.split("-")[0].upper()
            files.append({
                "file": fname,
                "path": path,
                "size_mb": size_mb,
                "symbol": sym,
                "filter_symbol": sym,
                "start": None,
                "end": None,
                "schema": "mbo",
                "multi_instrument": False,
            })

    return files


# ── WebSocket ──────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """WebSocket endpoint for real-time frame streaming."""
    connected = await ws_manager.connect(ws)
    if not connected:
        return

    try:
        while True:
            # Listen for client messages (pong, config commands, etc.)
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                # Handle pong or other client messages
                if data == "pong":
                    continue
            except asyncio.TimeoutError:
                # No message — that's fine, client is just listening
                continue
            except WebSocketDisconnect:
                break
    except Exception:
        pass
    finally:
        await ws_manager.disconnect(ws)


# ── Static Files (frontend) ───────────────────────────────────────

_web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_index():
    """Serve index.html with no-cache headers so JS always re-initializes."""
    from fastapi.responses import HTMLResponse
    index = os.path.join(_web_dir, "index.html")
    with open(index, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


if os.path.isdir(_web_dir):
    app.mount("/", StaticFiles(directory=_web_dir, html=True), name="frontend")


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class NoCacheMiddleware(BaseHTTPMiddleware):
    """Disable browser caching for JS/CSS/HTML during development."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.endswith((".js", ".css", ".html")) or path == "/":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response


app.add_middleware(NoCacheMiddleware)
