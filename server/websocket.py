"""WebSocket manager — broadcasts DepthIndicatorFrame to all connected clients."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

from .config import config

logger = logging.getLogger(__name__)


def _serialize_value(obj: Any) -> Any:
    """Recursively serialize dataclass/enum/timestamp values to JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        # Replace inf/nan with JSON-safe values — these crash JSON.parse in browsers
        if obj != obj:  # nan
            return 0.0
        if obj == float('inf'):
            return 999.0
        if obj == float('-inf'):
            return -999.0
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_value(item) for item in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize_value(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "__slots__"):
        return {k: _serialize_value(getattr(obj, k)) for k in obj.__slots__ if hasattr(obj, k)}
    if hasattr(obj, "__dict__"):
        return {k: _serialize_value(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return str(obj)


def serialize_frame(frame: Any) -> str:
    """Convert a DepthIndicatorFrame to a JSON string for WebSocket broadcast."""
    data = _serialize_value(frame)
    if isinstance(data, dict) and "timestamp_ms" in data:
        ts = data["timestamp_ms"]
        data["timestamp_iso"] = datetime.fromtimestamp(
            ts / 1000.0, tz=timezone.utc
        ).isoformat()
    return json.dumps(data)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts frames."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._last_broadcast: float = 0.0

    @property
    def client_count(self) -> int:
        """Number of currently connected clients."""
        return len(self._clients)

    async def connect(self, ws: WebSocket) -> bool:
        """Accept a WebSocket connection. Returns False if at capacity."""
        if len(self._clients) >= config.ws_max_clients:
            await ws.close(code=1013, reason="Server at capacity")
            return False
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.info("WebSocket connected — %d clients", self.client_count)
        return True

    async def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._clients.discard(ws)
        logger.info("WebSocket disconnected — %d clients", self.client_count)

    async def broadcast(self, message: str) -> None:
        """Send a message to all connected clients, respecting frame rate limit."""
        now = time.monotonic()
        min_interval = 1.0 / config.frame_rate_limit if config.frame_rate_limit > 0 else 0
        if now - self._last_broadcast < min_interval:
            return  # Skip frame — rate limited
        self._last_broadcast = now

        async with self._lock:
            clients = list(self._clients)

        stale: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                stale.append(ws)

        if stale:
            async with self._lock:
                for ws in stale:
                    self._clients.discard(ws)
            logger.info("Removed %d stale clients — %d remain", len(stale), self.client_count)

    async def broadcast_tick(self, tick: dict) -> None:
        """Fast-path broadcast for price ticks — bypasses frame rate limiting."""
        msg = json.dumps(tick)
        async with self._lock:
            clients = list(self._clients)

        stale: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)

        if stale:
            async with self._lock:
                for ws in stale:
                    self._clients.discard(ws)

    async def heartbeat_loop(self) -> None:
        """Periodically ping all clients to detect stale connections."""
        while True:
            await asyncio.sleep(15)
            async with self._lock:
                clients = list(self._clients)
            stale: list[WebSocket] = []
            for ws in clients:
                try:
                    await ws.send_json({"type": "ping", "ts": time.time()})
                except Exception:
                    stale.append(ws)
            if stale:
                async with self._lock:
                    for ws in stale:
                        self._clients.discard(ws)


# Singleton manager
ws_manager = WebSocketManager()
