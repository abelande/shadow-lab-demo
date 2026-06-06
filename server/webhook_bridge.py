"""TradingView webhook bridge — fires webhooks on high-confidence signals."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from .config import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/webhook", tags=["webhook"])


# ── Models ─────────────────────────────────────────────────────────

class WebhookConfigureRequest(BaseModel):
    """Configure the webhook URL and thresholds."""
    url: str = Field(..., description="TradingView webhook URL")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rate_limit_seconds: Optional[float] = Field(default=None, ge=1.0, le=3600.0)


class WebhookPayload(BaseModel):
    """Payload sent to TradingView webhook."""
    direction: str
    confidence: float
    price: float
    instrument: str
    regime: str
    timestamp: str


class WebhookHistoryEntry(BaseModel):
    """Record of a sent webhook."""
    sent_at: str
    payload: WebhookPayload
    status_code: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


# ── Webhook Manager ───────────────────────────────────────────────

class WebhookBridge:
    """Manages TradingView webhook delivery with rate limiting."""

    def __init__(self, max_history: int = 100) -> None:
        self._last_sent: float = 0.0
        self._history: deque[WebhookHistoryEntry] = deque(maxlen=max_history)
        self._http_session: Optional[object] = None

    @property
    def history(self) -> list[WebhookHistoryEntry]:
        """Return webhook history newest-first."""
        return list(reversed(self._history))

    def configure(self, request: WebhookConfigureRequest) -> dict:
        """Update webhook configuration."""
        config.webhook_url = request.url
        if request.confidence_threshold is not None:
            config.webhook_confidence_threshold = request.confidence_threshold
        if request.rate_limit_seconds is not None:
            config.webhook_rate_limit_seconds = request.rate_limit_seconds

        return {
            "status": "configured",
            "url": config.webhook_url,
            "confidence_threshold": config.webhook_confidence_threshold,
            "rate_limit_seconds": config.webhook_rate_limit_seconds,
        }

    async def maybe_fire(
        self,
        direction: float,
        confidence: float,
        price: float,
        regime: str,
        timestamp_ms: int,
    ) -> bool:
        """Fire webhook if conditions are met. Returns True if sent.

        Conditions:
        1. Webhook URL is configured
        2. Confidence exceeds threshold
        3. Rate limit has elapsed
        """
        if not config.webhook_url:
            return False

        if confidence < config.webhook_confidence_threshold:
            return False

        now = time.monotonic()
        if now - self._last_sent < config.webhook_rate_limit_seconds:
            return False

        # Build payload
        dir_str = "BUY" if direction > 0 else "SELL" if direction < 0 else "NEUTRAL"
        ts_iso = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()

        payload = WebhookPayload(
            direction=dir_str,
            confidence=round(confidence, 4),
            price=round(price, 6),
            instrument=config.instrument,
            regime=regime,
            timestamp=ts_iso,
        )

        # Send HTTP POST
        entry = await self._send(payload)
        self._history.append(entry)
        self._last_sent = now

        return entry.success

    async def _send(self, payload: WebhookPayload) -> WebhookHistoryEntry:
        """Send the webhook via HTTP POST."""
        import aiohttp

        sent_at = datetime.now(timezone.utc).isoformat()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.webhook_url,
                    json=payload.model_dump(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    return WebhookHistoryEntry(
                        sent_at=sent_at,
                        payload=payload,
                        status_code=resp.status,
                        success=200 <= resp.status < 300,
                    )
        except ImportError:
            # aiohttp not installed — use urllib fallback
            return await self._send_urllib(payload, sent_at)
        except Exception as e:
            logger.warning("Webhook send failed: %s", e)
            return WebhookHistoryEntry(
                sent_at=sent_at,
                payload=payload,
                success=False,
                error=str(e),
            )

    async def _send_urllib(self, payload: WebhookPayload, sent_at: str) -> WebhookHistoryEntry:
        """Fallback HTTP POST using urllib (no aiohttp dependency)."""
        import json
        import urllib.request
        import urllib.error

        try:
            data = json.dumps(payload.model_dump()).encode("utf-8")
            req = urllib.request.Request(
                config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(req, timeout=10)
            )
            return WebhookHistoryEntry(
                sent_at=sent_at,
                payload=payload,
                status_code=resp.status,
                success=200 <= resp.status < 300,
            )
        except urllib.error.HTTPError as e:
            return WebhookHistoryEntry(
                sent_at=sent_at,
                payload=payload,
                status_code=e.code,
                success=False,
                error=str(e),
            )
        except Exception as e:
            return WebhookHistoryEntry(
                sent_at=sent_at,
                payload=payload,
                success=False,
                error=str(e),
            )


# Singleton
webhook_bridge = WebhookBridge()


# ── Routes ─────────────────────────────────────────────────────────

@router.post("/configure")
async def webhook_configure(request: WebhookConfigureRequest) -> dict:
    """Configure the TradingView webhook URL and thresholds."""
    return webhook_bridge.configure(request)


@router.get("/history")
async def webhook_history(limit: int = 20) -> list[WebhookHistoryEntry]:
    """Get recent webhook send history."""
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    if limit > 100:
        limit = 100
    return webhook_bridge.history[:limit]
