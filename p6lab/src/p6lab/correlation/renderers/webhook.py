"""
WebhookRenderer — POST tier-filtered matches to Discord or Slack.

Supports both Discord and Slack incoming-webhook formats with a common
throttle so neither service rate-limits us.

Required user-provided config
-----------------------------
Discord:
    1. Server Settings → Integrations → Webhooks → New Webhook
    2. Select the target channel
    3. Copy the webhook URL (https://discord.com/api/webhooks/<id>/<token>)
    4. Expose as env var: ``P6LAB_DISCORD_WEBHOOK_URL``

Slack:
    1. Create an Incoming Webhook app at https://api.slack.com/apps
    2. Enable incoming webhooks, install to workspace, pick channel
    3. Copy the webhook URL
    4. Expose as env var: ``P6LAB_SLACK_WEBHOOK_URL``

Wiring:

    from p6lab.correlation.renderers import WebhookRenderer
    import os
    if url := os.environ.get("P6LAB_DISCORD_WEBHOOK_URL"):
        broker.subscribe(WebhookRenderer(url, platform="discord", tier_filter={"A"}))
    if url := os.environ.get("P6LAB_SLACK_WEBHOOK_URL"):
        broker.subscribe(WebhookRenderer(url, platform="slack", tier_filter={"A", "B"}))
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier → color (Discord embed colors; Slack uses the same palette via attachment color)
# ---------------------------------------------------------------------------

_TIER_COLORS = {
    "A": 0x4CAF50,   # green
    "B": 0xFFEB3B,   # yellow
    "C": 0x9E9E9E,   # gray
}


class WebhookRenderer:
    """Token-bucket-throttled webhook POSTer for match alerts.

    Parameters
    ----------
    url
        The full webhook URL. Never hardcode — pass from an env var.
    platform
        ``"discord"`` or ``"slack"``. Determines the payload schema.
    tier_filter
        Only send matches whose ``tier`` / ``confidence_tier`` is in this set.
        Default: tier A and B.
    min_score
        Additional floor on ``ensemble_score`` (0-1). Applied **after** the
        tier filter. Set e.g. ``min_score=0.80`` to fire only on tier A plus
        "high-confidence tier B" — matches that cleared the tier cutoff but
        also scored above your personal confidence bar. Default ``None`` =
        no extra filter (tier gate alone decides).
    max_per_minute
        Token-bucket ceiling. Excess matches are silently dropped with a
        warning log. Default 20 — well under Discord's 30/min webhook cap,
        with headroom for retries.
    timeout_seconds
        HTTP timeout for each POST. Short by design — we do not block the
        broker dispatch thread. If the webhook endpoint is down, we drop.
    """

    def __init__(
        self,
        url: str,
        *,
        platform: str = "discord",
        tier_filter: set[str] | None = None,
        min_score: float | None = None,
        max_per_minute: int = 20,
        timeout_seconds: float = 2.0,
        username: str = "p6lab",
    ) -> None:
        if not url or not url.startswith(("http://", "https://")):
            raise ValueError(f"invalid webhook URL: {url!r}")
        if platform not in ("discord", "slack"):
            raise ValueError(f"platform must be 'discord' or 'slack', got {platform!r}")
        if min_score is not None and not 0.0 <= min_score <= 1.0:
            raise ValueError(f"min_score must be in [0, 1], got {min_score!r}")
        self.url = url
        self.platform = platform
        self.tier_filter = tier_filter or {"A", "B"}
        self.min_score = min_score
        self.max_per_minute = max_per_minute
        self.timeout = timeout_seconds
        self.username = username

        self._lock = threading.Lock()
        self._send_times: deque[float] = deque()   # ts of posts in last 60s
        self.posts_sent = 0
        self.posts_dropped = 0
        # Fire webhook POSTs on a background thread so the broker never blocks
        self._thread_pool: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # Broker subscriber interface
    # ------------------------------------------------------------------

    def __call__(self, match: Any) -> None:
        tier = _attr(match, "confidence_tier") or _attr(match, "tier")
        if tier not in self.tier_filter:
            return
        if self.min_score is not None:
            score = float(_attr(match, "ensemble_score") or 0.0)
            if score < self.min_score:
                return
        if not self._consume_token():
            self.posts_dropped += 1
            logger.warning("WebhookRenderer: throttled (>%d/min) — dropped match",
                           self.max_per_minute)
            return

        # Detach HTTP I/O from the broker dispatch — subscribers must not block.
        # Copy the current ContextVar state so the HTTP thread inherits the
        # engine's correlation_id (otherwise log lines from `_post` would
        # have an empty context).
        import contextvars
        ctx = contextvars.copy_context()
        t = threading.Thread(
            target=lambda: ctx.run(self._post, match),
            daemon=True, name=f"{self.platform}-webhook",
        )
        t.start()
        # Keep a weak reference list for test-side cleanup (best-effort)
        self._thread_pool = [th for th in self._thread_pool if th.is_alive()]
        self._thread_pool.append(t)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _consume_token(self) -> bool:
        now = time.time()
        with self._lock:
            while self._send_times and (now - self._send_times[0]) > 60.0:
                self._send_times.popleft()
            if len(self._send_times) >= self.max_per_minute:
                return False
            self._send_times.append(now)
            return True

    def _post(self, match: Any) -> None:
        payload = (self._discord_payload(match)
                   if self.platform == "discord"
                   else self._slack_payload(match))
        body = json.dumps(payload).encode("utf-8")
        print(f"DEBUG URL: {self.url!r}")
        print(f"DEBUG BODY: {body[:500]}")
        req = urlrequest.Request(
            self.url, data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "p6lab-correlation-alerts/1.0",
                },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                if 200 <= resp.status < 300:
                    self.posts_sent += 1
                else:
                    logger.warning("%s webhook: HTTP %d", self.platform, resp.status)
        except URLError as e:
            logger.warning("%s webhook POST failed: %s", self.platform, e)
        except Exception:
            logger.exception("%s webhook unexpected error", self.platform)

    def _discord_payload(self, m: Any) -> dict:
        tier = _attr(m, "confidence_tier") or _attr(m, "tier") or "?"
        score = float(_attr(m, "ensemble_score") or 0.0)
        direction = _attr(m, "expected_direction") or "neutral"
        atr = float(_attr(m, "expected_move_atr") or 0.0)
        instrument = _attr(m, "instrument") or "?"
        regime = _attr(m, "regime") or "?"
        pattern_id = _attr(m, "pattern_id") or "?"

        arrow = {"bull": "📈", "bear": "📉"}.get(direction, "↔")
        return {
            "username": self.username,
            "embeds": [{
                "title": f"Tier {tier} — {pattern_id}",
                "description": f"{arrow} **{direction}** · {atr:.2f} ATR expected",
                "color": _TIER_COLORS.get(tier, 0x666666),
                "fields": [
                    {"name": "Score",      "value": f"{score:.3f}", "inline": True},
                    {"name": "Instrument", "value": instrument,      "inline": True},
                    {"name": "Regime",     "value": regime,          "inline": True},
                ],
                "timestamp": _isotime(_attr(m, "match_window_end_ms")),
            }],
        }

    def _slack_payload(self, m: Any) -> dict:
        tier = _attr(m, "confidence_tier") or _attr(m, "tier") or "?"
        score = float(_attr(m, "ensemble_score") or 0.0)
        direction = _attr(m, "expected_direction") or "neutral"
        atr = float(_attr(m, "expected_move_atr") or 0.0)
        instrument = _attr(m, "instrument") or "?"
        regime = _attr(m, "regime") or "?"
        pattern_id = _attr(m, "pattern_id") or "?"

        color_hex = f"#{_TIER_COLORS.get(tier, 0x666666):06x}"
        return {
            "username": self.username,
            "attachments": [{
                "color": color_hex,
                "title": f"Tier {tier} — {pattern_id}",
                "text": f"*{direction}* · {atr:.2f} ATR expected · score {score:.3f}",
                "fields": [
                    {"title": "Instrument", "value": instrument, "short": True},
                    {"title": "Regime",     "value": regime,     "short": True},
                ],
                "ts": int((_attr(m, "match_window_end_ms") or 0) / 1000),
            }],
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _isotime(ts_ms: Any) -> str | None:
    if ts_ms is None:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).isoformat()
