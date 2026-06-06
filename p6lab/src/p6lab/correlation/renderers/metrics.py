"""
MetricsRenderer — track engine match statistics.

Two modes:

- **Prometheus mode** (default if ``prometheus_client`` is installed): exposes
  counters/histograms that any Prometheus-compatible scraper can pull.
  The renderer doesn't run its own HTTP server — call
  ``MetricsRenderer.start_http_server(port)`` once at application startup if
  you want the built-in Prometheus exporter.

- **In-memory mode** (fallback if prometheus_client is missing): keeps a tiny
  running summary in ``self.snapshot()`` — tier counts, last N scores, uptime.
  Good for tests and lightweight monitoring without a Prometheus deployment.

Wiring:

    from p6lab.correlation.renderers import MetricsRenderer
    metrics = MetricsRenderer(prefix="p6lab_correlation")
    broker.subscribe(metrics)
    metrics.start_http_server(8001)   # optional, Prometheus mode only
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server  # type: ignore
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False


class MetricsRenderer:
    """Broker subscriber that tracks match statistics.

    Parameters
    ----------
    prefix
        Metric namespace — becomes the leading segment of every Prometheus
        metric name (e.g. ``p6lab_correlation_matches_total``).
    score_window
        Size of the rolling score buffer kept in in-memory mode. Ignored in
        Prometheus mode (histograms are unbounded).
    """

    def __init__(
        self,
        prefix: str = "p6lab_correlation",
        *,
        score_window: int = 500,
    ) -> None:
        self.prefix = prefix
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._prom = _HAS_PROM
        if _HAS_PROM:
            # idempotent: reuse existing metric objects if the prefix has been
            # registered before (e.g. in tests that build N renderers)
            self._matches_total = self._reg_counter(
                f"{prefix}_matches_total",
                "Total matches emitted by the engine",
                ["tier", "instrument", "regime"],
            )
            self._score_hist = self._reg_histogram(
                f"{prefix}_ensemble_score",
                "Distribution of ensemble_score across matches",
                ["tier"],
                buckets=(0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0),
            )
            self._last_match_gauge = self._reg_gauge(
                f"{prefix}_last_match_age_seconds",
                "Seconds since the most recent match was received",
            )
            self._atr_hist = self._reg_histogram(
                f"{prefix}_expected_move_atr",
                "Distribution of expected_move_atr per match",
                ["tier", "expected_direction"],
                buckets=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
            )
        # Always-on lightweight counters (used for in-memory mode + snapshot)
        self._mem_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "other": 0}
        self._scores: deque[float] = deque(maxlen=score_window)
        self._last_match_ts: float | None = None
        self._last_match: Any = None

    # ------------------------------------------------------------------
    # Broker subscriber interface
    # ------------------------------------------------------------------

    def __call__(self, match: Any) -> None:
        tier = _attr(match, "confidence_tier") or _attr(match, "tier") or "other"
        instrument = _attr(match, "instrument") or "unknown"
        regime = _attr(match, "regime") or "unknown"
        score = float(_attr(match, "ensemble_score") or 0.0)
        direction = _attr(match, "expected_direction") or "neutral"
        atr = float(_attr(match, "expected_move_atr") or 0.0)
        now = time.time()

        with self._lock:
            if self._prom:
                self._matches_total.labels(tier=tier, instrument=instrument, regime=regime).inc()
                self._score_hist.labels(tier=tier).observe(score)
                self._atr_hist.labels(tier=tier, expected_direction=direction).observe(atr)
                self._last_match_gauge.set(0)
            self._mem_counts[tier if tier in self._mem_counts else "other"] += 1
            self._scores.append(score)
            self._last_match_ts = now
            self._last_match = match

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def start_http_server(self, port: int = 8001) -> None:
        """Start the Prometheus text-format HTTP endpoint. No-op without prometheus_client."""
        if not _HAS_PROM:
            logger.warning("prometheus_client not installed — /metrics endpoint unavailable")
            return
        start_http_server(port)
        logger.info("Prometheus metrics available on :%d/metrics", port)

    def snapshot(self) -> dict[str, Any]:
        """Lightweight dict snapshot — always works, even without prometheus_client."""
        with self._lock:
            age = None if self._last_match_ts is None else time.time() - self._last_match_ts
            mean_score = sum(self._scores) / len(self._scores) if self._scores else 0.0
            return {
                "uptime_seconds": round(time.time() - self._started_at, 1),
                "tier_counts": dict(self._mem_counts),
                "total_matches": sum(self._mem_counts.values()),
                "last_match_age_seconds": None if age is None else round(age, 2),
                "rolling_mean_score": round(mean_score, 4),
                "prometheus_enabled": self._prom,
            }

    # ------------------------------------------------------------------
    # Prometheus registration helpers — idempotent across re-instantiation
    # ------------------------------------------------------------------

    @staticmethod
    def _reg_counter(name, doc, labels):
        from prometheus_client import REGISTRY
        existing = REGISTRY._names_to_collectors.get(name)  # noqa: SLF001
        if existing: return existing
        return Counter(name, doc, labels)

    @staticmethod
    def _reg_histogram(name, doc, labels, buckets):
        from prometheus_client import REGISTRY
        existing = REGISTRY._names_to_collectors.get(name)  # noqa: SLF001
        if existing: return existing
        return Histogram(name, doc, labels, buckets=buckets)

    @staticmethod
    def _reg_gauge(name, doc):
        from prometheus_client import REGISTRY
        existing = REGISTRY._names_to_collectors.get(name)  # noqa: SLF001
        if existing: return existing
        return Gauge(name, doc)


def _attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
