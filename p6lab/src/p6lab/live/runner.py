"""
LiveRunner — orchestrate a live-trading engine session end-to-end.

Wires together everything Waves 1 & 2 delivered:

    DatabentoLiveFeed ── feed.next() ───► FeatureAccumulator
                                            │
                                            ▼
                                     engine.match()
                                            │
                                            ▼
                                     MatchBroker.emit()
                                            │
        ┌───────────────┬──────────────────┼─────────────────┐
        ▼               ▼                  ▼                 ▼
    Audit log      Metrics       Discord / Slack       Chart overlay
    (JSONL)        (Prometheus)   (webhooks)            (WebSocket)

Configuration is env-driven — the same ``.env`` that controls local
renderers in Wave 2 #3 drives live runs too.

Usage (operator CLI — from p6lab/scripts/run_live.py):
    python -m p6lab.live.runner --symbol NQ --duration 600

Usage (in-process):
    runner = LiveRunner.from_env(symbol="NQ", dataset="GLBX.MDP3")
    await runner.run(duration_seconds=60)
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from p6lab._logging import configure_logging, with_context
from p6lab.correlation.match_broker import MatchBroker
from p6lab.correlation.renderers import (
    AuditLogRenderer, MetricsRenderer, WebhookRenderer,
)
from p6lab.features.l1_features import L1FeatureNames
from p6lab.live.activity_detector import (
    ActivityDetectorConfig, OnlineActivityDetector,
)
from p6lab.live.feature_accumulator import FeatureAccumulator, FeatureRow
from p6lab.patterns.template_matcher import MatchContext, TemplateMatcher
from p6lab.live.tier_filter import TaggingRenderer, PercentileTierFilter, TierFilterConfig


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LiveConfig:
    """All knobs needed to bring up a live engine session."""
    symbol: str = "NQ"
    dataset: str = "GLBX.MDP3"
    tick_size: float = 0.25
    num_levels: int = 20
    snapshot_interval_ms: int = 100
    window_seconds: float = 300.0      # engine's match() input window
    match_interval_ms: int = 1_000     # how often to call engine.match()

    # Model registry (Wave 1 #4 — CURRENT.json)
    registry_path: Path | None = None

    # Renderer config (env-driven by default)
    audit_log_path: Path | None = None
    # Wave 8.5 pre-Tier-2: outcome tracker renderer. When set, resolves
    # every match to a closed outcome (entry + horizon exit + realized
    # return + hit/miss) and appends to the JSONL. Required for Stages
    # 4 + 5 of the 30-day validation pipeline.
    outcomes_log_path: Path | None = None
    outcomes_horizon_ms: int = 60_000
    outcomes_reaggregate_every_n: int = 20
    outcomes_retire_below_hit_rate: float = 0.50
    enable_metrics: bool = True
    metrics_http_port: int | None = None
    discord_webhook_url: str | None = None
    slack_webhook_url: str | None = None
    webhook_tier_filter: set[str] = field(default_factory=lambda: {"A", "B"})
    webhook_min_score: float | None = None
    webhook_max_per_minute: int = 20
    # Wave 8.5-K — rolling-percentile tier filter for outcome tracker.
    # When set, OutcomeTrackerRenderer is wrapped in TaggingRenderer so only matches passing the filter are recorded as outcomes.
    # When None, no filtering is applied and all matches are recorded as outcomes.
    tier_filter_config: TierFilterConfig | None = None
    # Wave 9 A5 — online activity detector. When set, the runner runs a
    # CUSUM detector per snapshot and passes ``is_active`` into
    # ``engine.match()``. The engine emits the model's ``predict_proba``
    # on active snapshots and a base-rate prior on inactive ones (soft
    # gate). When None, no activity gating — the engine runs the model
    # on every snapshot that has ``feature_row``.
    activity_detector_config: ActivityDetectorConfig | None = None
    # Match context defaults — can be overridden per-call once VIX/TOD come online
    default_vix_level: float = 18.0
    default_regime: str = "normal"
    default_relative_volume: float = 1.0


# ---------------------------------------------------------------------------
# LiveRunner
# ---------------------------------------------------------------------------

class LiveRunner:
    """Drive a live engine session until the configured duration expires."""

    def __init__(
        self,
        config: LiveConfig,
        *,
        feed_factory: Callable[[], Any] | None = None,
        engine_factory: Callable[[MatchBroker], Any] | None = None,
    ) -> None:
        self.config = config
        # Dependency injection — tests pass the MockLiveFeed factory;
        # operators pass None and get the real DatabentoLiveFeed.
        self._feed_factory = feed_factory or self._default_feed_factory
        self._engine_factory = engine_factory or self._default_engine_factory
        self._broker: MatchBroker | None = None
        self._accumulator: FeatureAccumulator | None = None
        self._renderer_handles: dict[str, Any] = {}
        self._shutdown = asyncio.Event()
        self.stats: dict[str, int] = {
            "snapshots_ingested": 0,
            "match_calls": 0,
            "matches_emitted": 0,
        }
        self._tier_filter: PercentileTierFilter | None = (
        PercentileTierFilter(config.tier_filter_config)
        if config.tier_filter_config is not None else None
        )
        # Wave 9 A5 — online activity detector. None when not configured;
        # the runner skips the per-snapshot update and passes is_active=None
        # to engine.match(), which preserves Wave 9-A behavior.
        self._activity_detector: OnlineActivityDetector | None = (
            OnlineActivityDetector(config.activity_detector_config)
            if config.activity_detector_config is not None else None
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, **overrides: Any) -> "LiveRunner":
        """Build a runner with config pulled from env vars + optional overrides."""
        cfg = LiveConfig(**overrides)
        cfg.discord_webhook_url = cfg.discord_webhook_url or os.environ.get("P6LAB_DISCORD_WEBHOOK_URL")
        cfg.slack_webhook_url   = cfg.slack_webhook_url   or os.environ.get("P6LAB_SLACK_WEBHOOK_URL")
        if cfg.audit_log_path is None and (p := os.environ.get("P6LAB_AUDIT_LOG")):
            cfg.audit_log_path = Path(p)
        if cfg.metrics_http_port is None and (p := os.environ.get("P6LAB_METRICS_PORT")):
            cfg.metrics_http_port = int(p)
        if cfg.registry_path is None and (p := os.environ.get("P6LAB_MODEL_REGISTRY")):
            cfg.registry_path = Path(p)
        # Wave 8.5 pre-Tier-2: outcome tracker env override
        if cfg.outcomes_log_path is None and (p := os.environ.get("P6LAB_OUTCOMES_LOG")):
            cfg.outcomes_log_path = Path(p)
        return cls(cfg)

    async def run(self, *, duration_seconds: float | None = None) -> dict:
        """Run the live loop until shutdown or duration expires.

        Returns stats dict at exit. Safe to Ctrl+C.
        """
        self._install_signal_handlers()

        self._broker = MatchBroker()
        self._attach_renderers()

        engine = self._engine_factory(self._broker)
        self._accumulator = FeatureAccumulator(
            tick_size=self.config.tick_size,
            window_seconds=self.config.window_seconds,
            num_levels=self.config.num_levels,
        )

        feed = self._feed_factory()
        await feed.connect()
        logger.info("live runner: feed connected (symbol=%s dataset=%s)",
                    self.config.symbol, self.config.dataset)

        started_at = time.monotonic()
        last_match_ts = 0.0
        # Wave 9-A: cache the most recent FeatureRow returned by the
        # accumulator so engine.match() can stamp primary_proba on each
        # PatternMatch. Held across iterations because match() runs on a
        # cadence (match_interval_ms) coarser than ingest() (every snap).
        latest_row: FeatureRow | None = None
        try:
            while not self._shutdown.is_set():
                if duration_seconds is not None and \
                   time.monotonic() - started_at >= duration_seconds:
                    logger.info("live runner: duration %.0fs elapsed — stopping",
                                duration_seconds)
                    break
                snap = await feed.next()
                if snap is None:
                    # MockLiveFeed exhausts — check if pipe task is done
                    if hasattr(feed, '_ingest_task') and feed._ingest_task is not None and feed._ingest_task.done():
                        if hasattr(feed, '_event_queue') and feed._event_queue.empty():
                            logger.info("live runner: feed exhausted — stopping")
                        break
                    await asyncio.sleep(0.01)
                    continue
                self.stats["snapshots_ingested"] += 1
                new_row = self._accumulator.ingest(snap)
                if new_row is not None:
                    latest_row = new_row

                # Wave 9 A5: drive the activity detector on every snapshot
                # (ingest cadence), independent of the coarser match cadence.
                # The mid extracted here matches the accumulator's own mid
                # convention so the detector's CUSUM tracks the same series
                # the model was trained against.
                if self._activity_detector is not None:
                    snap_mid = _snap_mid(snap)
                    if snap_mid is not None:
                        self._activity_detector.update(
                            snap_mid,
                            int(getattr(snap, "timestamp_ms", 0) or 0),
                        )

                # Wave 8.5 pre-Tier-2: drive outcome tracker's price stream
                # so pending exits resolve. Mid extracted the same way the
                # accumulator does it — best-bid + best-ask / 2.
                tracker = self._renderer_handles.get("outcome_tracker")
                if tracker is not None:
                    mid = _snap_mid(snap)
                    if mid is not None:
                        tracker.on_price(
                            self.config.symbol, mid,
                            int(getattr(snap, "timestamp_ms", 0) or 0),
                        )

                now = time.monotonic()
                if now - last_match_ts < self.config.match_interval_ms / 1000.0:
                    continue
                last_match_ts = now

                windows = self._accumulator.window()
                if windows is None:
                    continue
                l2_window, l1_window = windows
                if len(l2_window) < 5:        # warmup — need a handful of rows
                    continue

                context = MatchContext(
                    time_of_day_minutes=self._current_tod_minutes(),
                    vix_level=self.config.default_vix_level,
                    vix_regime=self.config.default_regime,
                    relative_volume=self.config.default_relative_volume,
                    instrument=self.config.symbol,
                )
                # Wave 9-A: build a name → value dict for the primary
                # LightGBM model from the latest accumulated FeatureRow.
                # ``engine._predict_primary_proba`` aligns it to the model's
                # expected ``feature_names``; if a column is missing or the
                # model is unloaded, the engine quietly degrades to None
                # and matches go out matcher-only.
                feature_row_dict: dict[str, float] | None = None
                if latest_row is not None:
                    feature_row_dict = {
                        **{
                            name: float(latest_row.l1[i])
                            for i, name in enumerate(L1FeatureNames.ALL)
                            if i < latest_row.l1.shape[0]
                        },
                        **{
                            name: float(v)
                            for name, v in latest_row.l2_scalars.items()
                        },
                        "fi_fast": float(latest_row.fi_fast),
                        "fi_full": float(latest_row.fi_full),
                    }

                # Wave 9 A5: query the activity detector for the current
                # snapshot's mask state. None when no detector configured —
                # engine then preserves Wave 9-A behavior on the proba.
                is_active: bool | None = None
                if self._activity_detector is not None and latest_row is not None:
                    is_active = self._activity_detector.is_active_at(
                        int(latest_row.timestamp_ms),
                    )

                with with_context(
                    instrument=self.config.symbol,
                    symbol=self.config.symbol,
                ):
                    self.stats["match_calls"] += 1
                    matches = engine.match(
                        l2_window=l2_window,
                        l1_window=l1_window,
                        context=context,
                        feature_row=feature_row_dict,
                        is_active=is_active,
                    )
                    self.stats["matches_emitted"] += len(matches)
        finally:
            try:
                await feed.disconnect()
            except Exception:
                logger.warning("feed disconnect raised; ignoring")
            logger.info("live runner: stats=%s", self.stats)

        return dict(self.stats)

    # ------------------------------------------------------------------
    # Defaults — real implementations (operators)
    # ------------------------------------------------------------------

    def _default_feed_factory(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
        from p6.ingestion.databento_feed import DatabentoLiveFeed
        return DatabentoLiveFeed(
            symbol=self.config.symbol,
            dataset=self.config.dataset,
            snapshot_interval_ms=self.config.snapshot_interval_ms,
            num_levels=self.config.num_levels,
        )

    def _default_engine_factory(self, broker: MatchBroker):
        from p6lab.correlation.engine import CorrelationEngine
        from p6lab.correlation.scorer import EnsembleScorer
        from p6lab.patterns.library import PatternLibrary

        _P6LAB = Path(__file__).resolve().parents[3]
        lib_path = _P6LAB / "artifacts" / "p6lab" / "pattern_library" / "library.yaml"
        lib = PatternLibrary(lib_path); lib.load()
        matcher = TemplateMatcher()
        scorer = EnsembleScorer()
        engine = CorrelationEngine(
            library=lib, matcher=matcher, scorer=scorer, broker=broker,
        )
        registry = self.config.registry_path or (
            _P6LAB / "correlation_runs" / "models" / "CURRENT.json"
        )
        if registry.is_file():
            engine.load_current_model(registry)
            logger.info("live runner: loaded model via registry %s", registry)
        else:
            logger.warning("live runner: no model registry at %s — engine runs unloaded", registry)
        return engine

    # ------------------------------------------------------------------
    # Renderers — uses the Wave 2 install_broker_subscribers helper
    # ------------------------------------------------------------------

    def _attach_renderers(self) -> None:
        cfg = self.config
        if cfg.audit_log_path:
            audit = AuditLogRenderer(cfg.audit_log_path, include_run_meta=True)
            self._broker.subscribe(audit)
            self._renderer_handles["audit"] = audit
        # Wave 8.5 pre-Tier-2: outcome tracker. When outcomes_log_path is
        # set, every emitted match gets recorded as a pending outcome and
        # resolved at horizon → closed outcome jsonl. Required for
        # Stage 4 (retirement validation) of the 30-day validation.
        if cfg.outcomes_log_path:
            from p6lab.correlation.renderers.outcome_tracker import (
                OutcomeTrackerRenderer,
            )
            tracker = OutcomeTrackerRenderer(
                outcomes_path=cfg.outcomes_log_path,
                horizon_ms=cfg.outcomes_horizon_ms,
                reaggregate_every_n=cfg.outcomes_reaggregate_every_n,
                retire_below_hit_rate=cfg.outcomes_retire_below_hit_rate,
                tick_size=cfg.tick_size,
            )
            # Wave 8.5-K: optional rolling-percentile tier filter
            if cfg.tier_filter_config is not None:
                tier_filter = PercentileTierFilter(cfg.tier_filter_config)
                tier_log_path = (
                    cfg.outcomes_log_path.parent
                    / f"{cfg.outcomes_log_path.stem.replace('shadow', 'tiers')}.jsonl"
                )
                # NEW: pass tier_log_path as third arg
                wrapped = TaggingRenderer(tracker, tier_filter, tier_log_path)
                self._broker.subscribe(wrapped)
                logger.info(
                    "live runner: tier filter active"
                    "(tiers=%s warmup=%d log=%s)",
                    list(cfg.tier_filter_config.tier_percentiles.keys()),
                    cfg.tier_filter_config.warmup_samples,
                    tier_log_path,
                )
            else:
                self._broker.subscribe(tracker)
            self._renderer_handles["outcome_tracker"] = tracker
        if cfg.enable_metrics:
            metrics = MetricsRenderer()
            self._broker.subscribe(metrics)
            self._renderer_handles["metrics"] = metrics
            if cfg.metrics_http_port:
                metrics.start_http_server(cfg.metrics_http_port)
        if cfg.discord_webhook_url:
            self._broker.subscribe(WebhookRenderer(
                cfg.discord_webhook_url, platform="discord",
                tier_filter=cfg.webhook_tier_filter,
                min_score=cfg.webhook_min_score,
                max_per_minute=cfg.webhook_max_per_minute,
            ))
        if cfg.slack_webhook_url:
            self._broker.subscribe(WebhookRenderer(
                cfg.slack_webhook_url, platform="slack",
                tier_filter=cfg.webhook_tier_filter,
                min_score=cfg.webhook_min_score,
                max_per_minute=cfg.webhook_max_per_minute,
            ))
        logger.info("live runner: %d renderer(s) attached", self._broker.subscriber_count)

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._request_shutdown)
        except NotImplementedError:
            # Windows + some test runners don't support add_signal_handler
            pass

    def _request_shutdown(self) -> None:
        logger.info("live runner: shutdown signal received")
        self._shutdown.set()

    @staticmethod
    def _current_tod_minutes() -> int:
        now = time.gmtime()
        return now.tm_hour * 60 + now.tm_min

    @property
    def renderer_handles(self) -> dict[str, Any]:
        """Read-only view of attached renderers (audit / metrics /
        outcome_tracker / webhook). Wave 2's test_live_runner
        referenced this as a public attribute; now it is."""
        return dict(self._renderer_handles)

    @property
    def broker(self) -> Any:
        """Read access to the MatchBroker instance. Set at run() start."""
        return self._broker


# Wave 8.5 pre-Tier-2: helper for outcome-tracker price stream. Mirrors
# the mid extraction the accumulator does so both stay in lockstep.
def _snap_mid(snap: Any) -> float | None:
    mid = getattr(snap, "mid_price", None)
    if mid is not None:
        return float(mid)
    bids = getattr(snap, "bids", None) or []
    asks = getattr(snap, "asks", None) or []
    if bids and asks:
        b = getattr(bids[0], "price", None)
        a = getattr(asks[0], "price", None)
        if b is not None and a is not None:
            return (float(b) + float(a)) / 2.0
    return None

    # ------------------------------------------------------------------
    # Introspection (for tests)
    # ------------------------------------------------------------------

    @property
    def broker(self) -> MatchBroker | None:
        return self._broker

    @property
    def renderer_handles(self) -> dict[str, Any]:
        return dict(self._renderer_handles)


# ---------------------------------------------------------------------------
# __main__ — lightweight CLI (full-featured CLI lives in scripts/run_live.py)
# ---------------------------------------------------------------------------

def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="p6lab live runner")
    ap.add_argument("--symbol",   default="NQ")
    ap.add_argument("--dataset",  default="GLBX.MDP3")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="run for N seconds then exit (default 60)")
    ap.add_argument("--json-logs", action="store_true",
                    help="emit structured JSON log lines")
    ap.add_argument("--metrics-port", type=int, default=None)
    ap.add_argument("--audit-log",   type=str, default=None)
    args = ap.parse_args()

    configure_logging(level="INFO", json=args.json_logs)

    runner = LiveRunner.from_env(
        symbol=args.symbol, dataset=args.dataset,
        metrics_http_port=args.metrics_port,
        audit_log_path=Path(args.audit_log) if args.audit_log else None,
    )
    stats = asyncio.run(runner.run(duration_seconds=args.duration))
    print(f"final stats: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
