"""
p6lab live runner — operator entry point.

End-to-end wiring of Waves 1 & 2 deliverables for a live-trading session:

    DatabentoLiveFeed → engine → MatchBroker → renderers
                                                │
                                                ├─ audit JSONL (always-on)
                                                ├─ metrics (Prometheus)
                                                ├─ Discord / Slack webhooks
                                                └─ structured JSON logs

The real feed requires ``DATABENTO_API_KEY``. For lab rehearsal, pass
``--mock-source PATH.dbn.zst`` to swap in ``MockLiveFeed`` (plays a
committed .dbn.zst through the same queue interface the live feed uses).

Usage:

    # Real live (requires DATABENTO_API_KEY in .env)
    python scripts/run_live.py --symbol NQ --duration 3600 \\
        --audit-log artifacts/live/matches.jsonl --json-logs

    # Mock rehearsal against the committed sample
    python scripts/run_live.py --mock-source data/nq-mbo-sample-15min.dbn.zst \\
        --duration 30 --json-logs
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PROJECTS = ROOT.parent.parent
for p in (str(SRC), str(PROJECTS), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load .env files so operators don't need to `source .env` manually.
# Two paths are supported (both gitignored):
#   p6-v2/.env        → repo-root secrets (e.g. DATABENTO_API_KEY)
#   p6-v2/p6lab/.env  → lab-specific secrets (P6LAB_DISCORD_WEBHOOK_URL)
# The p6lab/.env is loaded LAST so its values win on overlap — that's the
# canonical place for p6lab-specific overrides.
for _env_path in (PROJECTS / "p6-v2" / ".env", ROOT / ".env"):
    if _env_path.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_path, override=False)
        except ImportError:
            # Fallback: minimal .env parser
            for _line in _env_path.read_text().splitlines():
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                import os as _os
                _os.environ.setdefault(_k.strip(), _v.strip())

from p6lab._logging import configure_logging                         # noqa: E402
from p6lab.live.runner import LiveConfig, LiveRunner                  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol",   default="NQ")
    ap.add_argument("--dataset",  default="GLBX.MDP3")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to run for (default: until Ctrl+C)")
    ap.add_argument("--json-logs", action="store_true",
                    help="emit structured JSON log lines (recommended for production)")
    ap.add_argument("--audit-log", type=str, default=None,
                    help="path to JSONL audit file — if missing, audit renderer is off")
    # Wave 8.5 pre-Tier-2
    ap.add_argument("--outcomes-log", type=str, default=None,
                    help="path to JSONL outcomes file (OutcomeTrackerRenderer) — "
                         "resolves each match at horizon to closed outcome. "
                         "Required for Wave 8.5 Stage 2/4 validation.")
    ap.add_argument("--outcomes-horizon-ms", type=int, default=60_000,
                    help="triple-barrier horizon in ms for outcome tracker (default: 60s)")
    ap.add_argument("--metrics-port", type=int, default=None,
                    help="expose Prometheus /metrics on this port — omit to disable")
    ap.add_argument("--mock-source", type=str, default=None,
                    help="play a .dbn.zst through MockLiveFeed instead of the real feed")
    ap.add_argument("--tier-filter", default="A,B",
                    help="comma-sep tier(s) for webhook renderers (default: A,B)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="min ensemble_score for webhook firing")
    ap.add_argument("--percentile-tier-filter", action="store_true",
                help="Apply rolling-percentile tier filter (Wave 8.5-K).")
    ap.add_argument("--tier-percentile-config", type=str, default=None,
                help='Optional JSON like \'{"A_strict":0.995,"A_relaxed":0.99}\'')
    ap.add_argument("--debug-attr-probe", action="store_true",
                help="Wave 8.5-K: log dir(match) once via TaggingRenderer "
                        "to confirm probability attribute name. Run once with this "
                        "flag, read the log, then drop it.")
    return ap.parse_args()



def _make_mock_feed_factory(source: str, symbol: str, num_levels: int):
    """Return a callable that constructs a MockLiveFeed on demand."""
    def _factory():
        # Import lazily so the real live path doesn't pull in tests/
        mock_path = ROOT / "tests" / "fixtures" / "mock_live_feed.py"
        import importlib.util as iu
        spec = iu.spec_from_file_location("mock_live_feed", mock_path)
        mod = iu.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod.MockLiveFeed(
            source_file=source, symbol=symbol,
            filter_symbol=symbol, num_levels=num_levels,
        )
    return _factory


def main() -> int:
    args = _parse_args()
    configure_logging(level="INFO", json=args.json_logs)
    log = logging.getLogger("p6lab.run_live")

    audit_path = Path(args.audit_log) if args.audit_log else None
    outcomes_path = Path(args.outcomes_log) if args.outcomes_log else None
    tier_filter = {t.strip().upper() for t in args.tier_filter.split(",") if t.strip()}  
    
    tier_filter_config = None
    if args.percentile_tier_filter:
        import json as _json
        from p6lab.live.tier_filter import TierFilterConfig
        pct_dict = (
            _json.loads(args.tier_percentile_config)
            if args.tier_percentile_config else None
        )
        kwargs = {"debug_attr_probe": args.debug_attr_probe}
        if pct_dict is not None:
            kwargs["tier_percentiles"] = pct_dict
        tier_filter_config = TierFilterConfig(**kwargs)
        
    print(f"DEBUG cfg-build: tier_filter_config={tier_filter_config!r}", flush=True)
    cfg = LiveConfig(
        symbol=args.symbol,
        dataset=args.dataset,
        audit_log_path=audit_path,
        outcomes_log_path=outcomes_path,
        outcomes_horizon_ms=args.outcomes_horizon_ms,
        metrics_http_port=args.metrics_port,
        webhook_tier_filter=tier_filter,
        webhook_min_score=args.min_score,
        tier_filter_config=tier_filter_config,

    )
    # Fill env-backed fields (Discord/Slack/model registry) via from_env semantics
    env_runner = LiveRunner.from_env(
        symbol=cfg.symbol, dataset=cfg.dataset,
        audit_log_path=cfg.audit_log_path,
        outcomes_log_path=cfg.outcomes_log_path,
        outcomes_horizon_ms=cfg.outcomes_horizon_ms,
        metrics_http_port=cfg.metrics_http_port,
        webhook_tier_filter=cfg.webhook_tier_filter,
        webhook_min_score=cfg.webhook_min_score,
        tier_filter_config=cfg.tier_filter_config,
    )
    cfg = env_runner.config   # picks up env-driven URLs

    # Swap in MockLiveFeed if --mock-source was passed (lab rehearsal path).
    feed_factory = None
    if args.mock_source:
        log.info("using MockLiveFeed against %s", args.mock_source)
        feed_factory = _make_mock_feed_factory(
            args.mock_source, cfg.symbol, cfg.num_levels,
        )

    runner = LiveRunner(cfg, feed_factory=feed_factory)
    stats = asyncio.run(runner.run(duration_seconds=args.duration))

    print("\n=== live runner done ===")
    for k, v in stats.items():
        print(f"  {k:24s}: {v}")
    if "metrics" in runner.renderer_handles:
        print(f"\n  metrics snapshot:")
        snap = runner.renderer_handles["metrics"].snapshot()
        for k, v in snap.items():
            print(f"    {k:22s}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
