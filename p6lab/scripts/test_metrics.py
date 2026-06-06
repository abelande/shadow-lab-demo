"""
Metrics E2E smoketest.

Spins up a MetricsRenderer, fires synthetic matches, inspects the
in-memory snapshot, and — if prometheus_client is installed — starts
the Prometheus HTTP endpoint on a free port and scrapes it with
urllib to confirm the text-format exposition works.

Run:
    python3 scripts/test_metrics.py
    make test-metrics
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from p6lab.correlation.match_broker import MatchBroker
from p6lab.correlation.renderers import MetricsRenderer


class _M:
    def __init__(self, tier: str, score: float, instrument="NQ", direction="bull", atr=1.0):
        self.confidence_tier = tier
        self.tier = tier
        self.pattern_id = f"p_{tier}_{score:.2f}"
        self.ensemble_score = score
        self.expected_direction = direction
        self.expected_move_atr = atr
        self.regime = "normal"
        self.instrument = instrument


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]; s.close()
    return port


def main() -> int:
    broker = MatchBroker()
    metrics = MetricsRenderer(prefix=f"smoketest_metrics_{int(time.time())}")
    broker.subscribe(metrics)

    # Fire a representative distribution
    fixtures = [
        ("A", 0.92, "NQ", "bull", 1.5),
        ("A", 0.88, "NQ", "bull", 1.1),
        ("B", 0.78, "NQ", "bear", 0.8),
        ("B", 0.74, "ES", "bull", 0.9),
        ("C", 0.65, "NQ", "neutral", 0.3),
        ("C", 0.62, "CL", "bear", 0.5),
        ("C", 0.68, "NQ", "bull", 0.7),
    ]
    print(f"emitting {len(fixtures)} matches...")
    for tier, score, inst, dir_, atr in fixtures:
        broker.emit(_M(tier, score, inst, dir_, atr))

    # --- Snapshot check (always works) ---
    snap = metrics.snapshot()
    print(f"snapshot: {snap}")

    ok = True
    expected_counts = {"A": 2, "B": 2, "C": 3, "other": 0}
    if snap["tier_counts"] != expected_counts:
        print(f"  [FAIL] tier_counts={snap['tier_counts']}, expected {expected_counts}")
        ok = False
    else:
        print(f"  [OK] tier_counts correct")

    if snap["total_matches"] != len(fixtures):
        print(f"  [FAIL] total_matches={snap['total_matches']}, expected {len(fixtures)}")
        ok = False
    else:
        print(f"  [OK] total_matches={snap['total_matches']}")

    if not 0.7 <= snap["rolling_mean_score"] <= 0.85:
        print(f"  [FAIL] rolling_mean_score={snap['rolling_mean_score']} out of plausible range")
        ok = False
    else:
        print(f"  [OK] rolling_mean_score={snap['rolling_mean_score']}")

    # --- Prometheus HTTP scrape (if the backend is available) ---
    if snap["prometheus_enabled"]:
        port = _free_port()
        metrics.start_http_server(port)
        time.sleep(0.2)
        print(f"\nscraping http://127.0.0.1:{port}/metrics ...")
        try:
            with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as resp:
                body = resp.read().decode()
        except Exception as e:
            print(f"  [FAIL] scrape error: {e}")
            return 1

        lines = [ln for ln in body.splitlines() if ln and not ln.startswith("#")]
        # Find our counter lines
        prefix = metrics.prefix
        counter_lines = [ln for ln in lines if ln.startswith(f"{prefix}_matches_total")]
        histo_lines  = [ln for ln in lines if ln.startswith(f"{prefix}_ensemble_score")]

        print(f"  {len(counter_lines)} counter lines, {len(histo_lines)} histogram lines")
        if counter_lines:
            print(f"  sample counter: {counter_lines[0]}")
        if histo_lines:
            print(f"  sample histo:   {histo_lines[0]}")

        if not counter_lines:
            print("  [FAIL] no matches_total counter in /metrics")
            ok = False
        else:
            print("  [OK] counter visible over HTTP")
    else:
        print("\nprometheus_client not installed — /metrics endpoint skipped.")

    print()
    if ok:
        print("OK — metrics smoketest PASSED")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
