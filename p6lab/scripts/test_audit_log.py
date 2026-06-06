"""
Audit log E2E smoketest.

Spins up a MatchBroker + AuditLogRenderer, fires N synthetic matches
(mixed tiers from multiple threads), then reads the resulting JSONL file
back and verifies every line is valid JSON with the expected schema.
Prints a summary and exits non-zero on any mismatch.

Run from anywhere:

    python3 scripts/test_audit_log.py
    make test-audit-log
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from p6lab.correlation.match_broker import MatchBroker
from p6lab.correlation.renderers import AuditLogRenderer

N_THREADS = 4
N_PER_THREAD = 25
EXPECTED = N_THREADS * N_PER_THREAD


class _M:
    def __init__(self, tier: str, pid: str, score: float):
        self.confidence_tier = tier
        self.tier = tier
        self.pattern_id = pid
        self.ensemble_score = score
        self.expected_direction = "bull" if tier == "A" else "bear"
        self.expected_move_atr = 1.2 if tier == "A" else 0.7
        self.template_similarity = 0.88
        self.mahalanobis_score = 0.75
        self.contextual_score = 0.70
        self.stage1_score = 0.82
        self.match_window_start_ms = int(time.time() * 1000) - 60_000
        self.match_window_end_ms = int(time.time() * 1000)
        self.regime = "normal"
        self.instrument = "NQ"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="audit_smoketest_"))
    log_path = tmp / "matches.jsonl"
    print(f"writing to: {log_path}")

    broker = MatchBroker()
    audit = AuditLogRenderer(log_path, include_run_meta=True)
    broker.subscribe(audit)

    def producer(tid: int):
        for i in range(N_PER_THREAD):
            tier = "A" if i % 3 == 0 else ("B" if i % 3 == 1 else "C")
            broker.emit(_M(tier=tier, pid=f"t{tid}-{i}", score=0.80 + 0.01 * i))

    threads = [threading.Thread(target=producer, args=(t,)) for t in range(N_THREADS)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.time() - t0

    # --- Verify file ---
    lines = log_path.read_text().splitlines()
    print(f"wrote  {len(lines)} lines in {elapsed*1000:.0f}ms ({EXPECTED} matches + 1 run_start header expected)")

    ok = True
    if len(lines) != EXPECTED + 1:
        print(f"  [FAIL] expected {EXPECTED + 1} lines, got {len(lines)}")
        ok = False

    # Header line
    try:
        header = json.loads(lines[0])
    except Exception as e:
        print(f"  [FAIL] header line is not JSON: {e}")
        return 1
    if header.get("_type") != "run_start":
        print(f"  [FAIL] first line missing _type=run_start")
        ok = False
    else:
        print(f"  [OK] header captured git_sha={header.get('git_sha','?')[:12]} "
              f"python={header.get('python','?')}")

    # Every remaining line must be valid JSON with required fields
    schema_fields = {"pattern_id", "confidence_tier", "ensemble_score"}
    for i, ln in enumerate(lines[1:], start=2):
        try:
            obj = json.loads(ln)
        except Exception as e:
            print(f"  [FAIL] line {i} not JSON: {e}")
            ok = False; continue
        missing = schema_fields - obj.keys()
        if missing:
            print(f"  [FAIL] line {i} missing fields: {missing}")
            ok = False

    # Tier distribution
    tiers = [json.loads(ln)["confidence_tier"] for ln in lines[1:]]
    from collections import Counter
    dist = Counter(tiers)
    print(f"  tier distribution: {dict(dist)}")

    # Stats: AuditLogRenderer.lines_written should equal EXPECTED
    if audit.lines_written != EXPECTED:
        print(f"  [FAIL] audit.lines_written={audit.lines_written}, expected {EXPECTED}")
        ok = False
    else:
        print(f"  [OK] audit.lines_written={audit.lines_written}")

    if ok:
        print("\nOK — audit log smoketest PASSED")
        return 0
    print("\nFAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
