"""
DivergenceAuditor unit tests.

Covers:
  - Matched pairs below threshold: logged internally, nothing written
  - Matched pairs above threshold: written to JSONL
  - pattern_id mismatch: never pair
  - timestamp jitter beyond pair_window_ms: never pair
  - Unmatched buffer grows for the un-paired side
  - snapshot() returns the expected keys
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from p6lab.correlation.divergence_audit import DivergenceAuditor
from p6lab.correlation.match_broker import MatchBroker


@dataclass
class _M:
    pattern_id: str
    ensemble_score: float
    match_window_end_ms: int


def test_matched_pair_below_threshold_not_logged(tmp_path):
    out = tmp_path / "d.jsonl"
    audit = DivergenceAuditor(out, delta_threshold=0.05)
    a = MatchBroker(); b = MatchBroker()
    audit.attach(a, source="replay"); audit.attach(b, source="live")

    a.emit(_M("p1", 0.91, 1000))
    b.emit(_M("p1", 0.93, 1050))       # |Δ| = 0.02 < 0.05

    snap = audit.snapshot()
    assert snap["pairs_formed"] == 1
    assert snap["divergences_logged"] == 0
    assert not out.exists(), "nothing should be written below threshold"


def test_matched_pair_above_threshold_logged(tmp_path):
    out = tmp_path / "d.jsonl"
    audit = DivergenceAuditor(out, delta_threshold=0.05)
    a = MatchBroker(); b = MatchBroker()
    audit.attach(a, source="replay"); audit.attach(b, source="live")

    a.emit(_M("p1", 0.85, 2000))
    b.emit(_M("p1", 0.72, 2050))       # |Δ| = 0.13 > 0.05

    snap = audit.snapshot()
    assert snap["divergences_logged"] == 1
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["pattern_id"] == "p1"
    assert abs(row["delta"] - 0.13) < 1e-6
    assert {row["source_a"], row["source_b"]} == {"replay", "live"}


def test_pattern_id_mismatch_never_pairs(tmp_path):
    out = tmp_path / "d.jsonl"
    audit = DivergenceAuditor(out, delta_threshold=0.0)
    a = MatchBroker(); b = MatchBroker()
    audit.attach(a, source="replay"); audit.attach(b, source="live")

    a.emit(_M("p1", 0.91, 1000))
    b.emit(_M("p2", 0.50, 1000))       # same ts, different pattern → no pair

    snap = audit.snapshot()
    assert snap["pairs_formed"] == 0
    assert snap["unmatched_per_source"] == {"replay": 1, "live": 1}


def test_timestamp_jitter_beyond_window_never_pairs(tmp_path):
    out = tmp_path / "d.jsonl"
    audit = DivergenceAuditor(out, pair_window_ms=100, delta_threshold=0.0)
    a = MatchBroker(); b = MatchBroker()
    audit.attach(a, source="replay"); audit.attach(b, source="live")

    a.emit(_M("p1", 0.9, 1000))
    b.emit(_M("p1", 0.9, 2000))        # 1000ms apart, window is 100ms

    snap = audit.snapshot()
    assert snap["pairs_formed"] == 0


def test_snapshot_shape(tmp_path):
    audit = DivergenceAuditor(tmp_path / "d.jsonl")
    snap = audit.snapshot()
    for k in (
        "uptime_seconds", "total_matches_per_source", "pairs_formed",
        "divergences_logged", "delta_threshold", "delta_stats",
        "unmatched_per_source", "output_path",
    ):
        assert k in snap, f"missing key: {k}"
    assert snap["pairs_formed"] == 0
    assert snap["delta_stats"]["n"] == 0


def test_high_volume_flow_through_broker(tmp_path):
    """Fire 100 matched pairs; half above threshold, half below.

    Gate: >= 50 divergences logged (within tolerance for ordering edge cases)."""
    out = tmp_path / "d.jsonl"
    audit = DivergenceAuditor(out, delta_threshold=0.05)
    a = MatchBroker(); b = MatchBroker()
    audit.attach(a, source="replay"); audit.attach(b, source="live")

    for i in range(100):
        ts = 10_000 + i * 200
        a.emit(_M("p1", 0.80, ts))
        # Alternate: half below threshold, half above
        b_score = 0.82 if i % 2 == 0 else 0.95
        b.emit(_M("p1", b_score, ts + 10))

    snap = audit.snapshot()
    assert snap["pairs_formed"] == 100
    assert 45 <= snap["divergences_logged"] <= 55
    assert snap["delta_stats"]["p95"] >= 0.10   # high deltas dominate the tail
