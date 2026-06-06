"""
Unit tests for the broker renderers (audit log, metrics, webhook).

Webhook HTTP is stubbed via ``unittest.mock`` so tests don't require network.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p6lab.correlation.match_broker import MatchBroker
from p6lab.correlation.renderers import (
    AuditLogRenderer, MetricsRenderer, WebhookRenderer,
)


@dataclass
class _Match:
    """Minimal PatternMatch-shaped dataclass for testing."""
    pattern_id: str = "test_pattern"
    ensemble_score: float = 0.91
    confidence_tier: str = "A"
    expected_direction: str = "bull"
    expected_move_atr: float = 1.25
    template_similarity: float = 0.88
    mahalanobis_score: float = 0.75
    contextual_score: float = 0.70
    stage1_score: float = 0.82
    match_window_start_ms: int = 1_700_000_000_000
    match_window_end_ms: int = 1_700_000_060_000
    regime: str = "normal"
    instrument: str = "NQ"


# ---------------------------------------------------------------------------
# AuditLogRenderer
# ---------------------------------------------------------------------------

def test_audit_log_appends_jsonl(tmp_path: Path):
    path = tmp_path / "matches.jsonl"
    audit = AuditLogRenderer(path)
    audit(_Match())
    audit(_Match(pattern_id="p2", confidence_tier="B"))
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["pattern_id"] == "test_pattern"
    assert first["confidence_tier"] == "A"


def test_audit_log_header_on_first_write(tmp_path: Path):
    path = tmp_path / "matches.jsonl"
    audit = AuditLogRenderer(path, include_run_meta=True)
    audit(_Match())
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    header = json.loads(lines[0])
    assert header["_type"] == "run_start"
    assert "python" in header
    # second line is the actual match
    assert json.loads(lines[1])["pattern_id"] == "test_pattern"


def test_audit_log_thread_safe_via_broker(tmp_path: Path):
    """Run matches through the broker — no interleaved / malformed lines."""
    path = tmp_path / "m.jsonl"
    audit = AuditLogRenderer(path)
    bus = MatchBroker()
    bus.subscribe(audit)

    import threading
    def fire():
        for i in range(20):
            bus.emit(_Match(pattern_id=f"p{i}"))
    threads = [threading.Thread(target=fire) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 100, f"expected 100 lines, got {len(lines)}"
    # Every line must be valid JSON
    for ln in lines:
        json.loads(ln)


# ---------------------------------------------------------------------------
# MetricsRenderer
# ---------------------------------------------------------------------------

def test_metrics_snapshot_counts_by_tier():
    metrics = MetricsRenderer()
    for tier in ["A", "A", "B", "C", "C", "C"]:
        metrics(_Match(confidence_tier=tier))
    snap = metrics.snapshot()
    assert snap["tier_counts"] == {"A": 2, "B": 1, "C": 3, "other": 0}
    assert snap["total_matches"] == 6


def test_metrics_rolling_score():
    metrics = MetricsRenderer(score_window=3)
    for s in [0.50, 0.70, 0.90, 0.80]:
        metrics(_Match(ensemble_score=s))
    # Window holds last 3 scores: 0.70, 0.90, 0.80 → mean 0.80
    snap = metrics.snapshot()
    assert snap["rolling_mean_score"] == pytest.approx(0.80, abs=1e-3)


def test_metrics_last_match_age():
    metrics = MetricsRenderer()
    assert metrics.snapshot()["last_match_age_seconds"] is None
    metrics(_Match())
    age = metrics.snapshot()["last_match_age_seconds"]
    assert 0 <= age <= 1


def test_metrics_prometheus_enabled_flag():
    """When prometheus_client is installed, the flag in snapshot() is True."""
    metrics = MetricsRenderer(prefix=f"test_p_{id(object())}")
    metrics(_Match())
    snap = metrics.snapshot()
    # Either True (installed) or False (fallback) — both valid, but must be boolean
    assert isinstance(snap["prometheus_enabled"], bool)


# ---------------------------------------------------------------------------
# WebhookRenderer
# ---------------------------------------------------------------------------

def test_webhook_rejects_bad_url():
    with pytest.raises(ValueError):
        WebhookRenderer("not a url", platform="discord")
    with pytest.raises(ValueError):
        WebhookRenderer("https://example.com", platform="telegram")


def test_webhook_tier_filter_blocks_low_tier():
    w = WebhookRenderer("https://example.com/x", platform="discord",
                        tier_filter={"A"})
    with patch.object(w, "_post") as mock_post:
        w(_Match(confidence_tier="C"))
        time.sleep(0.05)
        mock_post.assert_not_called()


def test_webhook_tier_a_sent():
    w = WebhookRenderer("https://example.com/x", platform="discord",
                        tier_filter={"A"})
    with patch.object(w, "_post") as mock_post:
        w(_Match(confidence_tier="A"))
        # wait for background thread
        time.sleep(0.1)
        mock_post.assert_called_once()


def test_webhook_default_tier_includes_A_and_B():
    """After the Tier-B rollout, the default filter covers A and B."""
    w = WebhookRenderer("https://example.com/x", platform="discord")
    assert w.tier_filter == {"A", "B"}


def test_webhook_min_score_blocks_low_confidence_tier_b():
    """min_score filters *within* the allowed tier set."""
    w = WebhookRenderer("https://example.com/x", platform="discord",
                        tier_filter={"A", "B"}, min_score=0.80)
    with patch.object(w, "_post") as mock_post:
        # Tier B but below 0.80 → dropped
        w(_Match(confidence_tier="B", ensemble_score=0.74))
        time.sleep(0.05)
        mock_post.assert_not_called()
        # Tier B above 0.80 → sent
        w(_Match(confidence_tier="B", ensemble_score=0.85))
        time.sleep(0.1)
        mock_post.assert_called_once()


def test_webhook_min_score_out_of_range_rejected():
    with pytest.raises(ValueError):
        WebhookRenderer("https://example.com/x", platform="discord", min_score=1.5)
    with pytest.raises(ValueError):
        WebhookRenderer("https://example.com/x", platform="discord", min_score=-0.1)


def test_webhook_rate_limit():
    w = WebhookRenderer("https://example.com/x", platform="discord",
                        tier_filter={"A"}, max_per_minute=2)
    with patch.object(w, "_post") as mock_post:
        for _ in range(5):
            w(_Match(confidence_tier="A"))
        time.sleep(0.1)
        # Exactly 2 posts allowed; 3 dropped
        assert mock_post.call_count == 2
        assert w.posts_dropped == 3


def test_webhook_discord_payload_shape():
    w = WebhookRenderer("https://example.com/x", platform="discord")
    payload = w._discord_payload(_Match())
    assert payload["username"] == "p6lab"
    assert len(payload["embeds"]) == 1
    emb = payload["embeds"][0]
    assert "Tier A" in emb["title"]
    assert emb["color"] == 0x4CAF50   # green
    assert any(f["name"] == "Score" for f in emb["fields"])


def test_webhook_slack_payload_shape():
    w = WebhookRenderer("https://example.com/x", platform="slack")
    payload = w._slack_payload(_Match())
    assert payload["username"] == "p6lab"
    assert len(payload["attachments"]) == 1
    att = payload["attachments"][0]
    assert att["color"] == "#4caf50"
    assert "Tier A" in att["title"]


def test_webhook_via_broker():
    """End-to-end: match through MatchBroker → WebhookRenderer → _post."""
    w = WebhookRenderer("https://example.com/x", platform="discord")
    bus = MatchBroker()
    bus.subscribe(w)
    with patch.object(w, "_post") as mock_post:
        bus.emit(_Match(confidence_tier="A"))
        time.sleep(0.1)
        mock_post.assert_called_once()
