"""Tests for spoof detection layer (Layer 4)."""
from __future__ import annotations
import pytest
from p6.spoof_detection.pull_before_touch import PullBeforeTouchDetector, PullBeforeTouchConfig
from p6.spoof_detection.layering_detector import LayeringDetector, LayeringConfig
from p6.spoof_detection.iceberg_inference import IcebergInference, IcebergConfig
from p6.spoof_detection.phantom_wall import PhantomWallDetector, PhantomWallConfig
from p6.spoof_detection.authenticity_scorer import AuthenticityScorer, AuthenticityConfig
from p6.models import Order, OrderAction, Side, SpoofType, SpoofEvent


def _mk(oid, side, price, size, ts, action=OrderAction.ADD):
    return Order(order_id=oid, side=side, price=price, size=size, timestamp_ms=ts, action=action)


# ── pull_before_touch ──────────────────────────────────────────────

def test_pull_before_touch_detects_repeated_cancel_near_best():
    """Requires min_repeats pulls at best to flag — supply enough events."""
    cfg = PullBeforeTouchConfig(threshold_ms=400, min_size=1.0, min_repeats=2, cooldown_ms=0)
    detector = PullBeforeTouchDetector(config=cfg)
    events = [
        _mk("s1", Side.ASK, 100.5, 50.0, 1000, OrderAction.ADD),
        _mk("s1", Side.ASK, 100.5, 50.0, 1100, OrderAction.CANCEL),
        _mk("s2", Side.ASK, 100.5, 50.0, 1200, OrderAction.ADD),
        _mk("s2", Side.ASK, 100.5, 50.0, 1300, OrderAction.CANCEL),
    ]
    result = detector.detect(events, best_bid=100.0, best_ask=100.5)
    assert len(result) == 1
    assert result[0].spoof_type == SpoofType.PULL_BEFORE_TOUCH


def test_pull_before_touch_ignores_single_cancel():
    """Single add+cancel should NOT trigger (could be normal market making)."""
    cfg = PullBeforeTouchConfig(threshold_ms=400, min_size=1.0, min_repeats=2)
    detector = PullBeforeTouchDetector(config=cfg)
    events = [
        _mk("s1", Side.ASK, 100.5, 50.0, 1000, OrderAction.ADD),
        _mk("s1", Side.ASK, 100.5, 50.0, 1150, OrderAction.CANCEL),
    ]
    result = detector.detect(events, best_bid=100.0, best_ask=100.5)
    assert len(result) == 0


def test_pull_before_touch_ignores_cancel_too_late():
    cfg = PullBeforeTouchConfig(threshold_ms=200, min_size=1.0, min_repeats=2, cooldown_ms=0)
    detector = PullBeforeTouchDetector(config=cfg)
    events = [
        _mk("s1", Side.ASK, 100.5, 50.0, 1000, OrderAction.ADD),
        _mk("s1", Side.ASK, 100.5, 50.0, 1500, OrderAction.CANCEL),
        _mk("s2", Side.ASK, 100.5, 50.0, 2000, OrderAction.ADD),
        _mk("s2", Side.ASK, 100.5, 50.0, 2500, OrderAction.CANCEL),
    ]
    result = detector.detect(events, best_bid=100.0, best_ask=100.5)
    assert len(result) == 0


def test_pull_before_touch_no_best_prices():
    detector = PullBeforeTouchDetector()
    events = [_mk("s1", Side.ASK, 100.5, 50.0, 1000, OrderAction.ADD)]
    assert detector.detect(events, best_bid=None, best_ask=None) == []


def test_pull_before_touch_ignores_small_orders():
    """Orders below min_size should be ignored."""
    cfg = PullBeforeTouchConfig(threshold_ms=400, min_size=10.0, min_repeats=2, cooldown_ms=0)
    detector = PullBeforeTouchDetector(config=cfg)
    events = [
        _mk("s1", Side.ASK, 100.5, 5.0, 1000, OrderAction.ADD),
        _mk("s1", Side.ASK, 100.5, 5.0, 1100, OrderAction.CANCEL),
        _mk("s2", Side.ASK, 100.5, 5.0, 1200, OrderAction.ADD),
        _mk("s2", Side.ASK, 100.5, 5.0, 1300, OrderAction.CANCEL),
    ]
    result = detector.detect(events, best_bid=100.0, best_ask=100.5)
    assert len(result) == 0


# ── layering detector ─────────────────────────────────────────────

def test_layering_detects_same_size_orders():
    cfg = LayeringConfig(min_levels=3, min_size=1.0, max_time_spread_ms=500)
    detector = LayeringDetector(config=cfg)
    events = [
        _mk("a1", Side.ASK, 101.0, 100.0, 1000),
        _mk("a2", Side.ASK, 101.5, 100.0, 1001),
        _mk("a3", Side.ASK, 102.0, 100.0, 1002),
        _mk("a4", Side.ASK, 102.5, 100.0, 1003),
    ]
    result = detector.detect(events)
    assert len(result) >= 1
    assert result[0].spoof_type == SpoofType.LAYERING


def test_layering_no_detection_varied_sizes():
    cfg = LayeringConfig(min_levels=3, min_size=1.0)
    detector = LayeringDetector(config=cfg)
    events = [
        _mk("a1", Side.ASK, 101.0, 10.0, 1000),
        _mk("a2", Side.ASK, 101.5, 50.0, 1001),
        _mk("a3", Side.ASK, 102.0, 200.0, 1002),
    ]
    result = detector.detect(events)
    assert len(result) == 0


def test_layering_no_detection_too_spread_in_time():
    """Same-size orders arriving far apart should not flag."""
    cfg = LayeringConfig(min_levels=3, min_size=1.0, max_time_spread_ms=100)
    detector = LayeringDetector(config=cfg)
    events = [
        _mk("a1", Side.ASK, 101.0, 100.0, 1000),
        _mk("a2", Side.ASK, 101.5, 100.0, 2000),
        _mk("a3", Side.ASK, 102.0, 100.0, 3000),
    ]
    result = detector.detect(events)
    assert len(result) == 0


# ── iceberg inference ─────────────────────────────────────────────

def test_iceberg_detects_repeated_small_fills():
    """Iceberg: many small fills with small visible clip → big hidden volume."""
    cfg = IcebergConfig(refill_count_threshold=3, max_visible_size=20.0, min_total_volume=10.0, confidence_floor=0.1)
    detector = IcebergInference(config=cfg)
    # Simulate iceberg: 8 fills of 5 lots at same price, 3 add events of 5 lots
    events = []
    for i in range(3):
        events.append(_mk(f"a{i}", Side.BID, 100.0, 5.0, 1000 + i * 100, OrderAction.ADD))
    for i in range(8):
        events.append(_mk(f"f{i}", Side.BID, 100.0, 5.0, 1050 + i * 100, OrderAction.FILL))
    result = detector.detect(events)
    assert len(result) >= 1
    assert result[0].spoof_type == SpoofType.ICEBERG


def test_iceberg_no_detection_large_visible():
    cfg = IcebergConfig(refill_count_threshold=3, max_visible_size=5.0, min_total_volume=10.0)
    detector = IcebergInference(config=cfg)
    events = [
        _mk("a1", Side.BID, 100.0, 50.0, 1000, OrderAction.ADD),
        _mk("f1", Side.BID, 100.0, 50.0, 1050, OrderAction.FILL),
        _mk("f2", Side.BID, 100.0, 50.0, 1100, OrderAction.FILL),
        _mk("f3", Side.BID, 100.0, 50.0, 1150, OrderAction.FILL),
    ]
    result = detector.detect(events)
    assert len(result) == 0


# ── phantom wall ──────────────────────────────────────────────────

def test_phantom_wall_detects_cancel_on_approach():
    cfg = PhantomWallConfig(large_size_threshold=50.0, approach_ticks=2.0, cancel_ms=500, min_wall_duration_ms=50)
    detector = PhantomWallDetector(config=cfg)
    events = [
        _mk("w1", Side.ASK, 101.0, 200.0, 1000, OrderAction.ADD),
        _mk("w1", Side.ASK, 101.0, 200.0, 1200, OrderAction.CANCEL),
    ]
    result = detector.detect(events, mid_price=100.0)
    assert len(result) == 1
    assert result[0].spoof_type == SpoofType.PHANTOM_WALL


def test_phantom_wall_ignores_small_orders():
    cfg = PhantomWallConfig(large_size_threshold=50.0)
    detector = PhantomWallDetector(config=cfg)
    events = [
        _mk("w1", Side.ASK, 101.0, 10.0, 1000, OrderAction.ADD),
        _mk("w1", Side.ASK, 101.0, 10.0, 1200, OrderAction.CANCEL),
    ]
    result = detector.detect(events, mid_price=100.0)
    assert len(result) == 0


def test_phantom_wall_ignores_far_from_mid():
    cfg = PhantomWallConfig(large_size_threshold=50.0, approach_ticks=2.0)
    detector = PhantomWallDetector(config=cfg)
    events = [
        _mk("w1", Side.ASK, 110.0, 200.0, 1000, OrderAction.ADD),
        _mk("w1", Side.ASK, 110.0, 200.0, 1200, OrderAction.CANCEL),
    ]
    result = detector.detect(events, mid_price=100.0)
    assert len(result) == 0


# ── authenticity scorer ───────────────────────────────────────────

def test_authenticity_scorer_clean_book():
    scorer = AuthenticityScorer()
    profile = scorer.score([], timestamp_ms=1000)
    assert profile.authenticity_score == 1.0


def test_authenticity_scorer_reduces_on_pull():
    scorer = AuthenticityScorer()
    events = [
        SpoofEvent(
            spoof_type=SpoofType.PULL_BEFORE_TOUCH,
            price=100.5, side=Side.ASK,
            confidence=0.8, timestamp_ms=1000,
        )
    ]
    profile = scorer.score(events, timestamp_ms=1000)
    assert profile.authenticity_score < 1.0
    assert profile.pull_score == 0.8


def test_authenticity_scorer_score_range():
    scorer = AuthenticityScorer()
    events = [
        SpoofEvent(spoof_type=SpoofType.PULL_BEFORE_TOUCH, price=100.5, side=Side.ASK, confidence=1.0, timestamp_ms=1000),
        SpoofEvent(spoof_type=SpoofType.LAYERING, price=101.0, side=Side.ASK, confidence=1.0, timestamp_ms=1000),
        SpoofEvent(spoof_type=SpoofType.PHANTOM_WALL, price=102.0, side=Side.ASK, confidence=1.0, timestamp_ms=1000),
    ]
    profile = scorer.score(events, timestamp_ms=1000)
    assert 0.0 <= profile.authenticity_score <= 1.0


def test_authenticity_scorer_floor():
    """Score should never drop below the configured floor."""
    cfg = AuthenticityConfig(floor=0.15)
    scorer = AuthenticityScorer(config=cfg)
    events = [
        SpoofEvent(spoof_type=SpoofType.PULL_BEFORE_TOUCH, price=100.5, side=Side.ASK, confidence=1.0, timestamp_ms=1000),
        SpoofEvent(spoof_type=SpoofType.LAYERING, price=101.0, side=Side.ASK, confidence=1.0, timestamp_ms=1000),
        SpoofEvent(spoof_type=SpoofType.PHANTOM_WALL, price=102.0, side=Side.ASK, confidence=1.0, timestamp_ms=1000),
    ]
    profile = scorer.score(events, timestamp_ms=1000)
    assert profile.authenticity_score >= cfg.floor
