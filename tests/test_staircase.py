"""Tests for FragilityScorer (Layer 1)."""
from __future__ import annotations
import pytest
from p6.staircase_analyzer.fragility_scorer import FragilityScorer
from p6.models import (
    OrderBookLevel, OrderBookSnapshot, Side, FragilityState,
    Order, OrderAction,
)


def _mk_level(price, side, volume, count, orders=None):
    return OrderBookLevel(
        price=price, side=side, volume=volume, order_count=count,
        orders=orders or [],
    )


def _mk_order(oid, side, price, size, ts=1000):
    return Order(order_id=oid, side=side, price=price, size=size, timestamp_ms=ts)


def test_fragility_scorer_wall_is_fragile():
    scorer = FragilityScorer()
    wall = _mk_level(
        101.0, Side.ASK, 400.0, 2,
        orders=[_mk_order("w1", Side.ASK, 101.0, 200.0), _mk_order("w2", Side.ASK, 101.0, 200.0)],
    )
    score = scorer.score_level(wall, median_count=20)
    assert score > 0.45


def test_fragility_scorer_distributed_level_is_solid():
    scorer = FragilityScorer()
    many = _mk_level(
        100.0, Side.BID, 100.0, 50,
        orders=[_mk_order(f"o{i}", Side.BID, 100.0, 2.0) for i in range(50)],
    )
    score = scorer.score_level(many, median_count=20)
    assert score < 0.35


def test_fragility_scorer_empty_level_is_max_fragile():
    scorer = FragilityScorer()
    empty = _mk_level(100.0, Side.BID, 0.0, 0)
    assert scorer.score_level(empty, median_count=10) == 1.0


def test_classify_fragile():
    scorer = FragilityScorer()
    assert scorer.classify(0.8) == FragilityState.FRAGILE


def test_classify_solid():
    scorer = FragilityScorer()
    assert scorer.classify(0.2) == FragilityState.SOLID


def test_classify_moderate():
    scorer = FragilityScorer()
    assert scorer.classify(0.5) == FragilityState.MODERATE


def test_build_profile_has_correct_counts(sample_snapshot):
    scorer = FragilityScorer()
    profile = scorer.build_profile(sample_snapshot)
    assert len(profile.ask_levels) == 3
    assert len(profile.bid_levels) == 3


def test_build_profile_imbalance_range(sample_snapshot):
    scorer = FragilityScorer()
    profile = scorer.build_profile(sample_snapshot)
    assert -1.0 <= profile.imbalance_ratio <= 1.0


def test_build_profile_wall_level_fragility(sample_snapshot):
    scorer = FragilityScorer()
    profile = scorer.build_profile(sample_snapshot)
    wall = next(l for l in profile.ask_levels if abs(l.price - 101.0) < 1e-9)
    assert wall.fragility_score > 0.45
    assert wall.avg_order_size >= 100.0
