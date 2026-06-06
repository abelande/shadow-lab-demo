"""Tests for core dataclasses and their properties."""
from __future__ import annotations
import pytest
from p6.models import (
    Order, OrderAction, Side, OrderBookLevel, OrderBookSnapshot,
    FragilityState, LevelProfile, StaircaseProfile,
    CupFlipState, GameState, ForceVector, FrequencyBand, BandEnergy,
    SpoofEvent, SpoofType, AuthenticityProfile,
    RegimeType, RegimeWeights, AggregatedSignal,
)


def test_order_creation():
    o = Order(order_id="x1", side=Side.BID, price=100.0, size=5.0, timestamp_ms=1000)
    assert o.order_id == "x1"
    assert o.side == Side.BID
    assert o.action == OrderAction.ADD
    assert o.is_aggressive is False


def test_order_book_level_avg_order_size():
    level = OrderBookLevel(price=100.0, side=Side.BID, volume=60.0, order_count=3)
    assert level.avg_order_size == 20.0


def test_order_book_level_avg_order_size_zero_count():
    level = OrderBookLevel(price=100.0, side=Side.BID, volume=60.0, order_count=0)
    assert level.avg_order_size == 0.0


def test_snapshot_best_bid_ask(sample_snapshot):
    assert sample_snapshot.best_bid == 100.0
    assert sample_snapshot.best_ask == 100.5


def test_snapshot_mid_price(sample_snapshot):
    mid = sample_snapshot.mid_price
    assert mid == pytest.approx(100.25)


def test_snapshot_spread(sample_snapshot):
    spread = sample_snapshot.spread
    assert spread == pytest.approx(0.5)


def test_snapshot_empty_bids_asks():
    snap = OrderBookSnapshot(timestamp_ms=0, symbol="X")
    assert snap.best_bid is None
    assert snap.best_ask is None
    assert snap.mid_price is None
    assert snap.spread is None


def test_side_enum_values():
    assert Side.BID.value == "BID"
    assert Side.ASK.value == "ASK"


def test_order_action_enum_values():
    assert OrderAction.ADD.value == "ADD"
    assert OrderAction.CANCEL.value == "CANCEL"
    assert OrderAction.FILL.value == "FILL"
    assert OrderAction.MODIFY.value == "MODIFY"


def test_cup_flip_state_enum():
    assert CupFlipState.BULL_STREAK.value == "BULL_STREAK"
    assert CupFlipState.BEAR_STREAK.value == "BEAR_STREAK"
    assert CupFlipState.BALANCED.value == "BALANCED"


def test_regime_type_enum():
    assert RegimeType.TRENDING.value == "TRENDING"
    assert RegimeType.RANGING.value == "RANGING"
    assert RegimeType.VOLATILE.value == "VOLATILE"
    assert RegimeType.UNKNOWN.value == "UNKNOWN"


def test_aggregated_signal_defaults():
    sig = AggregatedSignal(direction=0.5, confidence=0.7, urgency=0.3, size_multiplier=1.2)
    assert sig.regime == RegimeType.UNKNOWN
    assert sig.abstain is False
    assert sig.components == {}
    assert sig.timestamp_ms == 0
