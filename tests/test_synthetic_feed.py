"""Tests for SyntheticFeed data generator."""
from __future__ import annotations
import pytest
from p6.ingestion.synthetic import SyntheticFeed
from p6.models import OrderBookSnapshot, Side


@pytest.mark.asyncio
async def test_synthetic_feed_generates_snapshot():
    feed = SyntheticFeed(symbol="TEST", num_levels=5)
    await feed.connect()
    snap = await feed.next()
    assert isinstance(snap, OrderBookSnapshot)
    assert snap.symbol == "TEST"


@pytest.mark.asyncio
async def test_synthetic_feed_has_bids_and_asks():
    feed = SyntheticFeed(num_levels=10)
    await feed.connect()
    snap = await feed.next()
    assert len(snap.bids) == 10
    assert len(snap.asks) == 10


@pytest.mark.asyncio
async def test_synthetic_feed_bids_sorted_descending():
    feed = SyntheticFeed(num_levels=10)
    await feed.connect()
    snap = await feed.next()
    prices = [l.price for l in snap.bids]
    assert prices == sorted(prices, reverse=True)


@pytest.mark.asyncio
async def test_synthetic_feed_asks_sorted_ascending():
    feed = SyntheticFeed(num_levels=10)
    await feed.connect()
    snap = await feed.next()
    prices = [l.price for l in snap.asks]
    assert prices == sorted(prices)


@pytest.mark.asyncio
async def test_synthetic_feed_has_events():
    feed = SyntheticFeed(num_levels=5)
    await feed.connect()
    snap = await feed.next()
    assert len(snap.recent_events) > 0


@pytest.mark.asyncio
async def test_synthetic_feed_inject_wall():
    feed = SyntheticFeed(num_levels=5, base_price=100.0)
    await feed.connect()
    feed.inject_institutional_wall(Side.ASK, 102.0, 1000.0)
    snap = await feed.next()
    wall = next((l for l in snap.asks if abs(l.price - 102.0) < 0.1), None)
    assert wall is not None
    assert wall.order_count == 1
    assert wall.volume >= 1000.0


@pytest.mark.asyncio
async def test_synthetic_feed_mid_price_exists():
    feed = SyntheticFeed(num_levels=5)
    await feed.connect()
    snap = await feed.next()
    assert snap.mid_price is not None


def test_synthetic_feed_invalid_num_levels():
    with pytest.raises(ValueError):
        SyntheticFeed(num_levels=0)


def test_synthetic_feed_invalid_tick_size():
    with pytest.raises(ValueError):
        SyntheticFeed(tick_size=0)
