"""Synthetic order book data generator for testing and development."""
from __future__ import annotations

import math
import random
import time
from typing import List, Optional, Tuple

from ..models import (
    Order, OrderAction, OrderBookLevel, OrderBookSnapshot, Side,
)
from .base_feed import BaseFeed


class SyntheticFeed(BaseFeed):
    """Generates realistic order book snapshots with configurable parameters.

    Features:
        - Random-walk mid price with configurable volatility
        - Realistic volume distributions (heavier near mid, thinner far)
        - Occasional institutional orders (low prob, high volume, 1-2 order_count)
        - Trade tape with random aggressive fills
        - Order event stream (ADD/CANCEL/MODIFY/FILL)
        - Pattern injection: walls, spoofs, momentum runs, stop runs
    """

    def __init__(
        self,
        symbol: str = "SYN/USD",
        num_levels: int = 20,
        tick_size: float = 0.01,
        base_price: float = 100.0,
        volatility: float = 0.001,
    ):
        super().__init__(symbol=symbol, data_level="L3")
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        if tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        if base_price <= 0:
            raise ValueError("base_price must be > 0")

        self.num_levels = num_levels
        self.tick_size = tick_size
        self.volatility = volatility

        self._mid_price = base_price
        self._tick = 0
        self._order_id_counter = 0

        # Pending injections for the next tick
        self._injected_walls: List[Tuple[Side, float, float]] = []
        self._injected_spoofs: List[Tuple[Side, float, float]] = []
        self._pending_spoof_cancels: List[Tuple[Side, float, float]] = []
        self._momentum_runs: List[Tuple[Side, int]] = []
        self._stop_runs: List[Side] = []

    # ── helpers ──────────────────────────────────────────────────

    def _next_oid(self) -> str:
        self._order_id_counter += 1
        return f"SYN-{self._order_id_counter:08d}"

    def _now_ms(self) -> int:
        return int(time.time() * 1000) + self._tick

    def _round_price(self, p: float) -> float:
        return round(round(p / self.tick_size) * self.tick_size, 10)

    # ── volume distribution ──────────────────────────────────────

    def _level_volume(self, distance_from_mid: int) -> Tuple[float, int]:
        """Generate volume & order_count for a level at given distance.

        Near the mid: higher volume, more orders.
        Far from mid: lower volume, fewer orders.
        Occasional institutional: low order_count, high volume.
        """
        base_vol = max(0.5, 50.0 * math.exp(-0.12 * distance_from_mid))
        noise = random.uniform(0.5, 1.5)
        volume = round(base_vol * noise, 4)
        order_count = max(1, int(random.gauss(8 - distance_from_mid * 0.3, 2)))

        # 3% chance of institutional order
        if random.random() < 0.03:
            volume = round(volume * random.uniform(5, 20), 4)
            order_count = random.randint(1, 2)

        return volume, order_count

    def _build_orders(
        self, side: Side, price: float, volume: float, order_count: int, ts: int
    ) -> List[Order]:
        """Build individual Order objects for a level."""
        orders: List[Order] = []
        remaining = volume
        for i in range(order_count):
            if i == order_count - 1:
                size = remaining
            else:
                size = round(remaining * random.uniform(0.1, 0.6), 4)
                remaining -= size
            orders.append(Order(
                order_id=self._next_oid(),
                side=side,
                price=price,
                size=round(max(size, 0.0001), 4),
                timestamp_ms=ts,
                action=OrderAction.ADD,
            ))
        return orders

    # ── snapshot generation ──────────────────────────────────────

    async def next(self) -> Optional[OrderBookSnapshot]:
        """Generate the next synthetic order book snapshot."""
        if not self._connected:
            await self.connect()

        self._tick += 1
        ts = self._now_ms()

        # Random walk mid price
        self._mid_price *= (1 + random.gauss(0, self.volatility))
        mid = self._mid_price

        bids: List[OrderBookLevel] = []
        asks: List[OrderBookLevel] = []
        events: List[Order] = []

        # Build bid levels (descending price)
        for i in range(self.num_levels):
            price = self._round_price(mid - (i + 1) * self.tick_size)
            vol, cnt = self._level_volume(i)
            orders = self._build_orders(Side.BID, price, vol, cnt, ts)
            events.extend(orders)
            bids.append(OrderBookLevel(
                price=price, side=Side.BID, volume=vol,
                order_count=cnt, orders=orders,
            ))

        # Build ask levels (ascending price)
        for i in range(self.num_levels):
            price = self._round_price(mid + (i + 1) * self.tick_size)
            vol, cnt = self._level_volume(i)
            orders = self._build_orders(Side.ASK, price, vol, cnt, ts)
            events.extend(orders)
            asks.append(OrderBookLevel(
                price=price, side=Side.ASK, volume=vol,
                order_count=cnt, orders=orders,
            ))

        # Apply injected walls
        for side, price, volume in self._injected_walls:
            price = self._round_price(price)
            orders = self._build_orders(side, price, volume, 1, ts)
            level = OrderBookLevel(
                price=price, side=side, volume=volume,
                order_count=1, orders=orders,
            )
            events.extend(orders)
            if side == Side.BID:
                bids.append(level)
                bids.sort(key=lambda l: l.price, reverse=True)
            else:
                asks.append(level)
                asks.sort(key=lambda l: l.price)
        self._injected_walls.clear()

        # Apply injected spoofs (will be canceled next tick)
        for side, price, volume in self._injected_spoofs:
            price = self._round_price(price)
            orders = self._build_orders(side, price, volume, 1, ts)
            level = OrderBookLevel(
                price=price, side=side, volume=volume,
                order_count=1, orders=orders,
            )
            events.extend(orders)
            if side == Side.BID:
                bids.append(level)
                bids.sort(key=lambda l: l.price, reverse=True)
            else:
                asks.append(level)
                asks.sort(key=lambda l: l.price)
        self._pending_spoof_cancels.extend(self._injected_spoofs)
        self._injected_spoofs.clear()

        # Process spoof cancels from previous tick
        for side, price, volume in self._pending_spoof_cancels:
            events.append(Order(
                order_id=self._next_oid(),
                side=side,
                price=self._round_price(price),
                size=volume,
                timestamp_ms=ts,
                action=OrderAction.CANCEL,
            ))
        self._pending_spoof_cancels.clear()

        # Generate trades (random aggressive fills)
        trades: List[Order] = []
        num_trades = random.randint(0, 5)
        for _ in range(num_trades):
            trade_side = random.choice([Side.BID, Side.ASK])
            if trade_side == Side.BID and asks:
                trade_price = asks[0].price
            elif trade_side == Side.ASK and bids:
                trade_price = bids[0].price
            else:
                continue
            trade_size = round(random.uniform(0.01, 5.0), 4)
            trade = Order(
                order_id=self._next_oid(),
                side=trade_side,
                price=trade_price,
                size=trade_size,
                timestamp_ms=ts,
                action=OrderAction.FILL,
                is_aggressive=True,
            )
            trades.append(trade)
            events.append(trade)

        # Apply momentum runs
        for side, num_levels_run in self._momentum_runs:
            for j in range(num_levels_run):
                if side == Side.BID and asks:
                    fill_price = asks[0].price
                    fill_size = asks[0].volume
                    fill = Order(
                        order_id=self._next_oid(), side=Side.BID,
                        price=fill_price, size=fill_size,
                        timestamp_ms=ts, action=OrderAction.FILL,
                        is_aggressive=True,
                    )
                    trades.append(fill)
                    events.append(fill)
                    if len(asks) > 1:
                        asks.pop(0)
                elif side == Side.ASK and bids:
                    fill_price = bids[0].price
                    fill_size = bids[0].volume
                    fill = Order(
                        order_id=self._next_oid(), side=Side.ASK,
                        price=fill_price, size=fill_size,
                        timestamp_ms=ts, action=OrderAction.FILL,
                        is_aggressive=True,
                    )
                    trades.append(fill)
                    events.append(fill)
                    if len(bids) > 1:
                        bids.pop(0)
        self._momentum_runs.clear()

        # Apply stop runs
        for side in self._stop_runs:
            target = bids if side == Side.ASK else asks
            levels_to_eat = min(len(target), random.randint(5, 10))
            for _ in range(levels_to_eat):
                if not target:
                    break
                lvl = target.pop(0)
                fill = Order(
                    order_id=self._next_oid(),
                    side=Side.BID if side == Side.ASK else Side.ASK,
                    price=lvl.price, size=lvl.volume,
                    timestamp_ms=ts, action=OrderAction.FILL,
                    is_aggressive=True,
                )
                trades.append(fill)
                events.append(fill)
        self._stop_runs.clear()

        # Add MODIFY and CANCEL events for realism
        for _ in range(random.randint(0, 3)):
            mod_side = random.choice([Side.BID, Side.ASK])
            target = bids if mod_side == Side.BID else asks
            if target:
                lvl = random.choice(target)
                events.append(Order(
                    order_id=self._next_oid(), side=mod_side,
                    price=lvl.price, size=round(lvl.volume * random.uniform(0.5, 1.5), 4),
                    timestamp_ms=ts, action=random.choice([OrderAction.MODIFY, OrderAction.CANCEL]),
                ))

        return OrderBookSnapshot(
            timestamp_ms=ts,
            symbol=self._symbol,
            bids=bids,
            asks=asks,
            recent_trades=trades,
            recent_events=events,
        )

    # ── pattern injection ────────────────────────────────────────

    def inject_institutional_wall(self, side: Side, price: float, volume: float) -> None:
        """Place a large single-order level (institutional wall).

        Args:
            side: BID or ASK
            price: Price level for the wall
            volume: Total volume (will appear as 1 order)
        """
        if volume <= 0:
            raise ValueError("volume must be > 0")
        self._injected_walls.append((side, price, volume))

    def inject_spoof(self, side: Side, price: float, volume: float) -> None:
        """Place an order that will be canceled on the next tick.

        Args:
            side: BID or ASK
            price: Price level for the spoof
            volume: Spoofed volume
        """
        if volume <= 0:
            raise ValueError("volume must be > 0")
        self._injected_spoofs.append((side, price, volume))

    def inject_momentum_run(self, side: Side, num_levels: int = 3) -> None:
        """Simulate consecutive aggressive fills eating through levels.

        Args:
            side: BID (buys eating asks) or ASK (sells eating bids)
            num_levels: Number of levels to consume
        """
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        self._momentum_runs.append((side, num_levels))

    def inject_stop_run(self, side: Side) -> None:
        """Simulate one side being overwhelmed (stop run).

        Args:
            side: The side being overwhelmed (BID = bids get eaten,
                  ASK = asks get eaten)
        """
        self._stop_runs.append(side)
