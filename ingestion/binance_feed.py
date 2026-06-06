"""Binance L2 WebSocket feed for live order book data."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import List, Optional

from ..models import (
    Order, OrderAction, OrderBookLevel, OrderBookSnapshot, Side,
)
from .base_feed import BaseFeed

logger = logging.getLogger(__name__)


class BinanceFeed(BaseFeed):
    """Live L2 order book feed from Binance WebSocket API.

    Connects to the depth20@100ms stream for order book snapshots
    and the trade stream for real-time trade tape.

    Requires the `websockets` library: pip install websockets

    Attributes:
        symbol: Binance trading pair (e.g. "btcusdt")
        max_reconnect_attempts: Max consecutive reconnect attempts
        reconnect_delay: Base delay between reconnects (seconds)
    """

    BASE_URL = "wss://stream.binance.com:9443/ws"

    def __init__(
        self,
        symbol: str = "btcusdt",
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 1.0,
    ):
        super().__init__(symbol=symbol.lower(), data_level="L2")
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._ws = None
        self._trade_ws = None
        self._latest_snapshot: Optional[OrderBookSnapshot] = None
        self._recent_trades: List[Order] = []
        self._trade_buffer_size = 100
        self._reconnect_count = 0
        self._depth_task: Optional[asyncio.Task] = None
        self._trade_task: Optional[asyncio.Task] = None
        self._order_id_counter = 0

    def _next_oid(self) -> str:
        self._order_id_counter += 1
        return f"BN-{self._order_id_counter:08d}"

    @property
    def depth_url(self) -> str:
        return f"{self.BASE_URL}/{self._symbol}@depth20@100ms"

    @property
    def trade_url(self) -> str:
        return f"{self.BASE_URL}/{self._symbol}@trade"

    async def connect(self) -> None:
        """Connect to Binance WebSocket streams."""
        try:
            import websockets  # noqa: F401
        except ImportError:
            raise ImportError(
                "websockets library required: pip install websockets"
            )

        self._connected = True
        self._reconnect_count = 0

        self._depth_task = asyncio.create_task(self._depth_loop())
        self._trade_task = asyncio.create_task(self._trade_loop())

        # Wait briefly for first snapshot
        for _ in range(50):
            if self._latest_snapshot is not None:
                break
            await asyncio.sleep(0.1)

    async def _depth_loop(self) -> None:
        """Background loop consuming depth stream with auto-reconnect."""
        import websockets

        while self._connected:
            try:
                async with websockets.connect(self.depth_url) as ws:
                    self._ws = ws
                    self._reconnect_count = 0
                    logger.info(f"Connected to depth stream: {self._symbol}")

                    async for raw in ws:
                        if not self._connected:
                            break
                        try:
                            data = json.loads(raw)
                            self._process_depth(data)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Bad depth message: {e}")

            except Exception as e:
                if not self._connected:
                    break
                self._reconnect_count += 1
                if self._reconnect_count > self.max_reconnect_attempts:
                    logger.error("Max reconnect attempts reached for depth")
                    break
                delay = self.reconnect_delay * (2 ** min(self._reconnect_count, 6))
                logger.warning(f"Depth disconnected: {e}, reconnecting in {delay:.1f}s")
                await asyncio.sleep(delay)

    async def _trade_loop(self) -> None:
        """Background loop consuming trade stream with auto-reconnect."""
        import websockets

        reconnect_count = 0
        while self._connected:
            try:
                async with websockets.connect(self.trade_url) as ws:
                    self._trade_ws = ws
                    reconnect_count = 0
                    logger.info(f"Connected to trade stream: {self._symbol}")

                    async for raw in ws:
                        if not self._connected:
                            break
                        try:
                            data = json.loads(raw)
                            self._process_trade(data)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Bad trade message: {e}")

            except Exception as e:
                if not self._connected:
                    break
                reconnect_count += 1
                if reconnect_count > self.max_reconnect_attempts:
                    logger.error("Max reconnect attempts reached for trades")
                    break
                delay = self.reconnect_delay * (2 ** min(reconnect_count, 6))
                logger.warning(f"Trade disconnected: {e}, reconnecting in {delay:.1f}s")
                await asyncio.sleep(delay)

    def _process_depth(self, data: dict) -> None:
        """Convert Binance depth snapshot to OrderBookSnapshot."""
        ts = int(time.time() * 1000)

        bids: List[OrderBookLevel] = []
        for price_str, vol_str in data.get("bids", []):
            price = float(price_str)
            volume = float(vol_str)
            if volume > 0:
                bids.append(OrderBookLevel(
                    price=price, side=Side.BID,
                    volume=volume, order_count=0,
                ))

        asks: List[OrderBookLevel] = []
        for price_str, vol_str in data.get("asks", []):
            price = float(price_str)
            volume = float(vol_str)
            if volume > 0:
                asks.append(OrderBookLevel(
                    price=price, side=Side.ASK,
                    volume=volume, order_count=0,
                ))

        self._latest_snapshot = OrderBookSnapshot(
            timestamp_ms=ts,
            symbol=self._symbol.upper(),
            bids=bids,
            asks=asks,
            recent_trades=list(self._recent_trades),
            recent_events=[],
        )

    def _process_trade(self, data: dict) -> None:
        """Convert Binance trade to Order object."""
        buyer_is_maker = data.get("m", False)
        trade = Order(
            order_id=self._next_oid(),
            side=Side.ASK if buyer_is_maker else Side.BID,
            price=float(data["p"]),
            size=float(data["q"]),
            timestamp_ms=int(data["T"]),
            action=OrderAction.FILL,
            is_aggressive=True,
        )
        self._recent_trades.append(trade)
        if len(self._recent_trades) > self._trade_buffer_size:
            self._recent_trades = self._recent_trades[-self._trade_buffer_size:]

    async def next(self) -> Optional[OrderBookSnapshot]:
        """Return the latest snapshot."""
        if not self._connected:
            raise RuntimeError("Feed not connected. Call connect() first.")

        for _ in range(50):
            if self._latest_snapshot is not None:
                break
            await asyncio.sleep(0.1)

        return self._latest_snapshot

    async def disconnect(self) -> None:
        """Disconnect from Binance WebSocket streams."""
        self._connected = False

        for task in (self._depth_task, self._trade_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        for ws in (self._ws, self._trade_ws):
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass

        self._ws = None
        self._trade_ws = None
        self._latest_snapshot = None
        await super().disconnect()
