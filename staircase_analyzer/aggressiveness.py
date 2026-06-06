"""
Aggressiveness Classifier — passive vs aggressive order classification.

Aggressive: order crosses the spread (marketable limit or market order)
Passive:    order rests in the book (limit order inside own side)

Aggressive ratio at a level tells you whether that level was built by
patient liquidity providers or impatient takers.
"""
from __future__ import annotations
from typing import Dict
from ..models import OrderBookSnapshot, OrderBookLevel, Side, Order


class AggressivenessClassifier:
    """Classifies orders and levels by aggressiveness."""

    def classify_order(
        self, order: Order, best_bid: float | None, best_ask: float | None
    ) -> bool:
        """Returns True if order is aggressive (crosses spread)."""
        if best_bid is None or best_ask is None:
            return False
        if order.side == Side.BID:
            return order.price >= best_ask  # bid at or above ask = aggressive
        else:
            return order.price <= best_bid  # ask at or below bid = aggressive

    def level_aggressive_ratio(self, level: OrderBookLevel) -> float:
        """Fraction of orders at this level that were placed aggressively."""
        if not level.orders:
            return 0.0
        aggressive_count = sum(1 for o in level.orders if o.is_aggressive)
        return aggressive_count / len(level.orders)

    def compute(self, snapshot: OrderBookSnapshot) -> Dict[float, float]:
        """Returns {price: aggressive_ratio} for all levels."""
        result: Dict[float, float] = {}
        for level in snapshot.bids + snapshot.asks:
            result[level.price] = self.level_aggressive_ratio(level)
        return result

    def net_aggression(self, snapshot: OrderBookSnapshot) -> float:
        """
        Net aggressiveness: positive = more aggressive buying,
        negative = more aggressive selling. Range roughly [-1, 1].
        """
        bid_agg = 0.0
        ask_agg = 0.0
        bid_vol = 0.0
        ask_vol = 0.0

        for level in snapshot.bids:
            ratio = self.level_aggressive_ratio(level)
            bid_agg += ratio * level.volume
            bid_vol += level.volume

        for level in snapshot.asks:
            ratio = self.level_aggressive_ratio(level)
            ask_agg += ratio * level.volume
            ask_vol += level.volume

        total = bid_vol + ask_vol
        if total == 0:
            return 0.0

        return (bid_agg - ask_agg) / total
