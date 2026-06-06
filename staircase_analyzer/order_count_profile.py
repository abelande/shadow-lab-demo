"""
Order Count Profile — #ORD per price level.
This is the L3 edge: L2 books show volume but not how many discrete orders
comprise that volume. A level with 500 lots in 2 orders is VERY different
from 500 lots in 200 orders.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from ..models import OrderBookSnapshot, Side


class OrderCountProfiler:
    """Extracts order count per price level from L3 data."""

    def compute(self, snapshot: OrderBookSnapshot) -> Dict[float, int]:
        """Returns {price: order_count}."""
        counts: Dict[float, int] = {}
        for level in snapshot.bids + snapshot.asks:
            counts[level.price] = level.order_count
        return counts

    def bid_counts(self, snapshot: OrderBookSnapshot) -> List[Tuple[float, int]]:
        """[(price, count)] for bids, sorted price desc."""
        return [(l.price, l.order_count) for l in snapshot.bids]

    def ask_counts(self, snapshot: OrderBookSnapshot) -> List[Tuple[float, int]]:
        """[(price, count)] for asks, sorted price asc."""
        return [(l.price, l.order_count) for l in snapshot.asks]

    def concentration_ratio(self, snapshot: OrderBookSnapshot, side: Side) -> float:
        """
        Ratio of volume at best level to total volume on that side.
        High concentration = fragile top-of-book.
        """
        levels = snapshot.bids if side == Side.BID else snapshot.asks
        if not levels:
            return 0.0
        total = sum(l.volume for l in levels)
        if total == 0:
            return 0.0
        return levels[0].volume / total
