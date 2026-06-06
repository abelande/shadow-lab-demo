"""
Volume Profile — aggregate volume per price level.
The foundation of the staircase: how much size sits at each rung.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from ..models import OrderBookSnapshot, Side


class VolumeProfiler:
    """Builds a volume-per-price map from an L3 order book snapshot."""

    def compute(self, snapshot: OrderBookSnapshot) -> Dict[float, Dict[str, float]]:
        """
        Returns {price: {"volume": float, "side": "BID"|"ASK"}}
        sorted by price descending for bids, ascending for asks.
        """
        profile: Dict[float, Dict[str, float]] = {}

        for level in snapshot.bids:
            profile[level.price] = {
                "volume": level.volume,
                "side": Side.BID.value,
            }

        for level in snapshot.asks:
            profile[level.price] = {
                "volume": level.volume,
                "side": Side.ASK.value,
            }

        return profile

    def bid_volumes(self, snapshot: OrderBookSnapshot) -> List[Tuple[float, float]]:
        """Returns [(price, volume)] for bids, sorted price desc."""
        return [(l.price, l.volume) for l in snapshot.bids]

    def ask_volumes(self, snapshot: OrderBookSnapshot) -> List[Tuple[float, float]]:
        """Returns [(price, volume)] for asks, sorted price asc."""
        return [(l.price, l.volume) for l in snapshot.asks]

    def total_bid_volume(self, snapshot: OrderBookSnapshot) -> float:
        return sum(l.volume for l in snapshot.bids)

    def total_ask_volume(self, snapshot: OrderBookSnapshot) -> float:
        return sum(l.volume for l in snapshot.asks)

    def imbalance_ratio(self, snapshot: OrderBookSnapshot) -> float:
        """(bid_vol - ask_vol) / (bid_vol + ask_vol). Range [-1, 1]."""
        bid = self.total_bid_volume(snapshot)
        ask = self.total_ask_volume(snapshot)
        total = bid + ask
        if total == 0:
            return 0.0
        return (bid - ask) / total
