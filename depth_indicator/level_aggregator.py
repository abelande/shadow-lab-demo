"""Adaptive bucketing by round numbers + volume clustering."""
from __future__ import annotations
from collections import defaultdict
from typing import List
from ..models import OrderBookLevel, Side


class LevelAggregator:
    def __init__(self, round_step: float = 1.0, cluster_threshold: float = 0.2):
        self.round_step = round_step
        self.cluster_threshold = cluster_threshold

    def _bucket_price(self, p: float) -> float:
        return round(p / self.round_step) * self.round_step

    def aggregate(self, levels: List[OrderBookLevel]) -> List[OrderBookLevel]:
        if not levels:
            return []
        buckets: dict[tuple[Side, float], OrderBookLevel] = {}
        for l in levels:
            bp = self._bucket_price(l.price)
            k = (l.side, bp)
            if k not in buckets:
                buckets[k] = OrderBookLevel(price=bp, side=l.side, volume=0.0, order_count=0, orders=[])
            b = buckets[k]
            b.volume += l.volume
            b.order_count += l.order_count
            b.orders.extend(l.orders)

        out = list(buckets.values())
        bids = sorted([x for x in out if x.side == Side.BID], key=lambda x: x.price, reverse=True)
        asks = sorted([x for x in out if x.side == Side.ASK], key=lambda x: x.price)
        return bids + asks
