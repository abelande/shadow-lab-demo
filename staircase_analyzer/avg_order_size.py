"""
Average Order Size — vol/count → institutional vs retail fingerprint.

Institutional:  large avg order size, few orders
Retail:         small avg order size, many orders
Mixed:          moderate on both axes

This is the L3 edge made actionable.
"""
from __future__ import annotations
from typing import List, Tuple
from ..models import OrderBookSnapshot, OrderBookLevel, Side


class SizeClassification:
    INSTITUTIONAL = "INSTITUTIONAL"
    RETAIL = "RETAIL"
    MIXED = "MIXED"


class AvgOrderSizeAnalyzer:
    """Classifies each price level by average order size."""

    def __init__(
        self,
        institutional_threshold: float = 50.0,
        retail_threshold: float = 5.0,
    ):
        self.institutional_threshold = institutional_threshold
        self.retail_threshold = retail_threshold

    def avg_size(self, level: OrderBookLevel) -> float:
        if level.order_count == 0:
            return 0.0
        return level.volume / level.order_count

    def classify_level(self, level: OrderBookLevel) -> str:
        avg = self.avg_size(level)
        if avg >= self.institutional_threshold:
            return SizeClassification.INSTITUTIONAL
        elif avg <= self.retail_threshold:
            return SizeClassification.RETAIL
        return SizeClassification.MIXED

    def profile(
        self, snapshot: OrderBookSnapshot
    ) -> List[Tuple[float, float, str]]:
        """
        Returns [(price, avg_size, classification)] for all levels.
        """
        results = []
        for level in snapshot.bids + snapshot.asks:
            avg = self.avg_size(level)
            cls = self.classify_level(level)
            results.append((level.price, avg, cls))
        return results

    def institutional_volume_fraction(
        self, snapshot: OrderBookSnapshot, side: Side
    ) -> float:
        """Fraction of volume at institutional-classified levels."""
        levels = snapshot.bids if side == Side.BID else snapshot.asks
        total_vol = sum(l.volume for l in levels)
        if total_vol == 0:
            return 0.0
        inst_vol = sum(
            l.volume for l in levels
            if self.classify_level(l) == SizeClassification.INSTITUTIONAL
        )
        return inst_vol / total_vol
