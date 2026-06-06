"""
Fragility Scorer — FRAGILE vs SOLID classification per level.

FRAGILE: few orders, large avg size → one cancel wipes the level
SOLID:   many orders, small avg size → resilient, distributed liquidity
MODERATE: in between

Score: 0 (solid) → 1 (fragile)

The key insight: a 1000-lot level from 2 orders is infinitely more
fragile than a 1000-lot level from 200 orders. L3 makes this visible.
"""
from __future__ import annotations
from typing import List
import math
from ..models import (
    OrderBookSnapshot, OrderBookLevel, Side,
    FragilityState, LevelProfile, StaircaseProfile,
)
from .volume_profile import VolumeProfiler
from .order_count_profile import OrderCountProfiler
from .avg_order_size import AvgOrderSizeAnalyzer
from .aggressiveness import AggressivenessClassifier


class FragilityScorer:
    """
    Scores each level's fragility and assembles the full StaircaseProfile.
    """

    def __init__(
        self,
        fragile_threshold: float = 0.65,
        solid_threshold: float = 0.35,
        min_orders_solid: int = 5,
    ):
        self.fragile_threshold = fragile_threshold
        self.solid_threshold = solid_threshold
        self.min_orders_solid = min_orders_solid
        self._vol = VolumeProfiler()
        self._cnt = OrderCountProfiler()
        self._avg = AvgOrderSizeAnalyzer()
        self._agg = AggressivenessClassifier()

    def score_level(self, level: OrderBookLevel, median_count: float) -> float:
        """
        Fragility score 0-1.
        High when: few orders AND large avg size relative to peers.
        """
        if level.order_count == 0:
            return 1.0  # empty level is maximally fragile

        # Inverse count component: fewer orders → higher fragility
        count_component = 1.0 / (1.0 + math.log1p(level.order_count))

        # Concentration: what fraction of volume is the largest order?
        if level.orders:
            max_order = max(o.size for o in level.orders)
            concentration = max_order / level.volume if level.volume > 0 else 1.0
        else:
            # Without L3 detail, use count heuristic
            concentration = 1.0 / level.order_count if level.order_count > 0 else 1.0

        # Below median count penalized
        count_ratio = level.order_count / median_count if median_count > 0 else 1.0
        below_median_penalty = max(0, 1.0 - count_ratio)

        score = 0.4 * count_component + 0.35 * concentration + 0.25 * below_median_penalty
        return min(1.0, max(0.0, score))

    def classify(self, score: float) -> FragilityState:
        if score >= self.fragile_threshold:
            return FragilityState.FRAGILE
        elif score <= self.solid_threshold:
            return FragilityState.SOLID
        return FragilityState.MODERATE

    def build_profile(self, snapshot: OrderBookSnapshot) -> StaircaseProfile:
        """Build the full Layer 1 StaircaseProfile from a snapshot."""
        all_levels = snapshot.bids + snapshot.asks
        counts = [l.order_count for l in all_levels if l.order_count > 0]
        median_count = sorted(counts)[len(counts) // 2] if counts else 1.0

        def make_level_profile(level: OrderBookLevel) -> LevelProfile:
            avg = self._avg.avg_size(level)
            agg_ratio = self._agg.level_aggressive_ratio(level)
            frag_score = self.score_level(level, median_count)
            frag_state = self.classify(frag_score)
            return LevelProfile(
                price=level.price,
                side=level.side,
                volume=level.volume,
                order_count=level.order_count,
                avg_order_size=avg,
                aggressive_ratio=agg_ratio,
                fragility=frag_state,
                fragility_score=frag_score,
            )

        bid_profiles = [make_level_profile(l) for l in snapshot.bids]
        ask_profiles = [make_level_profile(l) for l in snapshot.asks]

        bid_total = self._vol.total_bid_volume(snapshot)
        ask_total = self._vol.total_ask_volume(snapshot)
        imbalance = self._vol.imbalance_ratio(snapshot)

        return StaircaseProfile(
            timestamp_ms=snapshot.timestamp_ms,
            bid_levels=bid_profiles,
            ask_levels=ask_profiles,
            bid_total_volume=bid_total,
            ask_total_volume=ask_total,
            imbalance_ratio=imbalance,
        )
