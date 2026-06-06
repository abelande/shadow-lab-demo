"""Render depth bars: red asks above, blue bids below, length ∝ volume."""
from __future__ import annotations
from typing import List, Tuple
from ..models import OrderBookLevel, Side, DepthBar


class BarRenderer:
    def render(self, bids: List[OrderBookLevel], asks: List[OrderBookLevel], authenticity: float = 1.0) -> Tuple[list[DepthBar], list[DepthBar]]:
        maxv = max([l.volume for l in bids + asks], default=1.0)

        bid_bars: list[DepthBar] = []
        ask_bars: list[DepthBar] = []

        cum = 0.0
        for l in bids:
            cum += l.volume
            bid_bars.append(DepthBar(
                price=l.price,
                side=Side.BID,
                volume=l.volume,
                order_count=l.order_count,
                cumulative_volume=cum,
                bar_length=(l.volume / maxv) if maxv else 0.0,
                is_round_number=abs(l.price - round(l.price)) < 1e-9,
                authenticity=authenticity,
            ))

        cum = 0.0
        for l in asks:
            cum += l.volume
            ask_bars.append(DepthBar(
                price=l.price,
                side=Side.ASK,
                volume=l.volume,
                order_count=l.order_count,
                cumulative_volume=cum,
                bar_length=(l.volume / maxv) if maxv else 0.0,
                is_round_number=abs(l.price - round(l.price)) < 1e-9,
                authenticity=authenticity,
            ))

        return bid_bars, ask_bars
