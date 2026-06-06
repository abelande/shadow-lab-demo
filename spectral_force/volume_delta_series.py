"""Build signed volume-delta time series from event tape."""
from __future__ import annotations
from typing import List
from ..models import Order, OrderAction, Side


class VolumeDeltaSeries:
    def build(self, events: List[Order], bucket_ms: int = 200) -> list[float]:
        if not events:
            return []
        events = sorted(events, key=lambda e: e.timestamp_ms)
        start = events[0].timestamp_ms
        buckets: dict[int, float] = {}
        for e in events:
            b = (e.timestamp_ms - start) // bucket_ms
            if e.action != OrderAction.FILL:
                continue
            # fill at ask = buy aggression (+), fill at bid = sell aggression (-)
            sgn = 1.0 if e.side == Side.ASK else -1.0
            buckets[b] = buckets.get(b, 0.0) + sgn * e.size
        max_b = max(buckets.keys(), default=-1)
        return [buckets.get(i, 0.0) for i in range(max_b + 1)]
