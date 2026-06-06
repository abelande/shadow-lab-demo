"""Detect failed fill sequences (stalls) after directional pressure."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from ..models import Order, OrderAction, Side


@dataclass
class StallSignal:
    side: Side
    failed_attempts: int
    window_ms: int
    confidence: float


class StallDetector:
    def __init__(self, min_failed_attempts: int = 3, window_ms: int = 1500):
        self.min_failed_attempts = min_failed_attempts
        self.window_ms = window_ms

    def detect(
        self,
        events: List[Order],
        push_side: Side,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
    ) -> StallSignal | None:
        """
        Stall = repeated aggressive attempts from push_side with low fill conversion.

        An order is aggressive if it crosses the spread:
          - BID ADD/MODIFY with price >= best_ask (buyer lifting the ask)
          - ASK ADD/MODIFY with price <= best_bid (seller hitting the bid)

        Falls back to the is_aggressive flag if best_bid/best_ask are unavailable.
        """
        if not events:
            return None
        t1 = events[-1].timestamp_ms
        window = [e for e in events if t1 - e.timestamp_ms <= self.window_ms]

        def _is_aggressive(e: Order) -> bool:
            if e.action not in (OrderAction.ADD, OrderAction.MODIFY):
                return False
            if best_bid is not None and best_ask is not None:
                if e.side == Side.BID and e.price >= best_ask:
                    return True
                if e.side == Side.ASK and e.price <= best_bid:
                    return True
                return False
            # Fallback: use the flag set at ingest time
            return e.is_aggressive

        attempts = [e for e in window if e.side == push_side and _is_aggressive(e)]
        fills = [e for e in window if e.side == push_side and e.action == OrderAction.FILL]
        failed = max(0, len(attempts) - len(fills))
        if failed < self.min_failed_attempts:
            return None
        conf = min(1.0, failed / (self.min_failed_attempts * 2))
        return StallSignal(side=push_side, failed_attempts=failed, window_ms=self.window_ms, confidence=conf)
