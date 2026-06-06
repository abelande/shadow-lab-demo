"""Net directional pressure scorer in range [-1,+1]."""
from __future__ import annotations
import math
from typing import List, Optional
from ..models import Order, OrderAction, Side, OrderBookSnapshot


class PressureScorer:
    def score(
        self,
        events: List[Order],
        decay_half_life_ms: float = 500.0,
        ofi: float = 0.0,
    ) -> float:
        """
        +: buy pressure
        -: sell pressure

        Heuristic from aggressive adds/fills and cancels on opposite side.
        Uses exponential time decay so recent events carry more weight.

        Args:
            events: Ordered list of events (most recent last).
            decay_half_life_ms: Half-life for exponential decay (ms). Events
                older than this contribute half the weight of the most recent.
        """
        if not events:
            return 0.0

        t_now = events[-1].timestamp_ms
        ln2 = math.log(2.0)

        buy = 0.0
        sell = 0.0
        for e in events:
            age_ms = max(0.0, t_now - e.timestamp_ms)
            # decay: 1.0 at age=0, 0.5 at age=half_life, etc.
            decay = math.exp(-ln2 * age_ms / decay_half_life_ms) if decay_half_life_ms > 0 else 1.0

            w = max(1.0, e.size) * decay

            if e.action == OrderAction.FILL:
                # fill on ask means buyers consuming asks
                if e.side == Side.ASK:
                    buy += 1.4 * w
                else:
                    sell += 1.4 * w
            elif e.action == OrderAction.CANCEL:
                # ask cancel opens upside; bid cancel opens downside
                if e.side == Side.ASK:
                    buy += 0.6 * w
                else:
                    sell += 0.6 * w
            elif e.action in (OrderAction.ADD, OrderAction.MODIFY) and e.is_aggressive:
                if e.side == Side.BID:
                    buy += 0.8 * w
                else:
                    sell += 0.8 * w

        tot = buy + sell
        if tot <= 0:
            return 0.0
        raw = (buy - sell) / tot
        event_pressure = max(-1.0, min(1.0, raw))

        # Blend with OFI when available — OFI captures volume-weighted
        # flow changes that event-count pressure misses.
        if ofi != 0.0:
            ofi_normalized = math.tanh(ofi)  # squash to [-1, 1]
            enriched = 0.6 * ofi_normalized + 0.4 * event_pressure
            return max(-1.0, min(1.0, enriched))
        return event_pressure

    @staticmethod
    def depth_context(snapshot: OrderBookSnapshot, n_levels: int = 5) -> float:
        """Bid/ask depth ratio across top N levels. >1 = bid-heavy, <1 = ask-heavy."""
        bid_vol = sum(lv.volume for lv in snapshot.bids[:n_levels]) if snapshot.bids else 0.0
        ask_vol = sum(lv.volume for lv in snapshot.asks[:n_levels]) if snapshot.asks else 0.0
        total = bid_vol + ask_vol
        if total <= 0:
            return 1.0
        return bid_vol / max(ask_vol, 1e-10)
