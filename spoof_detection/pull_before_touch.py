"""Detect orders canceled within T ms before being touched at best.

CME calibration notes:
- Normal HFT market making: add+cancel at best within <400ms is routine
- True spoofing indicators: large size, very fast cancel, repeated pattern
- We require minimum size AND repeated behavior to flag
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
from ..models import Order, OrderAction, Side, SpoofEvent, SpoofType


@dataclass
class PullBeforeTouchConfig:
    """Tunable thresholds for pull-before-touch detection."""
    threshold_ms: int = 150          # Max time between add and cancel
    min_size: float = 3.0            # Min order size (ES 1-lot is normal, 3+ is meaningful)
    min_repeats: int = 2             # Must see N pulls at same price to flag
    cooldown_ms: int = 500           # Suppress repeat flags within this window
    confidence_floor: float = 0.3    # Minimum confidence to emit event


class PullBeforeTouchDetector:
    def __init__(self, config: Optional[PullBeforeTouchConfig] = None):
        self.config = config or PullBeforeTouchConfig()
        self._pull_counts: dict[tuple, int] = defaultdict(int)  # (side, price) -> count
        self._last_flag_ts: dict[tuple, int] = {}  # (side, price) -> last flag timestamp

    def detect(self, events: List[Order], best_bid: float | None, best_ask: float | None) -> list[SpoofEvent]:
        out: list[SpoofEvent] = []
        if best_bid is None or best_ask is None:
            return out

        cfg = self.config
        adds: dict[str, Order] = {}

        # First pass: find add-then-cancel pairs
        pulls_this_window: dict[tuple, list] = defaultdict(list)

        for e in events:
            if e.action == OrderAction.ADD:
                adds[e.order_id] = e
            elif e.action == OrderAction.CANCEL and e.order_id in adds:
                a = adds[e.order_id]
                dt = e.timestamp_ms - a.timestamp_ms

                # Filter: must be near best, meaningful size, fast cancel
                near_best = abs(a.price - (best_bid if a.side == Side.BID else best_ask)) <= 1e-9
                large_enough = a.size >= cfg.min_size
                fast_enough = 0 <= dt <= cfg.threshold_ms

                if near_best and large_enough and fast_enough:
                    key = (a.side, a.price)
                    pulls_this_window[key].append((a, e, dt))

        # Second pass: only flag if we see repeated pulls (pattern, not noise)
        for key, pulls in pulls_this_window.items():
            if len(pulls) < cfg.min_repeats:
                continue

            # Cooldown: don't re-flag same price level too often
            last_flag = self._last_flag_ts.get(key, 0)
            latest_ts = max(p[1].timestamp_ms for p in pulls)
            if latest_ts - last_flag < cfg.cooldown_ms:
                continue

            # Aggregate into single event with confidence based on count and speed
            avg_dt = sum(p[2] for p in pulls) / len(pulls)
            total_size = sum(p[0].size for p in pulls)
            speed_conf = max(0, 1 - avg_dt / max(cfg.threshold_ms, 1))
            repeat_conf = min(1.0, len(pulls) / (cfg.min_repeats * 3))
            confidence = 0.5 * speed_conf + 0.5 * repeat_conf

            if confidence >= cfg.confidence_floor:
                side, price = key
                out.append(SpoofEvent(
                    spoof_type=SpoofType.PULL_BEFORE_TOUCH,
                    price=price,
                    side=side,
                    confidence=confidence,
                    timestamp_ms=latest_ts,
                    details=f"{len(pulls)} pulls, avg {avg_dt:.0f}ms, total size {total_size:.1f}"
                ))
                self._last_flag_ts[key] = latest_ts
                self._pull_counts[key] += len(pulls)

        return out
