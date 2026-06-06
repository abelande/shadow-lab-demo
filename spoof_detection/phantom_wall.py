"""Detect phantom wall: large order vanishes as price approaches.

CME calibration notes:
- ES typical large order: 50+ contracts at a level
- Institutional walls can be 200+ contracts
- Normal cancel of large order ≠ phantom wall; need price approach + fast cancel
- Raise size threshold significantly for ES futures
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from ..models import Order, OrderAction, Side, SpoofEvent, SpoofType


@dataclass
class PhantomWallConfig:
    """Tunable thresholds for phantom wall detection."""
    large_size_threshold: float = 50.0   # Min contracts to qualify as "wall" (raised from 100 generic)
    approach_ticks: float = 2.0          # Price must be within N ticks of wall
    cancel_ms: int = 500                 # Must cancel within this window (tightened from 800)
    min_wall_duration_ms: int = 100      # Wall must exist for at least this long (filters HFT)
    confidence_floor: float = 0.3


class PhantomWallDetector:
    def __init__(self, config: Optional[PhantomWallConfig] = None):
        self.config = config or PhantomWallConfig()

    def detect(self, events: List[Order], mid_price: float | None) -> list[SpoofEvent]:
        if mid_price is None:
            return []

        cfg = self.config
        out: list[SpoofEvent] = []
        adds: dict[str, Order] = {}

        for e in events:
            if e.action == OrderAction.ADD and e.size >= cfg.large_size_threshold:
                adds[e.order_id] = e
            elif e.action == OrderAction.CANCEL and e.order_id in adds:
                a = adds[e.order_id]
                dt = e.timestamp_ms - a.timestamp_ms
                approached = abs(a.price - mid_price) <= cfg.approach_ticks

                # Must have existed long enough to be visible, but canceled fast on approach
                existed_long_enough = dt >= cfg.min_wall_duration_ms
                canceled_fast = dt <= cfg.cancel_ms

                if approached and existed_long_enough and canceled_fast:
                    # Confidence: larger walls + closer to mid + faster cancel = higher
                    size_conf = min(1.0, a.size / (cfg.large_size_threshold * 4))
                    speed_conf = max(0, 1 - dt / max(cfg.cancel_ms, 1))
                    proximity_conf = max(0, 1 - abs(a.price - mid_price) / max(cfg.approach_ticks, 1e-9))
                    confidence = 0.4 * size_conf + 0.3 * speed_conf + 0.3 * proximity_conf

                    if confidence >= cfg.confidence_floor:
                        out.append(SpoofEvent(
                            spoof_type=SpoofType.PHANTOM_WALL,
                            price=a.price,
                            side=a.side,
                            confidence=confidence,
                            timestamp_ms=e.timestamp_ms,
                            details=f"wall size={a.size:.0f} canceled after {dt}ms on approach"
                        ))
        return out
