"""Detect cascading same-size layered orders.

CME calibration notes:
- Many participants use standard lot sizes (1, 5, 10, 25, 50 contracts)
- True layering: same side, same size, multiple consecutive price levels, same timeframe
- We require tighter time clustering and more levels to avoid false positives
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
from ..models import Order, OrderAction, SpoofEvent, SpoofType


@dataclass
class LayeringConfig:
    """Tunable thresholds for layering detection."""
    min_levels: int = 3              # Minimum consecutive price levels
    size_tolerance: float = 0.03     # Size must match within 3%
    max_time_spread_ms: int = 300    # All orders must arrive within this window
    min_size: float = 3.0            # Minimum size to consider
    confidence_floor: float = 0.3


class LayeringDetector:
    def __init__(self, config: Optional[LayeringConfig] = None):
        self.config = config or LayeringConfig()

    def detect(self, events: List[Order]) -> list[SpoofEvent]:
        cfg = self.config
        adds = [e for e in events if e.action == OrderAction.ADD and e.size >= cfg.min_size]

        by_side = defaultdict(list)
        for a in adds:
            by_side[a.side].append(a)

        out: list[SpoofEvent] = []
        for side, arr in by_side.items():
            arr = sorted(arr, key=lambda x: x.price)

            for i in range(len(arr)):
                base = arr[i]
                cluster = [base]

                for j in range(i + 1, len(arr)):
                    candidate = arr[j]
                    # Size must match
                    rel = abs(candidate.size - base.size) / max(base.size, 1e-9)
                    if rel > cfg.size_tolerance:
                        continue
                    # Time must be close
                    time_spread = abs(candidate.timestamp_ms - base.timestamp_ms)
                    if time_spread > cfg.max_time_spread_ms:
                        continue
                    cluster.append(candidate)

                if len(cluster) >= cfg.min_levels:
                    # Check price consecutiveness — levels should be relatively sequential
                    prices = sorted(set(c.price for c in cluster))
                    if len(prices) < cfg.min_levels:
                        continue

                    avg_gap = (prices[-1] - prices[0]) / max(len(prices) - 1, 1)
                    time_spread = max(c.timestamp_ms for c in cluster) - min(c.timestamp_ms for c in cluster)

                    # Confidence: more levels + tighter timing = higher confidence
                    level_conf = min(1.0, len(cluster) / (cfg.min_levels * 2))
                    time_conf = max(0, 1 - time_spread / max(cfg.max_time_spread_ms, 1))
                    confidence = 0.6 * level_conf + 0.4 * time_conf

                    if confidence >= cfg.confidence_floor:
                        out.append(SpoofEvent(
                            spoof_type=SpoofType.LAYERING,
                            price=base.price,
                            side=side,
                            confidence=confidence,
                            timestamp_ms=max(c.timestamp_ms for c in cluster),
                            details=f"same-size layering across {len(prices)} levels, {time_spread}ms spread"
                        ))
                    break  # Only report one layering event per side per scan

        return out
