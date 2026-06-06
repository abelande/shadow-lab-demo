"""Infer iceberg: small visible size repeatedly refills after fills.

CME calibration notes:
- ES minimum tick is 0.25 points
- Iceberg orders on CME show as repeated small-lot fills at same price
- True icebergs: 5+ refills at same price with consistent small clip size
- Raise thresholds for ES where 1-lot clips are common for non-icebergs
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
from ..models import Order, OrderAction, SpoofEvent, SpoofType


@dataclass
class IcebergConfig:
    """Tunable thresholds for iceberg detection."""
    refill_count_threshold: int = 4       # Minimum fills at same price
    max_visible_size: float = 8.0        # Max clip size that suggests hidden quantity
    min_total_volume: float = 30.0       # Total filled volume must exceed this
    confidence_floor: float = 0.3


class IcebergInference:
    def __init__(self, config: Optional[IcebergConfig] = None):
        self.config = config or IcebergConfig()

    def detect(self, events: List[Order]) -> list[SpoofEvent]:
        cfg = self.config
        out: list[SpoofEvent] = []

        # Track fills and adds at each (side, price)
        fills_by_price: dict[tuple, list] = defaultdict(list)
        add_sizes: dict[tuple, list] = defaultdict(list)

        for e in events:
            if e.action == OrderAction.FILL:
                fills_by_price[(e.side, e.price)].append(e)
            elif e.action == OrderAction.ADD:
                add_sizes[(e.side, e.price)].append(e.size)

        for key, fills in fills_by_price.items():
            if len(fills) < cfg.refill_count_threshold:
                continue

            sizes = add_sizes.get(key, [])
            if not sizes:
                continue

            avg_visible = sum(sizes) / len(sizes)
            total_filled = sum(f.size for f in fills)

            # Must look like an iceberg: small visible clips, significant total volume
            if avg_visible > cfg.max_visible_size:
                continue
            if total_filled < cfg.min_total_volume:
                continue

            # Confidence: more fills + smaller clips + larger total = higher
            fill_conf = min(1.0, len(fills) / (cfg.refill_count_threshold * 3))
            size_ratio = total_filled / max(avg_visible, 1e-9)
            hidden_conf = min(1.0, size_ratio / 20)
            confidence = 0.5 * fill_conf + 0.5 * hidden_conf

            if confidence >= cfg.confidence_floor:
                side, price = key
                out.append(SpoofEvent(
                    spoof_type=SpoofType.ICEBERG,
                    price=price,
                    side=side,
                    confidence=confidence,
                    timestamp_ms=fills[-1].timestamp_ms,
                    details=f"{len(fills)} fills, avg clip {avg_visible:.1f}, total {total_filled:.1f}"
                ))

        return out
