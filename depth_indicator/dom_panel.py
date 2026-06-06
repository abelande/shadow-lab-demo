"""DOM panel with VOL/#ORD/CUM columns."""
from __future__ import annotations
from typing import List
from ..models import DepthBar, DOMRow


class DOMPanel:
    def build(self, bid_bars: List[DepthBar], ask_bars: List[DepthBar]) -> List[DOMRow]:
        rows: list[DOMRow] = []
        for b in bid_bars:
            rows.append(DOMRow(price=b.price, volume=b.volume, order_count=b.order_count, cumulative_volume=b.cumulative_volume, side=b.side))
        for a in ask_bars:
            rows.append(DOMRow(price=a.price, volume=a.volume, order_count=a.order_count, cumulative_volume=a.cumulative_volume, side=a.side))
        # for display, sort by price desc
        return sorted(rows, key=lambda r: r.price, reverse=True)
