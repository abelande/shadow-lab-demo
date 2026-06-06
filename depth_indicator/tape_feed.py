"""Tape feed (TIME/SIDE/PRICE) rows."""
from __future__ import annotations
from typing import List
from ..models import Order, TapeEntry


class TapeFeed:
    def build(self, trades: List[Order], limit: int = 30) -> List[TapeEntry]:
        selected = trades[-limit:]
        return [
            TapeEntry(timestamp_ms=t.timestamp_ms, side=t.side, price=t.price, size=t.size)
            for t in selected
        ]
