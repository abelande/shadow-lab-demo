"""Stats header: CVD, T/s, ADD/CXL/MOD/FIL/LIVE."""
from __future__ import annotations
from ..models import OrderBookSnapshot, OrderAction, Side, StatsSnapshot


class StatsHeader:
    def build(self, snapshot: OrderBookSnapshot) -> StatsSnapshot:
        events = snapshot.recent_events
        if events:
            t0, t1 = events[0].timestamp_ms, events[-1].timestamp_ms
            dt = max(1, t1 - t0)
            trades_per_sec = (len(snapshot.recent_trades) * 1000.0) / dt
        else:
            trades_per_sec = 0.0

        cvd = 0.0
        for tr in snapshot.recent_trades:
            cvd += tr.size if tr.side == Side.ASK else -tr.size

        add_count = sum(1 for e in events if e.action == OrderAction.ADD)
        cancel_count = sum(1 for e in events if e.action == OrderAction.CANCEL)
        modify_count = sum(1 for e in events if e.action == OrderAction.MODIFY)
        fill_count = sum(1 for e in events if e.action == OrderAction.FILL)
        live_orders = sum(l.order_count for l in snapshot.bids + snapshot.asks)

        return StatsSnapshot(
            cvd=cvd,
            trades_per_sec=trades_per_sec,
            add_count=add_count,
            cancel_count=cancel_count,
            modify_count=modify_count,
            fill_count=fill_count,
            live_orders=live_orders,
        )
