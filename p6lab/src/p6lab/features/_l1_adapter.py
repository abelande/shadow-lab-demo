"""
p6lab.features._l1_adapter — p6-v2 OrderBookSnapshot → L1Snapshot + L1History.

Spec: p6-notebook-lab-spec.md §3.1 (TripleView emitter), §4.1 (L1 features)

The L1 feature pipeline operates on ``L1Snapshot`` + ``L1History``
(structures defined in ``l1_features.py``). p6-v2's live pipeline
produces ``OrderBookSnapshot`` objects with richer data: full book
depth, recent trades, recent events (ADD/CANCEL/MODIFY/FILL).

This adapter is the bridge. Given a stream of p6-v2 snapshots, it:
  1. Extracts best-bid / best-ask / sizes / top-of-book tick size
     into an ``L1Snapshot``.
  2. Bucketizes ``recent_events`` into passive-bid-adds and
     passive-ask-adds for the refresh-rate features.
  3. Classifies ``recent_trades`` into bid-side vs ask-side using
     Lee-Ready-style comparison to the midpoint (with a trade.side
     tiebreaker at-mid).
  4. Appends everything to an ``L1History`` with correct timestamps.

This is the ONLY module in p6lab that imports from p6-v2. Every other
module consumes ``L1Snapshot`` / ``L1History`` directly.

Usage::

    from p6lab.features._l1_adapter import L1Adapter
    from p6lab.features.l1_features import compute_l1_features

    adapter = L1Adapter(tick_size=0.25)
    history = adapter.history
    for snap in replay_feed:
        l1_snap = adapter.ingest(snap)
        features = compute_l1_features(l1_snap, history)
        # features is np.ndarray[16]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .l1_features import L1History, L1Snapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols — duck-typed surface of p6-v2 types
# ---------------------------------------------------------------------------

@runtime_checkable
class P6Order(Protocol):
    """Protocol matching p6-v2 models.Order. Subset we use."""
    order_id: str
    side: Any          # Side enum; .name is "BID" | "ASK"
    price: float
    size: float
    timestamp_ms: int
    action: Any        # OrderAction enum; .name is "ADD"|"CANCEL"|"MODIFY"|"FILL"


@runtime_checkable
class P6OrderBookLevel(Protocol):
    """Protocol matching p6-v2 models.OrderBookLevel. Subset we use."""
    price: float
    volume: float
    order_count: int


@runtime_checkable
class P6OrderBookSnapshot(Protocol):
    """Protocol matching p6-v2 models.OrderBookSnapshot. Subset we use."""
    timestamp_ms: int
    symbol: str
    bids: list           # list[P6OrderBookLevel]
    asks: list           # list[P6OrderBookLevel]
    recent_trades: list  # list[P6Order]
    recent_events: list  # list[P6Order]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@dataclass
class L1AdapterConfig:
    """Configuration for ``L1Adapter``."""
    tick_size: float = 0.25
    # How often (in snapshot calls) to trim the history to bounded memory.
    trim_every_n: int = 100


class L1Adapter:
    """Converts p6-v2 ``OrderBookSnapshot`` into L1Snapshot + updates L1History.

    Stateful: holds an ``L1History`` that accumulates across calls.
    Tracks the last-observed best-bid and best-ask prices so that ADD
    events can be classified as passive-bid-add vs passive-ask-add (the
    "at best" condition) against the book state AT EVENT TIME rather
    than the book state observed in the current snapshot.
    """

    def __init__(
        self,
        config: L1AdapterConfig | None = None,
    ) -> None:
        self._cfg = config or L1AdapterConfig()
        self.history = L1History()
        self._last_best_bid: float | None = None
        self._last_best_ask: float | None = None
        self._ingest_count: int = 0
        # Track events already processed (by order_id + timestamp + action)
        # so that repeated snapshots with overlapping recent_events don't
        # double-count. Key = (order_id, timestamp_ms, action_name).
        self._seen_events: set[tuple[str, int, str]] = set()
        self._seen_trades: set[tuple[str, int]] = set()

    # ──────────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────────

    def ingest(self, snapshot: P6OrderBookSnapshot) -> L1Snapshot:
        """Convert a p6-v2 snapshot to L1Snapshot and update history.

        Also:
          - Extracts passive-add events (ADD at best bid / best ask) from
            ``snapshot.recent_events`` and appends to history.
          - Classifies trades in ``snapshot.recent_trades`` as bid/ask
            side and appends to history.
          - Trims history periodically.

        Returns the ``L1Snapshot`` corresponding to this snapshot, also
        appended to history.
        """
        best_bid_px = snapshot.bids[0].price if snapshot.bids else 0.0
        best_ask_px = snapshot.asks[0].price if snapshot.asks else 0.0
        best_bid_sz = snapshot.bids[0].volume if snapshot.bids else 0.0
        best_ask_sz = snapshot.asks[0].volume if snapshot.asks else 0.0

        # Record last trade info (for L1Snapshot fields — optional)
        last_trade_price: float | None = None
        last_trade_size: float | None = None
        last_trade_side: str | None = None
        if snapshot.recent_trades:
            last = snapshot.recent_trades[-1]
            last_trade_price = last.price
            last_trade_size = last.size
            last_trade_side = self._trade_side_from_mid(
                last.price, last.side,
                0.5 * (best_bid_px + best_ask_px),
            )

        l1_snap = L1Snapshot(
            timestamp_ms=snapshot.timestamp_ms,
            best_bid=best_bid_px,
            best_ask=best_ask_px,
            best_bid_size=best_bid_sz,
            best_ask_size=best_ask_sz,
            last_trade_price=last_trade_price,
            last_trade_size=last_trade_size,
            last_trade_side=last_trade_side,
            tick_size=self._cfg.tick_size,
        )

        # Update history IN ORDER so tick-event detection sees the
        # correct "previous mid" when this snapshot arrives.
        self.history.append_snapshot(l1_snap)

        self._ingest_trades(snapshot)
        self._ingest_events(snapshot, best_bid_px, best_ask_px)

        self._last_best_bid = best_bid_px
        self._last_best_ask = best_ask_px

        self._ingest_count += 1
        if self._ingest_count % self._cfg.trim_every_n == 0:
            self.history.trim(snapshot.timestamp_ms)

        return l1_snap

    def reset(self) -> None:
        """Clear all accumulated state. Useful between replay files."""
        self.history = L1History()
        self._last_best_bid = None
        self._last_best_ask = None
        self._ingest_count = 0
        self._seen_events.clear()
        self._seen_trades.clear()

    # ──────────────────────────────────────────────────────────────
    # Event/trade classification
    # ──────────────────────────────────────────────────────────────

    def _ingest_trades(self, snapshot: P6OrderBookSnapshot) -> None:
        """Classify trades using Lee-Ready + trade.side tiebreaker at-mid."""
        for trade in snapshot.recent_trades or []:
            key = (trade.order_id, trade.timestamp_ms)
            if key in self._seen_trades:
                continue
            self._seen_trades.add(key)

            # Use snapshot's current mid as the reference point for
            # classification. Imperfect (trade may have executed before
            # this snapshot's book state), but matches the standard
            # Lee-Ready practice of using the prevailing quote.
            best_bid = self._first_bid(snapshot)
            best_ask = self._first_ask(snapshot)
            mid = 0.5 * (best_bid + best_ask) if best_bid and best_ask else 0.0

            side = self._trade_side_from_mid(trade.price, trade.side, mid)
            self.history.append_trade(
                timestamp_ms=trade.timestamp_ms,
                side=side,
                size=abs(trade.size) if trade.size else 0.0,
            )

    def _ingest_events(
        self,
        snapshot: P6OrderBookSnapshot,
        current_best_bid: float,
        current_best_ask: float,
    ) -> None:
        """Extract passive-add events at best bid / best ask.

        An event qualifies as a passive-add for refresh-rate features if:
          1. ``event.action`` is ADD
          2. ``event.price`` is at the best bid (for bid-add) or best ask
             (for ask-add) PREVAILING at event time (approximated as the
             max of the prior snapshot's best and the current snapshot's
             best — handles the case where the best moves between snaps).
          3. ``event.is_aggressive`` is False (passive only).
        """
        for ev in snapshot.recent_events or []:
            action_name = getattr(ev.action, "name", str(ev.action))
            if action_name != "ADD":
                continue

            # Skip aggressive adds (those cross the spread)
            is_agg = getattr(ev, "is_aggressive", False)
            if is_agg:
                continue

            key = (ev.order_id, ev.timestamp_ms, action_name)
            if key in self._seen_events:
                continue
            self._seen_events.add(key)

            side_name = getattr(ev.side, "name", str(ev.side)).upper()
            # "At best" comparison with small tolerance for float equality
            if side_name == "BID":
                target = max(
                    current_best_bid,
                    self._last_best_bid if self._last_best_bid is not None else current_best_bid,
                )
                if self._approx_eq(ev.price, target):
                    self.history.append_bid_add(ev.timestamp_ms)
            elif side_name == "ASK":
                target = min(
                    current_best_ask,
                    self._last_best_ask if self._last_best_ask is not None else current_best_ask,
                )
                if self._approx_eq(ev.price, target):
                    self.history.append_ask_add(ev.timestamp_ms)

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _first_bid(snapshot: P6OrderBookSnapshot) -> float:
        return snapshot.bids[0].price if snapshot.bids else 0.0

    @staticmethod
    def _first_ask(snapshot: P6OrderBookSnapshot) -> float:
        return snapshot.asks[0].price if snapshot.asks else 0.0

    def _approx_eq(self, a: float, b: float) -> bool:
        """Float equality within half a tick — tolerates FP roundoff."""
        return abs(a - b) < (self._cfg.tick_size * 0.5)

    def _trade_side_from_mid(self, price: float, trade_side: Any, mid: float) -> str:
        """Lee-Ready-ish classification with trade.side as at-mid tiebreaker.

        Returns "bid" = seller hit the bid (seller-initiated), or "ask" =
        buyer lifted the ask (buyer-initiated). Matches the convention
        used by L1 feature [12] ``trade_at_bid_ratio``.
        """
        if price > mid:
            return "ask"  # buyer-initiated
        if price < mid:
            return "bid"  # seller-initiated
        # At mid: use the order's side field. p6-v2 convention is that
        # the side represents the passive (resting) side, so:
        #   resting BID consumed → seller hit the bid → "bid"
        #   resting ASK consumed → buyer lifted the ask → "ask"
        side_name = getattr(trade_side, "name", str(trade_side)).upper()
        return "bid" if side_name == "BID" else "ask"