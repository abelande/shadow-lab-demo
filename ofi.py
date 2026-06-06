"""Order Flow Imbalance (OFI) + VPIN tracker.

Implements the price-matched OFI formula from Cont, Kukanov & Stoikov (2014)
and Volume-Synchronized Probability of Informed Trading (VPIN) with the
Chakrabarty, Li, Nguyen & Van Ness (2007) hybrid trade classification
algorithm (SSRN 958178).

The hybrid outperforms naive Lee-Ready by ~2% overall and ~5% for trades
inside the quotes. It uses a spread-decile approach: quote rule near the
quotes (outer 30% of spread), tick rule near the midpoint (inner 40%),
and tick rule for trades at or outside the quotes.

This is a ground-up implementation against p6-v2 types, NOT a port of
p4-clones/mesh/ofi.py (which has critical bugs: index-based level matching,
incomplete size accounting on price shifts, missing trade classification).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .models import OrderBookSnapshot, OrderBookLevel, Side


@dataclass(frozen=True)
class OFIConfig:
    n_levels: int = 10              # depth levels to track
    ema_span: int = 20              # EMA smoothing period
    vpin_bucket_size: float = 500.0 # volume per VPIN bucket (contracts)
    vpin_lookback: int = 10         # rolling window of buckets for VPIN


class OFITracker:
    """Stateful OFI + VPIN computation across sequential snapshots.

    OFI matches levels by price (not by array index) across consecutive
    snapshots, correctly handling level appearance, disappearance, and
    price shifts. Each price's delta is proximity-weighted: levels closer
    to best bid/ask contribute more.

    VPIN classifies trades via the Chakrabarty et al. (2007) hybrid
    algorithm and accumulates buy/sell volume into fixed-size buckets.
    """

    def __init__(self, config: Optional[OFIConfig] = None) -> None:
        self._cfg = config or OFIConfig()
        self._alpha = 2.0 / (self._cfg.ema_span + 1)

        # OFI state
        self._prev_bid_map: dict[float, float] = {}
        self._prev_ask_map: dict[float, float] = {}
        self._prev_best_bid: Optional[float] = None
        self._prev_best_ask: Optional[float] = None
        self._ema_ofi: float = 0.0
        self._has_prev: bool = False

        # VPIN state
        self._vpin_buy_vol: float = 0.0
        self._vpin_sell_vol: float = 0.0
        self._vpin_total_vol: float = 0.0
        self._vpin_buckets: deque[float] = deque(maxlen=self._cfg.vpin_lookback)
        self._prev_trade_price: Optional[float] = None  # for tick rule

    def update(self, snapshot: OrderBookSnapshot) -> tuple[float, float]:
        """Compute OFI and VPIN from the current snapshot.

        Returns (ofi_ema, vpin). On the first call (no previous snapshot),
        returns (0.0, 0.0).
        """
        ofi = self._compute_ofi(snapshot)
        vpin = self._compute_vpin(snapshot)
        return ofi, vpin

    # ── OFI ────────────────────────────────────────────────────────

    def _compute_ofi(self, snapshot: OrderBookSnapshot) -> float:
        n = self._cfg.n_levels
        curr_bid_map = {
            lv.price: lv.volume for lv in snapshot.bids[:n]
        }
        curr_ask_map = {
            lv.price: lv.volume for lv in snapshot.asks[:n]
        }

        if not self._has_prev:
            self._prev_bid_map = curr_bid_map
            self._prev_ask_map = curr_ask_map
            self._prev_best_bid = snapshot.best_bid
            self._prev_best_ask = snapshot.best_ask
            self._has_prev = True
            return 0.0

        best_bid = snapshot.best_bid or 0.0

        # Bid-side OFI: positive delta = buy pressure increasing
        bid_ofi = self._side_ofi(
            curr_map=curr_bid_map,
            prev_map=self._prev_bid_map,
            best_price=best_bid,
            descending=True,
        )

        best_ask = snapshot.best_ask or 0.0

        # Ask-side OFI: positive delta = sell pressure increasing (subtracted)
        ask_ofi = self._side_ofi(
            curr_map=curr_ask_map,
            prev_map=self._prev_ask_map,
            best_price=best_ask,
            descending=False,
        )

        raw_ofi = bid_ofi - ask_ofi

        # EMA smoothing
        self._ema_ofi = self._alpha * raw_ofi + (1.0 - self._alpha) * self._ema_ofi

        # Store for next call
        self._prev_bid_map = curr_bid_map
        self._prev_ask_map = curr_ask_map
        self._prev_best_bid = snapshot.best_bid
        self._prev_best_ask = snapshot.best_ask

        return self._ema_ofi

    @staticmethod
    def _side_ofi(
        curr_map: dict[float, float],
        prev_map: dict[float, float],
        best_price: float,
        descending: bool,
    ) -> float:
        """Compute proximity-weighted volume delta for one side.

        Matches levels by price across snapshots. Levels that appear
        contribute their full volume as positive delta; levels that
        disappear contribute their full volume as negative delta.
        """
        all_prices = sorted(set(curr_map) | set(prev_map), reverse=descending)
        ofi = 0.0
        for rank, price in enumerate(all_prices):
            curr_vol = curr_map.get(price, 0.0)
            prev_vol = prev_map.get(price, 0.0)
            delta = curr_vol - prev_vol
            # Proximity weight: rank 0 (closest to best) gets weight 1.0,
            # rank 1 gets 0.5, rank 2 gets 0.33, etc.
            weight = 1.0 / (1.0 + rank)
            ofi += weight * delta
        return ofi

    # ── VPIN ───────────────────────────────────────────────────────

    def _compute_vpin(self, snapshot: OrderBookSnapshot) -> float:
        """VPIN with Chakrabarty et al. (2007) hybrid trade classification.

        Spread-decile algorithm: uses quote rule near the quotes (outer 30%
        of spread), tick rule near the midpoint (inner 40%), and tick rule
        for trades at/outside the quotes.
        """
        mid = snapshot.mid_price
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        if mid is None or best_bid is None or best_ask is None:
            return self._current_vpin()

        spread = best_ask - best_bid

        for trade in (snapshot.recent_trades or []):
            size = abs(trade.size) if trade.size else 0.0
            if size <= 0:
                continue

            is_buy = self._classify_trade(
                trade_price=trade.price,
                trade_side=trade.side,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
            )

            if is_buy:
                self._vpin_buy_vol += size
            else:
                self._vpin_sell_vol += size

            self._prev_trade_price = trade.price
            self._vpin_total_vol += size

            # Check if bucket is full
            if self._vpin_total_vol >= self._cfg.vpin_bucket_size:
                total = max(self._vpin_total_vol, 1e-10)
                bucket_val = abs(self._vpin_buy_vol - self._vpin_sell_vol) / total
                self._vpin_buckets.append(bucket_val)
                self._vpin_buy_vol = 0.0
                self._vpin_sell_vol = 0.0
                self._vpin_total_vol = 0.0

        return self._current_vpin()

    def _classify_trade(
        self,
        trade_price: float,
        trade_side: Side,
        best_bid: float,
        best_ask: float,
        spread: float,
    ) -> bool:
        """Classify a trade as buyer-initiated (True) or seller-initiated (False).

        Implements the Chakrabarty, Li, Nguyen & Van Ness (2007) hybrid
        algorithm (SSRN 958178, Figure 1):

        Zone 1: trade_price >= best_ask → at/above ask.
                Tick rule; fallback = buy (quote rule).
        Zone 2: trade_price <= best_bid → at/below bid.
                Tick rule; fallback = sell (quote rule).
        Zone 3: Inside quotes, upper 30% of spread (near ask).
                Quote rule → buy.
        Zone 4: Inside quotes, lower 30% of spread (near bid).
                Quote rule → sell.
        Zone 5: Inside quotes, middle 40% of spread (near midpoint).
                Tick rule; fallback = trade.side inference.
        """
        # Zones 1 & 2: at or outside the quotes → tick rule
        if trade_price >= best_ask:
            return self._tick_rule(trade_price, fallback_buy=True)
        if trade_price <= best_bid:
            return self._tick_rule(trade_price, fallback_buy=False)

        # Inside the quotes: compute relative position in [0, 1]
        # 0 = at bid, 1 = at ask
        if spread > 1e-12:
            relative_pos = (trade_price - best_bid) / spread
        else:
            # Zero spread — use trade.side as tiebreaker
            return trade_side == Side.ASK

        # Zone 3: upper 30% of spread (>= 0.70) → quote rule → buy
        if relative_pos >= 0.70:
            return True

        # Zone 4: lower 30% of spread (<= 0.30) → quote rule → sell
        if relative_pos <= 0.30:
            return False

        # Zone 5: middle 40% (0.30 < relative_pos < 0.70) → tick rule
        return self._tick_rule(trade_price, fallback_buy=(trade_side == Side.ASK))

    def _tick_rule(self, trade_price: float, fallback_buy: bool) -> bool:
        """Tick rule: buy if price > prev trade, sell if price < prev trade.

        Falls back to ``fallback_buy`` when no previous trade price exists
        (first trade of the session or after reset).
        """
        if self._prev_trade_price is None:
            return fallback_buy
        if trade_price > self._prev_trade_price:
            return True   # uptick → buy
        if trade_price < self._prev_trade_price:
            return False  # downtick → sell
        # Zero tick — use fallback
        return fallback_buy

    def _current_vpin(self) -> float:
        if not self._vpin_buckets:
            return 0.0
        return sum(self._vpin_buckets) / len(self._vpin_buckets)

    def reset(self) -> None:
        """Clear all state. Useful between replay files."""
        self._prev_bid_map.clear()
        self._prev_ask_map.clear()
        self._prev_best_bid = None
        self._prev_best_ask = None
        self._ema_ofi = 0.0
        self._has_prev = False
        self._vpin_buy_vol = 0.0
        self._vpin_sell_vol = 0.0
        self._vpin_total_vol = 0.0
        self._vpin_buckets.clear()
        self._prev_trade_price = None
