"""LevelTracker: stateful level lifecycle tracker across OrderBookSnapshots.

Maintains per-level state across time, scoring significance and advancing
through lifecycle states: FORMING → RESTING → TESTED → DEFENDED/BROKEN/PULLED.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    InstrumentVisualConfig,
    LevelLifecycle,
    LevelState,
    Order,
    OrderAction,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    SpoofEvent,
    SpoofType,
)

# Price key type: (price, side)
_LevelKey = Tuple[float, Side]

# Number of volume history samples to retain per level
_VOLUME_HISTORY_LEN = 20

# How long (ms) to keep BROKEN/PULLED levels for fade-out animation
_FADE_DURATION_MS = 15_000

# Volume drop threshold to trigger TESTED → BROKEN (fraction of peak)
_BROKEN_VOLUME_FRACTION = 0.25

# Volume retained threshold to trigger TESTED → DEFENDED (fraction of peak)
_DEFENDED_VOLUME_FRACTION = 0.60

# Min fills at a level before it can be DEFENDED
_MIN_FILLS_FOR_DEFENDED = 1

# Min refills before iceberg is suspected
_MIN_REFILLS_FOR_ICEBERG = 2


def _round_number_bonus(price: float, cfg: InstrumentVisualConfig) -> float:
    """Return 0-1 bonus for levels at significant round numbers."""
    step = cfg.round_number_step
    if step <= 0:
        return 0.0
    remainder = price % step
    if remainder < cfg.tick_size or (step - remainder) < cfg.tick_size:
        return 1.0
    half = step / 2.0
    if abs(remainder - half) < cfg.tick_size:
        return 0.4
    return 0.0


def _recurrence_bonus(key: _LevelKey, historical: Set[_LevelKey]) -> float:
    """Return 0.0 or 1.0 based on whether this level has appeared before."""
    return 1.0 if key in historical else 0.0


def _compute_significance(
    price: float,
    side: Side,
    volume: float,
    age_ms: int,
    order_count: int,
    cfg: InstrumentVisualConfig,
    historical: Set[_LevelKey],
) -> float:
    vol_score = min(1.0, volume / max(1.0, cfg.significant_volume))
    age_score = min(1.0, age_ms / max(1, cfg.significant_age_ms))
    ord_score = min(1.0, order_count / max(1, cfg.significant_order_count))
    rnd_score = _round_number_bonus(price, cfg)
    rec_score = _recurrence_bonus((price, side), historical)
    return (
        0.30 * vol_score
        + 0.25 * age_score
        + 0.20 * ord_score
        + 0.15 * rnd_score
        + 0.10 * rec_score
    )


def _price_at_level(price: float, side: Side, best_bid: Optional[float], best_ask: Optional[float], tick: float) -> bool:
    """Return True if the market price is touching this level."""
    if side == Side.ASK and best_bid is not None:
        return abs(best_bid - price) <= tick
    if side == Side.BID and best_ask is not None:
        return abs(best_ask - price) <= tick
    return False


def _advance_lifecycle(
    state: LevelState,
    new_vol: float,
    fill_vol: float,
    cancel_vol: float,
    best_bid: Optional[float],
    best_ask: Optional[float],
    age_ms: int,
    cfg: InstrumentVisualConfig,
) -> LevelLifecycle:
    """Advance lifecycle state given current market conditions."""
    lc = state.lifecycle
    tick = cfg.tick_size

    touching = _price_at_level(state.price, state.side, best_bid, best_ask, tick)

    if lc == LevelLifecycle.FORMING:
        if age_ms >= cfg.significant_age_ms:
            return LevelLifecycle.RESTING
        return LevelLifecycle.FORMING

    if lc == LevelLifecycle.RESTING:
        if touching and fill_vol > 0:
            return LevelLifecycle.TESTED
        return LevelLifecycle.RESTING

    if lc == LevelLifecycle.TESTED:
        if state.fill_count >= _MIN_FILLS_FOR_DEFENDED:
            if new_vol >= state.peak_volume * _DEFENDED_VOLUME_FRACTION:
                return LevelLifecycle.DEFENDED
            if new_vol <= state.peak_volume * _BROKEN_VOLUME_FRACTION:
                return LevelLifecycle.BROKEN
        return LevelLifecycle.TESTED

    if lc == LevelLifecycle.DEFENDED:
        if touching and new_vol <= state.peak_volume * _BROKEN_VOLUME_FRACTION:
            return LevelLifecycle.BROKEN
        return LevelLifecycle.DEFENDED

    # BROKEN / PULLED stay as-is until removed
    return lc


class LevelTracker:
    """Maintains stateful level lifecycles across OrderBookSnapshot updates.

    Usage::

        tracker = LevelTracker(InstrumentVisualConfig.for_symbol("NQ"))
        levels = tracker.update(snapshot, spoof_events=[], authenticity_score=1.0)
    """

    def __init__(self, cfg: Optional[InstrumentVisualConfig] = None) -> None:
        self._cfg = cfg or InstrumentVisualConfig.for_symbol("NQ")
        # Active level states keyed by (price, side)
        self._levels: Dict[_LevelKey, LevelState] = {}
        # Prices that have been seen before (for recurrence bonus)
        self._historical: Set[_LevelKey] = set()

    @property
    def config(self) -> InstrumentVisualConfig:
        return self._cfg

    def update(
        self,
        snapshot: OrderBookSnapshot,
        spoof_events: Optional[List[SpoofEvent]] = None,
        authenticity_score: float = 1.0,
    ) -> List[LevelState]:
        """Update level states from a new snapshot.

        Returns only levels with significance > 0.3.
        """
        now_ms = snapshot.timestamp_ms
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        spoof_events = spoof_events or []

        # --- Build indexes from snapshot ---
        current: Dict[_LevelKey, OrderBookLevel] = {}
        for lvl in snapshot.bids:
            current[(lvl.price, Side.BID)] = lvl
        for lvl in snapshot.asks:
            current[(lvl.price, Side.ASK)] = lvl

        fills_by_price: Dict[_LevelKey, float] = defaultdict(float)
        cancels_by_price: Dict[_LevelKey, float] = defaultdict(float)
        for event in (snapshot.recent_events or []):
            key: _LevelKey = (event.price, event.side)
            if event.action == OrderAction.FILL:
                fills_by_price[key] += event.size
            elif event.action == OrderAction.CANCEL:
                cancels_by_price[key] += event.size

        spoof_index: Dict[_LevelKey, SpoofEvent] = {}
        for se in spoof_events:
            spoof_index[(se.price, se.side)] = se

        # --- Update existing levels ---
        to_remove: List[_LevelKey] = []
        updated: Dict[_LevelKey, LevelState] = {}

        for key, state in self._levels.items():
            price, side = key

            if key in current:
                book_level = current[key]
                fill_vol = fills_by_price.get(key, 0.0)
                cancel_vol = cancels_by_price.get(key, 0.0)
                new_vol = book_level.volume
                old_vol = state.volume
                age_ms = now_ms - state.first_seen_ms

                # Refill detection: price was partially filled but volume recovered
                refill_count = state.refill_count
                if fill_vol > 0 and new_vol >= old_vol * 0.70 and old_vol > 0:
                    refill_count += 1

                fill_count = state.fill_count + (1 if fill_vol > 0 else 0)
                iceberg_suspected = state.iceberg_suspected or (refill_count >= _MIN_REFILLS_FOR_ICEBERG)

                volume_history = (state.volume_history + [new_vol])[-_VOLUME_HISTORY_LEN:]
                peak_volume = max(state.peak_volume, new_vol)

                new_lc = _advance_lifecycle(
                    state=state,
                    new_vol=new_vol,
                    fill_vol=fill_vol,
                    cancel_vol=cancel_vol,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    age_ms=age_ms,
                    cfg=self._cfg,
                )

                significance = _compute_significance(
                    price, side, new_vol, age_ms,
                    book_level.order_count, self._cfg, self._historical,
                )

                spoof_ev = spoof_index.get(key)
                spoof_type = spoof_ev.spoof_type if spoof_ev else state.spoof_type
                auth = (1.0 - spoof_ev.confidence) if spoof_ev else state.authenticity

                updated[key] = replace(
                    state,
                    volume=new_vol,
                    peak_volume=peak_volume,
                    order_count=book_level.order_count,
                    lifecycle=new_lc,
                    last_seen_ms=now_ms,
                    age_ms=age_ms,
                    significance=significance,
                    authenticity=auth,
                    spoof_type=spoof_type,
                    iceberg_suspected=iceberg_suspected,
                    volume_history=volume_history,
                    fill_count=fill_count,
                    refill_count=refill_count,
                )

            else:
                # Level no longer in book
                age_ms = now_ms - state.first_seen_ms

                if state.lifecycle in (LevelLifecycle.BROKEN, LevelLifecycle.PULLED):
                    # Keep for fade-out period
                    if (now_ms - state.last_seen_ms) >= _FADE_DURATION_MS:
                        self._historical.add(key)
                        to_remove.append(key)
                    else:
                        updated[key] = state
                    continue

                if state.lifecycle == LevelLifecycle.FORMING:
                    # Never matured — just remove
                    to_remove.append(key)
                    continue

                # Was RESTING/TESTED/DEFENDED — determine exit reason
                fill_vol = fills_by_price.get(key, 0.0)
                if fill_vol > 0:
                    new_lc = LevelLifecycle.BROKEN
                else:
                    new_lc = LevelLifecycle.PULLED

                self._historical.add(key)
                updated[key] = replace(
                    state,
                    lifecycle=new_lc,
                    last_seen_ms=now_ms,
                    age_ms=age_ms,
                )

        # Remove stale keys
        for key in to_remove:
            updated.pop(key, None)

        # --- Add new levels ---
        for key, book_level in current.items():
            if key not in self._levels and key not in updated:
                price, side = key
                spoof_ev = spoof_index.get(key)
                spoof_type = spoof_ev.spoof_type if spoof_ev else None
                auth = (1.0 - spoof_ev.confidence) if spoof_ev else authenticity_score

                updated[key] = LevelState(
                    price=price,
                    side=side,
                    volume=book_level.volume,
                    peak_volume=book_level.volume,
                    order_count=book_level.order_count,
                    lifecycle=LevelLifecycle.FORMING,
                    first_seen_ms=now_ms,
                    last_seen_ms=now_ms,
                    age_ms=0,
                    significance=0.0,
                    authenticity=auth,
                    spoof_type=spoof_type,
                    iceberg_suspected=False,
                    volume_history=[book_level.volume],
                    fill_count=0,
                    refill_count=0,
                )

        self._levels = updated

        # Return levels above significance threshold
        return [s for s in self._levels.values() if s.significance > 0.3]

    def get_all_levels(self) -> List[LevelState]:
        """Return all tracked levels regardless of significance threshold."""
        return list(self._levels.values())

    def reset(self) -> None:
        """Clear all tracked levels."""
        self._historical.update(self._levels.keys())
        self._levels.clear()
