"""
VPIN — Volume-Synchronized Probability of Informed Trading
Spec §4.4 | OB-reference.md:1625-1644

Feeds l2_features.trade_flow_toxicity and fragility_index.FT.

Two trade-classification methods are supported:
  - Lee-Ready: quote rule (compare trade price to mid) with tick-rule
    fallback for trades exactly at the mid. Requires synchronized quotes.
  - BVC (Bulk Volume Classification): statistical split using the
    standard normal CDF of the standardized price change. Quote-free.

Volume is bucketed into equal-volume bins of size
``bucket_size_fraction × avg_daily_volume``. Trades are split across
bucket boundaries so the totals remain exact.

VPIN over the last ``window_size`` buckets is the mean of
``|buy - sell| / total`` per bucket.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ClassificationMethod(Enum):
    LEE_READY = "lee_ready"
    BVC = "bvc"


@dataclass(frozen=True)
class VolumeBucket:
    bucket_id: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    buy_volume: float
    sell_volume: float
    total_volume: float
    vpin_contribution: float


@dataclass
class VPINConfig:
    bucket_size_fraction: float = 1.0 / 50
    window_size: int = 50
    method: ClassificationMethod = ClassificationMethod.LEE_READY
    avg_daily_volume: float = 0.0


@dataclass
class VPINState:
    buckets: list[VolumeBucket] = field(default_factory=list)
    current_buy_volume: float = 0.0
    current_sell_volume: float = 0.0
    current_bucket_start_ms: int = 0
    bucket_counter: int = 0
    bucket_target_volume: float = 0.0
    last_classification: Literal["buy", "sell"] = "buy"


def _norm_cdf(z: float) -> float:
    """Standard normal CDF using erf (no scipy dep needed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def classify_trade_lee_ready(
    trade_price: float,
    prev_trade_price: float,
    bid_at_trade: float,
    ask_at_trade: float,
) -> Literal["buy", "sell"]:
    """Lee-Ready quote test + tick-rule fallback."""
    if bid_at_trade > 0 and ask_at_trade > 0:
        mid = 0.5 * (bid_at_trade + ask_at_trade)
        if trade_price > mid:
            return "buy"
        if trade_price < mid:
            return "sell"
    # Tick rule fallback
    if trade_price > prev_trade_price:
        return "buy"
    if trade_price < prev_trade_price:
        return "sell"
    # Tied — caller may keep prior; default to "buy"
    return "buy"


def classify_trade_bvc(
    price_change: float,
    volume: float,
    daily_volatility: float,
) -> tuple[float, float]:
    """Bulk Volume Classification — split volume by Φ(Δp / σ)."""
    if daily_volatility <= 0:
        return (0.5 * volume, 0.5 * volume)
    z = price_change / daily_volatility
    buy_frac = _norm_cdf(z)
    return (volume * buy_frac, volume * (1.0 - buy_frac))


def _ensure_target(state: VPINState, config: VPINConfig) -> None:
    if state.bucket_target_volume <= 0:
        if config.avg_daily_volume <= 0:
            raise ValueError("VPINConfig.avg_daily_volume must be > 0")
        state.bucket_target_volume = config.bucket_size_fraction * config.avg_daily_volume


def _finalize_bucket(state: VPINState, end_ms: int) -> VolumeBucket:
    total = state.current_buy_volume + state.current_sell_volume
    bucket = VolumeBucket(
        bucket_id=state.bucket_counter,
        start_timestamp_ms=state.current_bucket_start_ms,
        end_timestamp_ms=end_ms,
        buy_volume=state.current_buy_volume,
        sell_volume=state.current_sell_volume,
        total_volume=total,
        vpin_contribution=(
            abs(state.current_buy_volume - state.current_sell_volume) / total
            if total > 0 else 0.0
        ),
    )
    state.buckets.append(bucket)
    state.bucket_counter += 1
    state.current_buy_volume = 0.0
    state.current_sell_volume = 0.0
    state.current_bucket_start_ms = end_ms
    return bucket


def update_vpin_state(
    state: VPINState,
    config: VPINConfig,
    timestamp_ms: int,
    trade_price: float,
    trade_volume: float,
    side: Literal["buy", "sell"],
) -> VolumeBucket | None:
    """Feed a classified trade into the VPIN state machine.

    Trades that exceed the bucket's remaining capacity are split — the
    head fills the current bucket (which is then finalized), the tail
    becomes the start of the next bucket. Returns the LAST bucket
    finalized in this call (None if none was filled).
    """
    _ensure_target(state, config)
    if state.current_bucket_start_ms == 0:
        state.current_bucket_start_ms = timestamp_ms
    state.last_classification = side

    last_finalized: VolumeBucket | None = None
    remaining = trade_volume
    while remaining > 0:
        used = state.current_buy_volume + state.current_sell_volume
        capacity = state.bucket_target_volume - used
        if remaining < capacity:
            if side == "buy":
                state.current_buy_volume += remaining
            else:
                state.current_sell_volume += remaining
            remaining = 0.0
        else:
            # Fill exactly the remaining capacity, finalize, continue
            if side == "buy":
                state.current_buy_volume += capacity
            else:
                state.current_sell_volume += capacity
            remaining -= capacity
            last_finalized = _finalize_bucket(state, timestamp_ms)
    return last_finalized


def compute_vpin(state: VPINState, config: VPINConfig) -> float | None:
    """VPIN over the last ``config.window_size`` buckets, or None if too few."""
    if len(state.buckets) < config.window_size:
        return None
    window = state.buckets[-config.window_size:]
    return sum(b.vpin_contribution for b in window) / len(window)
