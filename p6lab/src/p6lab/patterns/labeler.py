"""
Forward Outcome Labeler
Spec §5.3 | OB-reference.md:805-809

Labels pattern instances with forward returns at 1m/5m/15m/1h.
Classification: continuation (>+0.5 ATR), reversal (<-0.5 ATR), neutral.
"""
from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

logger = logging.getLogger(__name__)


ATR_THRESHOLD = 0.5  # per OB-reference.md:805-809

HORIZONS = ["1m", "5m", "15m", "1h"]
HORIZON_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
}


class OutcomeClass(str, Enum):
    CONTINUATION = "continuation"  # > +0.5 ATR
    REVERSAL = "reversal"          # < -0.5 ATR
    NEUTRAL = "neutral"            # between
    INCOMPLETE = "incomplete"      # session ended before horizon


@dataclass(frozen=True)
class PatternOutcome:
    """Forward outcome for a single pattern instance at one horizon."""
    horizon: str
    raw_return_ticks: float
    atr_normalized_return: float
    classification: OutcomeClass
    pattern_timestamp_ms: int
    outcome_timestamp_ms: int | None  # None if incomplete


@dataclass(frozen=True)
class MultiHorizonOutcome:
    """All horizon outcomes for one pattern instance."""
    pattern_timestamp_ms: int
    symbol: str
    pattern_id: str
    outcomes: dict[str, PatternOutcome]  # keyed by horizon string


def classify_outcome(
    atr_normalized_return: float,
    direction: Literal["long", "short"] = "long",
    threshold: float = ATR_THRESHOLD,
) -> OutcomeClass:
    """
    Classify forward return into continuation/reversal/neutral.

    For long signals: > +threshold = continuation, < -threshold = reversal.
    For short signals: signs reversed (i.e. downward move is continuation).
    """
    if math.isnan(atr_normalized_return):
        return OutcomeClass.NEUTRAL
    signed = atr_normalized_return if direction == "long" else -atr_normalized_return
    if signed > threshold:
        return OutcomeClass.CONTINUATION
    if signed < -threshold:
        return OutcomeClass.REVERSAL
    return OutcomeClass.NEUTRAL


def _extract_price_series(event_stream: list) -> tuple[list[int], list[float]]:
    """Extract (ts, price) pairs from events, sorted by timestamp.

    Accepts events with .timestamp_ms and .price attributes OR (ts, price) tuples.
    Events without a numeric price are skipped.
    """
    ts_list: list[int] = []
    px_list: list[float] = []
    for e in event_stream:
        if isinstance(e, tuple) and len(e) == 2:
            ts, px = e
        else:
            ts = getattr(e, "timestamp_ms", None)
            px = getattr(e, "price", None)
        if ts is None or px is None:
            continue
        try:
            px_f = float(px)
        except (TypeError, ValueError):
            continue
        if math.isnan(px_f):
            continue
        ts_list.append(int(ts))
        px_list.append(px_f)
    # Ensure sorted (stable) — callers often pass already-sorted streams
    if ts_list and any(ts_list[i] > ts_list[i + 1] for i in range(len(ts_list) - 1)):
        order = sorted(range(len(ts_list)), key=lambda i: ts_list[i])
        ts_list = [ts_list[i] for i in order]
        px_list = [px_list[i] for i in order]
    return ts_list, px_list


def _price_at_or_before(ts_list: list[int], px_list: list[float], target_ms: int) -> float | None:
    """Return the last observed price at-or-before target_ms, or None."""
    if not ts_list:
        return None
    idx = bisect.bisect_right(ts_list, target_ms) - 1
    if idx < 0:
        return None
    return px_list[idx]


def label_pattern_instance(
    event_stream: list,  # forward events from pattern timestamp
    pattern_timestamp_ms: int,
    pattern_direction: Literal["long", "short"],
    instrument_atr: float,
    tick_size: float,
    session_end_ms: int | None = None,
    symbol: str = "",
    pattern_id: str = "",
    *,
    _price_series: tuple[list[int], list[float]] | None = None,
) -> MultiHorizonOutcome:
    """
    Given a pattern instance at time t, walk forward and compute
    outcomes at each horizon. If session ends before a horizon,
    mark that horizon as INCOMPLETE (exclude from hit-rate).

    `instrument_atr` is in price units (not ticks).

    Pass `_price_series=(ts_list, px_list)` to reuse a pre-extracted
    price series across many calls — batch callers (miner) should do
    this to avoid O(E) re-extraction per pattern instance.
    """
    if instrument_atr <= 0:
        raise ValueError(f"instrument_atr must be positive, got {instrument_atr}")
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")

    if _price_series is not None:
        ts_list, px_list = _price_series
    else:
        ts_list, px_list = _extract_price_series(event_stream)
    entry_px = _price_at_or_before(ts_list, px_list, pattern_timestamp_ms)
    if entry_px is None:
        # No prior price — use first forward price as entry
        if px_list:
            entry_px = px_list[0]
        else:
            # Nothing usable → all horizons incomplete
            return MultiHorizonOutcome(
                pattern_timestamp_ms=pattern_timestamp_ms,
                symbol=symbol, pattern_id=pattern_id,
                outcomes={
                    h: PatternOutcome(
                        horizon=h, raw_return_ticks=float("nan"),
                        atr_normalized_return=float("nan"),
                        classification=OutcomeClass.INCOMPLETE,
                        pattern_timestamp_ms=pattern_timestamp_ms,
                        outcome_timestamp_ms=None,
                    ) for h in HORIZONS
                },
            )

    outcomes: dict[str, PatternOutcome] = {}
    for horizon in HORIZONS:
        target = pattern_timestamp_ms + HORIZON_MS[horizon]
        # Session boundary: incomplete if horizon exceeds session
        if session_end_ms is not None and target > session_end_ms:
            outcomes[horizon] = PatternOutcome(
                horizon=horizon, raw_return_ticks=float("nan"),
                atr_normalized_return=float("nan"),
                classification=OutcomeClass.INCOMPLETE,
                pattern_timestamp_ms=pattern_timestamp_ms,
                outcome_timestamp_ms=None,
            )
            continue
        # No price observed at/after target in the stream → incomplete
        last_ts = ts_list[-1] if ts_list else pattern_timestamp_ms
        if last_ts < target:
            outcomes[horizon] = PatternOutcome(
                horizon=horizon, raw_return_ticks=float("nan"),
                atr_normalized_return=float("nan"),
                classification=OutcomeClass.INCOMPLETE,
                pattern_timestamp_ms=pattern_timestamp_ms,
                outcome_timestamp_ms=None,
            )
            continue
        exit_px = _price_at_or_before(ts_list, px_list, target)
        # exit_px must exist because last_ts >= target and entry_px existed
        assert exit_px is not None
        raw_delta_price = exit_px - entry_px
        raw_return_ticks = raw_delta_price / tick_size
        atr_normalized = raw_delta_price / instrument_atr
        classification = classify_outcome(atr_normalized, pattern_direction)
        outcomes[horizon] = PatternOutcome(
            horizon=horizon,
            raw_return_ticks=raw_return_ticks,
            atr_normalized_return=atr_normalized,
            classification=classification,
            pattern_timestamp_ms=pattern_timestamp_ms,
            outcome_timestamp_ms=target,
        )

    return MultiHorizonOutcome(
        pattern_timestamp_ms=pattern_timestamp_ms,
        symbol=symbol, pattern_id=pattern_id,
        outcomes=outcomes,
    )


def compute_outcome_statistics(
    outcomes: list[MultiHorizonOutcome],
    horizon: str,
) -> dict:
    """
    Aggregate stats for a set of outcomes at one horizon.

    Returns: {mean_atr, std, hit_rate, n}. INCOMPLETE outcomes are
    excluded from all statistics.
    hit_rate = fraction classified CONTINUATION among non-incomplete.
    """
    if horizon not in HORIZON_MS:
        raise ValueError(f"Unknown horizon: {horizon}. Expected one of {HORIZONS}")

    atrs: list[float] = []
    n_continuation = 0
    for mh in outcomes:
        po = mh.outcomes.get(horizon)
        if po is None or po.classification == OutcomeClass.INCOMPLETE:
            continue
        if math.isnan(po.atr_normalized_return):
            continue
        atrs.append(po.atr_normalized_return)
        if po.classification == OutcomeClass.CONTINUATION:
            n_continuation += 1

    n = len(atrs)
    if n == 0:
        return {"mean_atr": float("nan"), "std": float("nan"),
                "hit_rate": float("nan"), "n": 0}
    mean = sum(atrs) / n
    var = sum((x - mean) ** 2 for x in atrs) / n
    std = math.sqrt(var)
    return {
        "mean_atr": mean,
        "std": std,
        "hit_rate": n_continuation / n,
        "n": n,
    }
