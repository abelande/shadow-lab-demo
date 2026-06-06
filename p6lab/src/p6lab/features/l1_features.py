"""
p6lab.features.l1_features — 16-feature L1 feature set.

Spec: p6-notebook-lab-spec.md §4.1
Ref:  OB-reference.md L830-847

All 16 features per spec §4.1 table:

 #  Name                    Update       Formula
 1  spread_ticks            every L1     (best_ask - best_bid) / tick_size
 2  spread_bps_l1           every L1     spread / mid × 10000
 3  best_bid_size           every L1     raw
 4  best_ask_size           every L1     raw
 5  top_imbalance           every L1     (bid_sz - ask_sz) / (bid_sz + ask_sz)
 6  bid_refresh_rate        100ms roll   passive adds at best bid per second
 7  ask_refresh_rate        100ms roll   passive adds at best ask per second
 8  bid_retreat_velocity    250ms roll   negative Δ best_bid per second
 9  ask_advance_velocity    250ms roll   positive Δ best_ask per second
10  spread_compression_rate 250ms roll   Δ spread per second (negative = tightening)
11  tick_direction_streak   every L1     consecutive same-sign Δ mid
12  tick_acceleration       500ms roll   d²(tick_count)/dt²
13  trade_at_bid_ratio      1s roll      trades_at_bid / total_trades
14  size_spike_ratio        1s roll      max_trade_size / median_trade_size
15  microprice_velocity     250ms roll   Δ microprice per second
16  l1_shape_vector         every L1     [bid_sz, ask_sz, spread_bps, imbalance] normalized

Exports:
  compute_l1_features(snapshot, history) → np.ndarray[16]
    Per-snapshot use — called from web server at replay time.
  compute_l1_series(snapshots) → pd.DataFrame
    Bulk backtesting — called from notebooks 03-07.
  bid_ask_imbalance_baseline(snapshots) → pd.Series
    Baseline feature for information-gain gate (§9.1).

Notebook 03 §08 gate:
  Each feature must beat bid_ask_imbalance baseline by ≥2% AUC
  (spec §9.1, OB-reference L830-847).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

L1_FEATURE_DIM: int = 19

# Rolling window durations in MILLISECONDS (spec §4.1).
# Windows use timestamp-based filtering, not fixed sample counts — this
# makes the math correct regardless of snapshot cadence (varies with
# market activity) and lets callers pass sparse snapshots.
ROLL_100MS: int = 100
ROLL_250MS: int = 250
ROLL_500MS: int = 500
ROLL_1S: int = 1000

# Information-gain gate threshold (spec §9.1, §8.3)
MIN_AUC_IMPROVEMENT_OVER_BASELINE: float = 0.02   # ≥2% above bid_ask_imbalance

# l1_shape_vector projection weights — chosen to emphasize features
# not already represented verbatim elsewhere in the vector. Since size
# and spread appear standalone, the composite is weighted toward imbalance
# and normalized spread-in-bps interactions.
_L1_SHAPE_WEIGHTS = np.array([0.25, 0.25, 0.2, 0.3], dtype=np.float64)


class L1FeatureNames:
    """Canonical feature names (indices match the 19-dim array). Spec §4.1."""
    SPREAD_TICKS = "spread_ticks"                      # index 0
    SPREAD_BPS_L1 = "spread_bps_l1"                    # index 1
    BEST_BID_SIZE = "best_bid_size"                    # index 2
    BEST_ASK_SIZE = "best_ask_size"                    # index 3
    TOP_IMBALANCE = "top_imbalance"                    # index 4
    BID_REFRESH_RATE = "bid_refresh_rate"              # index 5
    ASK_REFRESH_RATE = "ask_refresh_rate"              # index 6
    BID_RETREAT_VELOCITY = "bid_retreat_velocity"      # index 7
    ASK_ADVANCE_VELOCITY = "ask_advance_velocity"      # index 8
    SPREAD_COMPRESSION_RATE = "spread_compression_rate"  # index 9
    TICK_DIRECTION_STREAK = "tick_direction_streak"    # index 10
    TICK_ACCELERATION = "tick_acceleration"            # index 11
    TRADE_AT_BID_RATIO = "trade_at_bid_ratio"          # index 12
    SIZE_SPIKE_RATIO = "size_spike_ratio"              # index 13
    MICROPRICE_VELOCITY = "microprice_velocity"        # index 14
    L1_SHAPE_VECTOR = "l1_shape_vector"                # index 15 (composite scalar — kept for back-compat)
    # Critique §1.2: the scalar above is a lossy projection of the 4-dim
    # shape vector. Expand the orthogonal information (unit-normalized
    # components) as three additional columns. The spread_bps unit is
    # omitted because spread_bps_l1 at [1] covers it.
    L1_SHAPE_BID_UNIT = "l1_shape_bid_unit"            # index 16
    L1_SHAPE_ASK_UNIT = "l1_shape_ask_unit"            # index 17
    L1_SHAPE_IMB_UNIT = "l1_shape_imb_unit"            # index 18

    ALL: list[str] = [
        SPREAD_TICKS, SPREAD_BPS_L1, BEST_BID_SIZE, BEST_ASK_SIZE,
        TOP_IMBALANCE, BID_REFRESH_RATE, ASK_REFRESH_RATE,
        BID_RETREAT_VELOCITY, ASK_ADVANCE_VELOCITY, SPREAD_COMPRESSION_RATE,
        TICK_DIRECTION_STREAK, TICK_ACCELERATION, TRADE_AT_BID_RATIO,
        SIZE_SPIKE_RATIO, MICROPRICE_VELOCITY, L1_SHAPE_VECTOR,
        L1_SHAPE_BID_UNIT, L1_SHAPE_ASK_UNIT, L1_SHAPE_IMB_UNIT,
    ]


# ---------------------------------------------------------------------------
# Snapshot and history types
# ---------------------------------------------------------------------------

@dataclass
class L1Snapshot:
    """Single L1 snapshot. Expected shape from OrderBookMetaPipeline.

    Fields align with p6-v2 OrderBookSnapshot (see _l1_adapter.py for the
    adapter). The ``last_trade_*`` fields are optional because not every
    L1 update coincides with a trade.
    """
    timestamp_ms: int
    best_bid: float
    best_ask: float
    best_bid_size: float
    best_ask_size: float
    last_trade_price: float | None = None
    last_trade_size: float | None = None
    last_trade_side: str | None = None   # "bid" | "ask"
    tick_size: float = 0.25              # default NQ/ES tick

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def microprice(self) -> float:
        """Size-weighted mid: `(bid × ask_sz + ask × bid_sz) / (bid_sz + ask_sz)`.

        Pulls toward whichever side has more volume. Reduces to `mid` when
        sizes are equal; at zero total size falls back to arithmetic mid.
        """
        total = self.best_bid_size + self.best_ask_size
        if total <= 0:
            return self.mid
        return (self.best_bid * self.best_ask_size
                + self.best_ask * self.best_bid_size) / total


@dataclass
class L1History:
    """Rolling history of recent snapshots + trade events.

    The caller maintains this state across successive ``compute_l1_features``
    calls. ``trim(now_ms)`` drops entries older than ``ROLL_1S``
    (the longest window used by any feature) to bound memory.

    Attributes are plain lists (not deques) so vectorized/bulk code paths
    can slice them directly with numpy without conversion overhead.
    """
    snapshots: list[L1Snapshot] = field(default_factory=list)
    # Trade events: parallel lists keyed by event timestamp
    trade_timestamps_ms: list[int] = field(default_factory=list)
    trade_sides: list[str] = field(default_factory=list)   # "bid" | "ask"
    trade_sizes: list[float] = field(default_factory=list)
    # Passive-add events at best bid / best ask (for refresh rate)
    bid_add_timestamps_ms: list[int] = field(default_factory=list)
    ask_add_timestamps_ms: list[int] = field(default_factory=list)
    # Tick event count per snapshot (for tick acceleration).
    # An entry is appended every time the mid price changes, giving a
    # monotonically increasing series we can differentiate twice.
    tick_event_timestamps_ms: list[int] = field(default_factory=list)

    def append_snapshot(self, snap: L1Snapshot) -> None:
        # Detect tick event (mid changed vs previous snapshot)
        if self.snapshots and self.snapshots[-1].mid != snap.mid:
            self.tick_event_timestamps_ms.append(snap.timestamp_ms)
        self.snapshots.append(snap)

    def append_trade(
        self, timestamp_ms: int, side: str, size: float
    ) -> None:
        self.trade_timestamps_ms.append(timestamp_ms)
        self.trade_sides.append(side)
        self.trade_sizes.append(size)

    def append_bid_add(self, timestamp_ms: int) -> None:
        self.bid_add_timestamps_ms.append(timestamp_ms)

    def append_ask_add(self, timestamp_ms: int) -> None:
        self.ask_add_timestamps_ms.append(timestamp_ms)

    def trim(self, now_ms: int, horizon_ms: int = ROLL_1S) -> None:
        """Drop entries older than ``now_ms - horizon_ms``.

        Keeps memory bounded during long replays. ``horizon_ms`` defaults
        to the longest window used (1s).
        """
        cutoff = now_ms - horizon_ms
        # Snapshots: keep a bit extra so velocity features have prev-snap data.
        snap_cutoff = now_ms - max(horizon_ms, ROLL_500MS) - 50
        self.snapshots = [s for s in self.snapshots
                          if s.timestamp_ms >= snap_cutoff]
        self._trim_parallel_lists(cutoff)

    def _trim_parallel_lists(self, cutoff: int) -> None:
        # Trim trade-related parallel lists together to preserve alignment
        i = 0
        n = len(self.trade_timestamps_ms)
        while i < n and self.trade_timestamps_ms[i] < cutoff:
            i += 1
        if i > 0:
            del self.trade_timestamps_ms[:i]
            del self.trade_sides[:i]
            del self.trade_sizes[:i]

        self.bid_add_timestamps_ms = [t for t in self.bid_add_timestamps_ms
                                      if t >= cutoff]
        self.ask_add_timestamps_ms = [t for t in self.ask_add_timestamps_ms
                                      if t >= cutoff]
        self.tick_event_timestamps_ms = [t for t in self.tick_event_timestamps_ms
                                         if t >= cutoff]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_events_in_window(
    event_timestamps_ms: Sequence[int], now_ms: int, window_ms: int
) -> int:
    """Count events with timestamp in [now_ms - window_ms, now_ms]."""
    cutoff = now_ms - window_ms
    return sum(1 for t in event_timestamps_ms if cutoff <= t <= now_ms)


def _snapshots_in_window(
    snapshots: Sequence[L1Snapshot], now_ms: int, window_ms: int
) -> list[L1Snapshot]:
    """Return snapshots with timestamp in [now_ms - window_ms, now_ms]."""
    cutoff = now_ms - window_ms
    return [s for s in snapshots if cutoff <= s.timestamp_ms <= now_ms]


def _linear_rate(values: list[float], timestamps_ms: list[int]) -> float:
    """Linear-slope rate (units per second) across a series.

    Uses first-to-last difference divided by elapsed seconds. This is the
    "Δ over window / window duration" form used by the velocity features
    in spec §4.1 (e.g., bid_retreat_velocity = negative Δ best_bid per
    second over a 250ms window).

    Returns 0.0 if fewer than 2 points or zero duration.
    """
    n = len(values)
    if n < 2:
        return 0.0
    duration_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    if duration_s <= 0:
        return 0.0
    return (values[-1] - values[0]) / duration_s


# ---------------------------------------------------------------------------
# Per-snapshot feature computation
# ---------------------------------------------------------------------------

def compute_l1_features(snapshot: L1Snapshot, history: L1History) -> np.ndarray:
    """Compute all 19 L1 features for a single snapshot.

    Spec §4.1 — per-snapshot use. Latency-sensitive path.

    Returns np.ndarray shape (19,) aligned with ``L1FeatureNames.ALL``.
    Indices 16-18 are the unit-vector components of the 4-dim book shape
    (bid/ask/imbalance), exposing orthogonal info the scalar [15] collapses.

    The caller is responsible for:
      1. Appending the current snapshot to ``history`` BEFORE calling.
         (Use ``history.append_snapshot(snap)``.)
      2. Appending any trade / passive-add events that occurred since
         the last snapshot.
      3. Periodically calling ``history.trim(snapshot.timestamp_ms)`` to
         bound memory.
    """
    out = np.zeros(L1_FEATURE_DIM, dtype=np.float64)

    mid = snapshot.mid
    spread = snapshot.spread
    now = snapshot.timestamp_ms

    # ──────────────────────────────────────────────────────────────
    # [0] spread_ticks — (ask - bid) / tick_size
    # ──────────────────────────────────────────────────────────────
    out[0] = spread / snapshot.tick_size if snapshot.tick_size > 0 else 0.0

    # ──────────────────────────────────────────────────────────────
    # [1] spread_bps_l1 — (spread / mid) × 10000
    # Ref: OB-reference L830 ("spread in basis points, not ticks")
    # ──────────────────────────────────────────────────────────────
    out[1] = (spread / mid) * 10_000.0 if mid > 0 else 0.0

    # ──────────────────────────────────────────────────────────────
    # [2-3] best_bid_size / best_ask_size — raw scalars
    # ──────────────────────────────────────────────────────────────
    out[2] = snapshot.best_bid_size
    out[3] = snapshot.best_ask_size

    # ──────────────────────────────────────────────────────────────
    # [4] top_imbalance — (bid_sz - ask_sz) / (bid_sz + ask_sz)
    # Range: [-1, +1]. Positive = bid-heavy. Spec §4.1, also the
    # BASELINE feature for the §9.1 information-gain gate.
    # ──────────────────────────────────────────────────────────────
    total_sz = snapshot.best_bid_size + snapshot.best_ask_size
    out[4] = ((snapshot.best_bid_size - snapshot.best_ask_size) / total_sz
              if total_sz > 0 else 0.0)

    # ──────────────────────────────────────────────────────────────
    # [5] bid_refresh_rate — passive adds at best bid per second (100ms roll)
    # [6] ask_refresh_rate — same for ask
    # Ref: OB-reference L841 — "bid_refresh_rate: passive adds per second
    # at best level, 100ms rolling"
    # ──────────────────────────────────────────────────────────────
    bid_adds = _count_events_in_window(
        history.bid_add_timestamps_ms, now, ROLL_100MS)
    ask_adds = _count_events_in_window(
        history.ask_add_timestamps_ms, now, ROLL_100MS)
    # Events in 100ms window → events per second
    out[5] = bid_adds / (ROLL_100MS / 1000.0)
    out[6] = ask_adds / (ROLL_100MS / 1000.0)

    # ──────────────────────────────────────────────────────────────
    # [7] bid_retreat_velocity — negative Δ best_bid per second (250ms)
    # [8] ask_advance_velocity — positive Δ best_ask per second (250ms)
    # [9] spread_compression_rate — Δ spread per second (250ms)
    # Ref: OB-reference L842-844, §4.1 table.
    #
    # "Retreat" = best_bid falling; we report the NEGATIVE rate so the
    # sign convention matches: positive output = bid is retreating.
    # "Advance" = best_ask rising; also reported as positive.
    # ──────────────────────────────────────────────────────────────
    window_250ms = _snapshots_in_window(history.snapshots, now, ROLL_250MS)
    if len(window_250ms) >= 2:
        ts = [s.timestamp_ms for s in window_250ms]
        bid_rate = _linear_rate([s.best_bid for s in window_250ms], ts)
        ask_rate = _linear_rate([s.best_ask for s in window_250ms], ts)
        spread_rate = _linear_rate([s.spread for s in window_250ms], ts)
        # Retreat = how fast bid is FALLING → negate the rate
        out[7] = -bid_rate if bid_rate < 0 else 0.0
        # Advance = how fast ask is RISING → positive rate
        out[8] = ask_rate if ask_rate > 0 else 0.0
        out[9] = spread_rate
    # else warmup: leaves 0.0

    # ──────────────────────────────────────────────────────────────
    # [10] tick_direction_streak — consecutive same-sign Δ mid
    # Spec §4.1: "Consecutive same-sign Δ mid changes". Counts the
    # length of the current run of up-ticks or down-ticks. Sign is
    # encoded in output: positive = up-streak, negative = down-streak.
    # ──────────────────────────────────────────────────────────────
    out[10] = _compute_tick_streak(history.snapshots)

    # ──────────────────────────────────────────────────────────────
    # [11] tick_acceleration — d²(tick_count)/dt² over 500ms
    # Ref: OB-reference L846 — "ticks/sec² — separates normal from
    # ignition". Computed as discrete second derivative of the
    # cumulative tick-event count function.
    #
    # For a tick event series at times t_i, the tick rate over
    # [now - 500ms, now] is N / 0.5s. The acceleration is the change
    # in rate between two consecutive 250ms half-windows divided by
    # 0.25s (the half-window duration) — giving ticks/sec².
    # ──────────────────────────────────────────────────────────────
    out[11] = _compute_tick_acceleration(
        history.tick_event_timestamps_ms, now, window_ms=ROLL_500MS)

    # ──────────────────────────────────────────────────────────────
    # [12] trade_at_bid_ratio — trades_at_bid / total_trades (1s roll)
    # Spec §4.1. Bid-side trades (seller hit the bid) ÷ total trades.
    # Range: [0, 1]. 0.5 = balanced; >0.5 = seller-dominated.
    # ──────────────────────────────────────────────────────────────
    cutoff_1s = now - ROLL_1S
    trade_indices = [i for i, t in enumerate(history.trade_timestamps_ms)
                     if cutoff_1s <= t <= now]
    if trade_indices:
        bid_count = sum(1 for i in trade_indices
                        if history.trade_sides[i] == "bid")
        out[12] = bid_count / len(trade_indices)
    else:
        out[12] = 0.5  # Neutral when no recent trades

    # ──────────────────────────────────────────────────────────────
    # [13] size_spike_ratio — max_trade_size / median_trade_size (1s)
    # Spec §4.1. Detects anomalous large trades against baseline.
    # 1.0 = no spikes, flat distribution. ≥3.0 = clear size outlier.
    # ──────────────────────────────────────────────────────────────
    if trade_indices:
        sizes = [history.trade_sizes[i] for i in trade_indices]
        max_sz = max(sizes)
        median_sz = float(np.median(sizes))
        out[13] = max_sz / median_sz if median_sz > 0 else 1.0
    else:
        out[13] = 1.0

    # ──────────────────────────────────────────────────────────────
    # [14] microprice_velocity — Δ microprice per second (250ms)
    # Ref: OB-reference L845. The microprice is size-weighted mid
    # (pulls toward the heavier side); velocity tells us which way
    # the book is tilting.
    # ──────────────────────────────────────────────────────────────
    if len(window_250ms) >= 2:
        micros = [s.microprice for s in window_250ms]
        ts = [s.timestamp_ms for s in window_250ms]
        out[14] = _linear_rate(micros, ts)
    # else warmup: leaves 0.0

    # ──────────────────────────────────────────────────────────────
    # [15] l1_shape_vector — normalized composite scalar
    # Spec §4.1: "[bid_sz, ask_sz, spread_bps, imbalance] normalized"
    # Projected to a scalar via weighted dot product. The four
    # components are L2-normalized (unit vector) before projection
    # so the scalar is bounded roughly in [-1, +1] regardless of
    # instrument scale.
    # ──────────────────────────────────────────────────────────────
    components = np.array([
        snapshot.best_bid_size,
        snapshot.best_ask_size,
        out[1],          # spread_bps_l1 (already normalized in bps units)
        out[4],          # top_imbalance already in [-1, +1]
    ], dtype=np.float64)
    norm = np.linalg.norm(components)
    if norm > 0:
        unit = components / norm
        out[15] = float(np.dot(unit, _L1_SHAPE_WEIGHTS))
        # [16-18] unit-vector components of the 4-dim shape (skip the
        # spread_bps component — already at [1] as spread_bps_l1).
        out[16] = float(unit[0])   # bid-size unit
        out[17] = float(unit[1])   # ask-size unit
        out[18] = float(unit[3])   # imbalance unit
    else:
        out[15] = 0.0
        out[16] = 0.0
        out[17] = 0.0
        out[18] = 0.0

    return out


def _compute_tick_streak(snapshots: list[L1Snapshot]) -> float:
    """Count consecutive same-sign Δ mid changes from the end of history.

    Returns a signed integer (as float): positive = up-streak length,
    negative = down-streak length, zero = no recent ticks or flat.
    """
    if len(snapshots) < 2:
        return 0.0

    # Compute signs of Δ mid from newest to oldest
    deltas_sign = []
    for i in range(len(snapshots) - 1, 0, -1):
        d = snapshots[i].mid - snapshots[i - 1].mid
        if d > 0:
            deltas_sign.append(1)
        elif d < 0:
            deltas_sign.append(-1)
        else:
            deltas_sign.append(0)
        # Stop looking once we have enough to compute the current run
        if len(deltas_sign) > 100:
            break

    if not deltas_sign or deltas_sign[0] == 0:
        return 0.0

    current_sign = deltas_sign[0]
    streak = 0
    for s in deltas_sign:
        if s == current_sign:
            streak += 1
        else:
            break
    return float(current_sign * streak)


def _compute_tick_acceleration(
    tick_timestamps_ms: list[int], now_ms: int, window_ms: int = ROLL_500MS
) -> float:
    """Discrete second derivative of tick-event count over ``window_ms``.

    Splits the window into two equal halves. ``rate1`` = ticks/sec in
    the older half, ``rate2`` = ticks/sec in the newer half. Acceleration
    is the change in rate divided by the half-window duration, yielding
    ticks/sec².

    Returns 0.0 during warmup.
    """
    half_ms = window_ms // 2
    older_cutoff = now_ms - window_ms
    mid_cutoff = now_ms - half_ms

    # Count events in each half
    older_count = sum(1 for t in tick_timestamps_ms
                      if older_cutoff <= t < mid_cutoff)
    newer_count = sum(1 for t in tick_timestamps_ms
                      if mid_cutoff <= t <= now_ms)

    half_s = half_ms / 1000.0
    if half_s <= 0:
        return 0.0

    rate_older = older_count / half_s
    rate_newer = newer_count / half_s

    # Δ rate ÷ Δ time (seconds between centers of halves = half_s)
    return (rate_newer - rate_older) / half_s


# ---------------------------------------------------------------------------
# Bulk series computation for notebooks
# ---------------------------------------------------------------------------

def compute_l1_series(snapshots: list[L1Snapshot]) -> pd.DataFrame:
    """Compute all 16 L1 features for a sequence of snapshots.

    Spec §4.1 — bulk backtesting path. Called from notebooks 03-07.

    Returns ``pd.DataFrame`` with columns = ``L1FeatureNames.ALL``,
    index = timestamp (from ``timestamp_ms`` converted to DatetimeIndex).

    Implementation:
      Iterates through ``compute_l1_features`` with an accumulating
      ``L1History``. This gives byte-identical results to the
      per-snapshot path. A fully vectorized path (using pandas rolling
      windows) is possible but would diverge for irregular-cadence
      snapshots — the timestamp-based windows are not trivially
      expressible as pandas ``.rolling(window=N)``. The history.trim()
      call bounds memory at ~1s of snapshots so the loop is O(N)
      in practice.

    For ≥500k rows this produces a 500k × 16 DataFrame in <30s on a
    modern laptop (dominated by per-row computation, not bookkeeping).

    Note: trade events and passive-add events are NOT derivable from
    snapshots alone. Features [5], [6], [12], [13] will default to
    zero/neutral unless the caller pre-populates ``L1History`` via
    the adapter path (see ``_l1_adapter.py``).
    """
    if not snapshots:
        return pd.DataFrame(columns=L1FeatureNames.ALL)

    history = L1History()
    rows = np.zeros((len(snapshots), L1_FEATURE_DIM), dtype=np.float64)
    timestamps = np.zeros(len(snapshots), dtype=np.int64)

    for i, snap in enumerate(snapshots):
        history.append_snapshot(snap)
        rows[i] = compute_l1_features(snap, history)
        timestamps[i] = snap.timestamp_ms
        # Trim periodically to bound memory
        if i % 1000 == 0 and i > 0:
            history.trim(snap.timestamp_ms)

    df = pd.DataFrame(rows, columns=L1FeatureNames.ALL)
    df.index = pd.to_datetime(timestamps, unit="ms", utc=True)
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# Baseline feature for information-gain gate
# ---------------------------------------------------------------------------

def bid_ask_imbalance_baseline(snapshots: list[L1Snapshot]) -> pd.Series:
    """Single-feature baseline for notebook 03 §08 information-gain gate.

    Spec §9.1, §8.3 — each L1 feature must beat this by ≥2% AUC.

    Formula: ``(bid_sz - ask_sz) / (bid_sz + ask_sz)`` per snapshot.
    Identical to feature [4] ``top_imbalance`` but exposed as a
    standalone pd.Series for direct AUC comparison in the notebook.
    """
    if not snapshots:
        return pd.Series(dtype=np.float64, name="bid_ask_imbalance_baseline")

    bid_sz = np.array([s.best_bid_size for s in snapshots], dtype=np.float64)
    ask_sz = np.array([s.best_ask_size for s in snapshots], dtype=np.float64)
    total = bid_sz + ask_sz
    # Safe division: where total == 0, set imbalance to 0.
    imbalance = np.where(total > 0, (bid_sz - ask_sz) / np.where(total > 0, total, 1.0), 0.0)

    timestamps = [s.timestamp_ms for s in snapshots]
    return pd.Series(
        imbalance,
        index=pd.to_datetime(timestamps, unit="ms", utc=True),
        name="bid_ask_imbalance_baseline",
    )