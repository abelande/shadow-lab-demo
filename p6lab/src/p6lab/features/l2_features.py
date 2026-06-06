"""
p6lab.features.l2_features — 18-feature L2 feature set.

Spec: p6-notebook-lab-spec.md §4.2 | OB-reference.md L432-451
Wave 9 §H.1: features 12-15 added in Phase A2a; 14 + 16-17 reworked in the
cup_flip consumption refactor — the lab now reads streak state from
``p6v2.cup_flip.StreakDetector`` (same detector class used by
``p6lab.patterns.cascade_taxonomy``) rather than a parallel re-implementation.
Eliminates the unbounded ``trade_streak`` growth and aligns ML feature
semantics with the production pattern-detection runtime.

18 features (canonical order, see L2FeatureNames):
  0  bid_ask_imbalance
  1  depth_ratio
  2  weighted_mid
  3  spread_bps
  4  depth_within_pct
  5  imbalance_ema
  6  depth_change_rate_5s
  7  depth_change_rate_30s
  8  level_persistence
  9  refresh_rate
  10 book_shape_vector       (scalar projection — see compute_book_shape_vector for the full 40-dim)
  11 trade_flow_toxicity     (VPIN, see vpin.py)
  12 signed_flow_60s         (Wave 9 A2a — signed FILL volume / total |volume|, last 60s)
  13 imbalance_velocity_5s   (Wave 9 A2a — d(bid_ask_imbalance)/dt over last 5s)
  14 current_streak_length   (cup_flip-derived — signed gap-tolerant count of current streak)
  15 liquidity_withdrawal_asym  (Wave 9 A2a — bid-side withdrawal vs ask-side withdrawal, last 5s)
  16 current_streak_velocity (cup_flip-derived — signed levels-per-second of current streak)
  17 current_streak_vw_strength  (cup_flip-derived — signed volume-weighted strength w/ recency decay)
"""
from __future__ import annotations

import bisect
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# p6v2.cup_flip lives at projects/p6-v2/cup_flip and uses relative imports
# (``from ..models import ...``), so it must be imported as ``p6v2.cup_flip``.
# Same sys.path trick used in p6lab.patterns.cascade_taxonomy.
_PROJECTS = Path(__file__).resolve().parents[5]  # .../projects/
if str(_PROJECTS) not in sys.path:
    sys.path.insert(0, str(_PROJECTS))

from p6v2.cup_flip.streak_detector import StreakDetector  # noqa: E402
from p6v2.models import Order, OrderAction, Side          # noqa: E402

logger = logging.getLogger(__name__)

L2_FEATURE_DIM: int = 18
BOOK_SHAPE_VECTOR_DIM: int = 40
BOOK_DEPTH_LEVELS: int = 20

DEPTH_WITHIN_PCT: float = 0.005
IMBALANCE_EMA_ALPHA: float = 0.1

# Wave 9 A2a — momentum feature windows.
SIGNED_FLOW_WINDOW_MS: int = 60_000     # 60s for cumulative trade-flow signal
IMBALANCE_VELOCITY_WINDOW_MS: int = 5_000   # 5s lookback for d(imbalance)/dt
LIQUIDITY_WITHDRAWAL_WINDOW_MS: int = 5_000  # 5s for bid/ask depth-withdraw delta

# cup_flip detector defaults — match what cascade_taxonomy uses so the ML
# feature view of "streak" is identical to the pattern-detection view.
DEFAULT_STREAK_MIN_LENGTH: int = 3
DEFAULT_STREAK_GAP_TOLERANCE: int = 1

# cup_flip's Streak.velocity returns 999.0 as a sentinel when duration is 0
# (multiple fills in the same millisecond). Sanitize at feature-extraction
# time so the ML model doesn't see a magic outlier.
_VELOCITY_SENTINEL: float = 999.0


class L2FeatureNames:
    BID_ASK_IMBALANCE = "bid_ask_imbalance"
    DEPTH_RATIO = "depth_ratio"
    WEIGHTED_MID = "weighted_mid"
    SPREAD_BPS = "spread_bps"
    DEPTH_WITHIN_PCT = "depth_within_pct"
    IMBALANCE_EMA = "imbalance_ema"
    DEPTH_CHANGE_RATE_5S = "depth_change_rate_5s"
    DEPTH_CHANGE_RATE_30S = "depth_change_rate_30s"
    LEVEL_PERSISTENCE = "level_persistence"
    REFRESH_RATE = "refresh_rate"
    BOOK_SHAPE_VECTOR = "book_shape_vector"
    TRADE_FLOW_TOXICITY = "trade_flow_toxicity"
    # Wave 9 A2a momentum features (indices 12, 13, 15).
    SIGNED_FLOW_60S = "signed_flow_60s"
    IMBALANCE_VELOCITY_5S = "imbalance_velocity_5s"
    LIQUIDITY_WITHDRAWAL_ASYM = "liquidity_withdrawal_asym"
    # cup_flip-derived streak features (indices 14, 16, 17). Replaces the
    # parallel `trade_streak` reimplementation that grew unboundedly on
    # one-sided runs (std=815k on day-1 audit).
    CURRENT_STREAK_LENGTH = "current_streak_length"
    CURRENT_STREAK_VELOCITY = "current_streak_velocity"
    CURRENT_STREAK_VW_STRENGTH = "current_streak_vw_strength"

    ALL: list[str] = [
        BID_ASK_IMBALANCE, DEPTH_RATIO, WEIGHTED_MID, SPREAD_BPS,
        DEPTH_WITHIN_PCT, IMBALANCE_EMA, DEPTH_CHANGE_RATE_5S,
        DEPTH_CHANGE_RATE_30S, LEVEL_PERSISTENCE, REFRESH_RATE,
        BOOK_SHAPE_VECTOR, TRADE_FLOW_TOXICITY,
        SIGNED_FLOW_60S, IMBALANCE_VELOCITY_5S, CURRENT_STREAK_LENGTH,
        LIQUIDITY_WITHDRAWAL_ASYM,
        CURRENT_STREAK_VELOCITY, CURRENT_STREAK_VW_STRENGTH,
    ]


@dataclass
class L2Snapshot:
    timestamp_ms: int
    symbol: str
    mid_price: float
    book_levels: list[tuple[float, float, float]]   # (price, bid_sz, ask_sz)
    last_trade_side: str | None = None
    last_trade_size: float | None = None
    # Wave 4 Phase 1A: raw event stream since previous snapshot. When
    # populated, compute_l2_features appends ADD-action event timestamps
    # to history.refresh_event_timestamps so `refresh_rate` feature is
    # actually live (was always 0 before — dead feature).
    recent_events: list = field(default_factory=list)


@dataclass
class L2History:
    snapshots: list[L2Snapshot] = field(default_factory=list)
    imbalance_ema_state: float = 0.0
    recent_add_events: int = 0
    refresh_window_ms: int = 1_000
    refresh_event_timestamps: list[int] = field(default_factory=list)
    vpin_value: float = 0.0  # cached, refreshed externally
    # Wave 9 A2a — rolling FILL events for signed_flow.
    # Each entry: (timestamp_ms, side_sign in {-1, +1}, abs_size).
    trade_events: list[tuple[int, int, float]] = field(default_factory=list)
    # cup_flip-derived streak state (replaces the old trade_streak int).
    # Default uses cup_flip's defaults so the ML feature view matches
    # cascade_taxonomy's view. Override by constructing your own detector:
    #     L2History(streak_detector=StreakDetector(min_streak_length=5))
    streak_detector: StreakDetector = field(
        default_factory=lambda: StreakDetector(
            min_streak_length=DEFAULT_STREAK_MIN_LENGTH,
            gap_tolerance=DEFAULT_STREAK_GAP_TOLERANCE,
        ),
    )
    # Wave 9 A2a — per-snapshot bid/ask total snapshot (used by
    # liquidity_withdrawal_asym to detect side-specific depth retreat).
    side_totals: list[tuple[int, float, float]] = field(default_factory=list)

    def append(self, snap: L2Snapshot, max_history_ms: int = 60_000) -> None:
        self.snapshots.append(snap)
        cutoff = snap.timestamp_ms - max_history_ms
        # Trim
        while self.snapshots and self.snapshots[0].timestamp_ms < cutoff:
            self.snapshots.pop(0)
        while self.refresh_event_timestamps and self.refresh_event_timestamps[0] < cutoff:
            self.refresh_event_timestamps.pop(0)
        while self.trade_events and self.trade_events[0][0] < cutoff:
            self.trade_events.pop(0)
        while self.side_totals and self.side_totals[0][0] < cutoff:
            self.side_totals.pop(0)


def _totals(snap: L2Snapshot) -> tuple[float, float]:
    total_bid = sum(b for _, b, _ in snap.book_levels)
    total_ask = sum(a for _, _, a in snap.book_levels)
    return total_bid, total_ask


def _best_bid_ask(snap: L2Snapshot) -> tuple[float, float]:
    """Best bid = highest price with bid_sz > 0; best ask = lowest with ask_sz > 0."""
    best_bid = 0.0
    best_ask = 0.0
    for price, bid_sz, ask_sz in snap.book_levels:
        if bid_sz > 0 and price > best_bid:
            best_bid = price
        if ask_sz > 0 and (best_ask == 0.0 or price < best_ask):
            best_ask = price
    return best_bid, best_ask


def _depth_at_or_before(history: L2History, target_ms: int) -> float | None:
    """Total depth (bid+ask) of the snapshot at-or-before target_ms."""
    if not history.snapshots:
        return None
    # Binary search by timestamp
    timestamps = [s.timestamp_ms for s in history.snapshots]
    idx = bisect.bisect_right(timestamps, target_ms) - 1
    if idx < 0:
        return None
    s = history.snapshots[idx]
    tb, ta = _totals(s)
    return tb + ta


def _imbalance_ema_at_or_before(
    history: L2History, target_ms: int,
) -> float | None:
    """Re-derive bid_ask_imbalance for the snapshot at-or-before target_ms.

    Used by ``imbalance_velocity_5s`` to compute d(imbalance)/dt without
    storing extra state — recomputes from totals on the historical
    snapshot. Cheap because L2History keeps only ~60s of snapshots.
    """
    if not history.snapshots:
        return None
    timestamps = [s.timestamp_ms for s in history.snapshots]
    idx = bisect.bisect_right(timestamps, target_ms) - 1
    if idx < 0:
        return None
    s = history.snapshots[idx]
    tb, ta = _totals(s)
    total = tb + ta
    return (tb - ta) / (total + 1e-9) if total > 0 else 0.0


def _side_totals_at_or_before(
    history: L2History, target_ms: int,
) -> tuple[float, float] | None:
    """(total_bid, total_ask) for the snapshot at-or-before target_ms."""
    if not history.side_totals:
        return None
    timestamps = [t for t, _, _ in history.side_totals]
    idx = bisect.bisect_right(timestamps, target_ms) - 1
    if idx < 0:
        return None
    _, tb, ta = history.side_totals[idx]
    return tb, ta


def _extract_fill_events(
    snapshot: L2Snapshot,
) -> list[tuple[int, int, float, float]]:
    """Pull (ts_ms, side_sign, price, abs_size) from snapshot.recent_events.

    Three event shapes seen in practice:
      - dataclass with .action (OrderAction enum), .side, .price, .size, .timestamp_ms
      - dict with same keys
      - p6 OrderEvent — action.name == 'FILL'

    Side sign:
      - +1 for buy / 'B' / 'BUY' (aggressor lifts ASK)
      - -1 for sell / 'S' / 'SELL' (aggressor hits BID)
      - 0  for unknown (dropped)

    Price falls back to ``snapshot.mid_price`` if absent on the event so
    StreakDetector still has a level to track (with degenerate ``depth=1``
    consequences if every fill in a streak reports the same fallback price).

    Defensive: returns empty list if no FILL events parsable. Features
    that depend on this gracefully zero out when the upstream snap
    doesn't carry trade events.
    """
    out: list[tuple[int, int, float, float]] = []
    for ev in snapshot.recent_events or []:
        action = getattr(ev, "action", None)
        if action is None and isinstance(ev, dict):
            action = ev.get("action")
        action_name = ""
        if action is not None:
            action_name = (getattr(action, "name", None) or str(action)).upper()
        if "FILL" not in action_name and "TRADE" not in action_name:
            continue

        side_raw = getattr(ev, "side", None)
        if side_raw is None and isinstance(ev, dict):
            side_raw = ev.get("side")
        side_str = str(side_raw).upper() if side_raw is not None else ""
        if side_str.startswith("B") or side_str == "1" or side_str == "+1":
            side_sign = 1
        elif side_str.startswith("S") or side_str == "-1":
            side_sign = -1
        else:
            continue  # unknown side — drop

        size = getattr(ev, "size", None)
        if size is None and isinstance(ev, dict):
            size = ev.get("size")
        if size is None:
            continue
        try:
            size_f = float(size)
        except (TypeError, ValueError):
            continue
        if size_f <= 0:
            continue

        ts = getattr(ev, "timestamp_ms", None)
        if ts is None and isinstance(ev, dict):
            ts = ev.get("timestamp_ms")
        if ts is None:
            ts = snapshot.timestamp_ms

        price = getattr(ev, "price", None)
        if price is None and isinstance(ev, dict):
            price = ev.get("price")
        try:
            price_f = float(price) if price is not None else float(snapshot.mid_price)
        except (TypeError, ValueError):
            price_f = float(snapshot.mid_price)

        out.append((int(ts), side_sign, price_f, size_f))
    return out


def compute_book_shape_vector(snapshot: L2Snapshot) -> np.ndarray:
    """40-dim normalized depth profile: [bid_20..bid_1 | ask_1..ask_20].

    Each side is normalized so it sums to 1 (or is zero if that side is empty).
    """
    bids: list[float] = []
    asks: list[float] = []
    for _price, bid_sz, ask_sz in snapshot.book_levels[:BOOK_DEPTH_LEVELS]:
        bids.append(float(bid_sz))
        asks.append(float(ask_sz))
    while len(bids) < BOOK_DEPTH_LEVELS:
        bids.append(0.0)
    while len(asks) < BOOK_DEPTH_LEVELS:
        asks.append(0.0)

    bid_total = sum(bids)
    ask_total = sum(asks)
    bid_norm = [b / bid_total for b in bids] if bid_total > 0 else [0.0] * BOOK_DEPTH_LEVELS
    ask_norm = [a / ask_total for a in asks] if ask_total > 0 else [0.0] * BOOK_DEPTH_LEVELS

    # bid 20..1 (deepest first), then ask 1..20 (shallowest first)
    return np.asarray(list(reversed(bid_norm)) + ask_norm, dtype=np.float64)


def compute_l2_features(snapshot: L2Snapshot, history: L2History) -> np.ndarray:
    """Compute all 18 L2 features. Mutates ``history`` (EMA, fill buffer,
    streak detector state, side totals, and appends snapshot)."""
    out = np.zeros(L2_FEATURE_DIM, dtype=np.float64)
    total_bid, total_ask = _totals(snapshot)
    total = total_bid + total_ask

    # [0] bid_ask_imbalance
    out[0] = (total_bid - total_ask) / (total + 1e-9) if total > 0 else 0.0
    # [1] depth_ratio
    out[1] = total_bid / (total_ask + 1e-9) if total_ask > 0 else (
        float("inf") if total_bid > 0 else 1.0
    )
    if not np.isfinite(out[1]):
        out[1] = 1e6 if total_bid > 0 else 1.0
    # [2] weighted_mid
    num = sum(p * (b + a) for p, b, a in snapshot.book_levels)
    den = total
    out[2] = num / den if den > 0 else snapshot.mid_price
    # [3] spread_bps
    best_bid, best_ask = _best_bid_ask(snapshot)
    if best_bid > 0 and best_ask > 0 and snapshot.mid_price > 0:
        out[3] = (best_ask - best_bid) / snapshot.mid_price * 10_000.0
    # [4] depth_within_pct
    if snapshot.mid_price > 0:
        cutoff = snapshot.mid_price * DEPTH_WITHIN_PCT
        out[4] = sum(
            (b + a) for p, b, a in snapshot.book_levels
            if abs(p - snapshot.mid_price) <= cutoff
        )
    # [5] imbalance_ema (update history state in place)
    history.imbalance_ema_state = (
        IMBALANCE_EMA_ALPHA * out[0]
        + (1.0 - IMBALANCE_EMA_ALPHA) * history.imbalance_ema_state
    )
    out[5] = history.imbalance_ema_state
    # [6] depth_change_rate_5s
    target_5s = snapshot.timestamp_ms - 5_000
    prior_5s = _depth_at_or_before(history, target_5s)
    if prior_5s is not None:
        out[6] = (total - prior_5s) / 5.0
    # [7] depth_change_rate_30s
    target_30s = snapshot.timestamp_ms - 30_000
    prior_30s = _depth_at_or_before(history, target_30s)
    if prior_30s is not None:
        out[7] = (total - prior_30s) / 30.0
    # [8] level_persistence — fraction of (price, size) tuples unchanged vs prev snapshot
    if history.snapshots:
        prev = history.snapshots[-1]
        prev_set = {(round(p, 6), round(b, 6), round(a, 6))
                    for p, b, a in prev.book_levels}
        cur_set = {(round(p, 6), round(b, 6), round(a, 6))
                   for p, b, a in snapshot.book_levels}
        if cur_set:
            out[8] = len(prev_set & cur_set) / len(cur_set)
    else:
        out[8] = 1.0
    # Wave 4 Phase 1A: absorb ADD-action events from the snapshot's
    # recent_events list into the rolling refresh buffer. Handles three
    # event shapes seen in practice: dataclass (.action.name), dict
    # ({'action': 'ADD'}), P6Order-like (str(ev.action) == 'OrderAction.ADD').
    for ev in snapshot.recent_events or []:
        action = getattr(ev, "action", None)
        if action is None and isinstance(ev, dict):
            action = ev.get("action")
        action_name = getattr(action, "name", None) or str(action) if action else ""
        if "ADD" in action_name.upper():
            ev_ts = getattr(ev, "timestamp_ms", None)
            if ev_ts is None and isinstance(ev, dict):
                ev_ts = ev.get("timestamp_ms")
            if ev_ts is not None:
                history.refresh_event_timestamps.append(int(ev_ts))

    # [9] refresh_rate — events/sec over last refresh_window_ms
    cutoff_ms = snapshot.timestamp_ms - history.refresh_window_ms
    recent_count = sum(1 for ts in history.refresh_event_timestamps if ts >= cutoff_ms)
    out[9] = recent_count * 1000.0 / max(1, history.refresh_window_ms)
    # [10] book_shape_vector scalar — L2-norm of the 40-d profile
    bsv = compute_book_shape_vector(snapshot)
    out[10] = float(np.linalg.norm(bsv))
    # [11] trade_flow_toxicity — externally cached VPIN
    out[11] = history.vpin_value

    # ── Wave 9 A2a + cup_flip refactor — momentum + streak features ──────
    # Pull this snapshot's FILL events (best-effort), update the rolling
    # buffer (signed_flow uses it) AND feed cup_flip's StreakDetector
    # (which owns the gap-tolerant, min-length-filtered streak state used
    # for features 14, 16, 17). Defensive: with no FILL events the streak
    # features stay at whatever the detector held coming in, which is the
    # correct "no new fills → state unchanged" semantic.
    fill_events = _extract_fill_events(snapshot)
    for ts, side_sign, price_f, size_f in fill_events:
        history.trade_events.append((ts, side_sign, size_f))
        # Side mapping into cup_flip's frame: a buyer-aggressor lifts a
        # resting ASK, so side_sign=+1 → Side.ASK (bullish streak).
        cup_side = Side.ASK if side_sign > 0 else Side.BID
        history.streak_detector.process_fill(Order(
            order_id="",
            side=cup_side,
            price=price_f,
            size=size_f,
            timestamp_ms=int(ts),
            action=OrderAction.FILL,
            is_aggressive=True,
        ))

    # [12] signed_flow_60s — sum(signed * |size|) / sum(|size|), 60s window.
    cutoff_60s = snapshot.timestamp_ms - SIGNED_FLOW_WINDOW_MS
    signed_sum = 0.0
    abs_sum = 0.0
    for ts, side_sign, size_f in history.trade_events:
        if ts < cutoff_60s:
            continue
        signed_sum += side_sign * size_f
        abs_sum += size_f
    out[12] = signed_sum / abs_sum if abs_sum > 0 else 0.0

    # [13] imbalance_velocity_5s — d(bid_ask_imbalance)/dt over last 5s.
    target_5s_imb = snapshot.timestamp_ms - IMBALANCE_VELOCITY_WINDOW_MS
    prior_imb = _imbalance_ema_at_or_before(history, target_5s_imb)
    out[13] = (
        (out[0] - prior_imb) / (IMBALANCE_VELOCITY_WINDOW_MS / 1000.0)
        if prior_imb is not None else 0.0
    )

    # [14] current_streak_length — signed gap-tolerant count from cup_flip's
    # StreakDetector. Replaces the unbounded trade_streak count; cup_flip's
    # detector segments streaks at gap_tolerance opposing fills so length
    # has natural saturation per segment.
    streak = history.streak_detector.current_streak
    if streak is None:
        out[14] = 0.0
        streak_sign = 0.0
        streak_velocity = 0.0
        streak_vw = 0.0
    else:
        streak_sign = 1.0 if streak.side == Side.ASK else -1.0
        out[14] = streak_sign * float(streak.length)
        raw_v = float(streak.velocity)
        # cup_flip emits 999.0 as a "zero-duration" sentinel — sanitize so
        # the ML model doesn't see a magic outlier.
        streak_velocity = streak_sign * (0.0 if raw_v >= _VELOCITY_SENTINEL else raw_v)
        streak_vw = streak_sign * float(streak.volume_weighted_strength)

    # [15] liquidity_withdrawal_asym — bid-side withdrawal vs ask-side
    # withdrawal over LIQUIDITY_WITHDRAWAL_WINDOW_MS. Δ_bid < 0 means the
    # bid side lost depth; same for ask. Asymmetry is signed: positive
    # when the bid side withdrew more than the ask side (net selling
    # pressure visible in the book), negative for the opposite.
    target_5s_liq = snapshot.timestamp_ms - LIQUIDITY_WITHDRAWAL_WINDOW_MS
    prior_sides = _side_totals_at_or_before(history, target_5s_liq)
    if prior_sides is not None:
        prior_tb, prior_ta = prior_sides
        d_bid = total_bid - prior_tb
        d_ask = total_ask - prior_ta
        # Signed asymmetry of the *withdrawals* (not of the deltas
        # themselves): if both sides shed depth, who shed more?
        bid_withdraw = max(0.0, -d_bid)
        ask_withdraw = max(0.0, -d_ask)
        denom = bid_withdraw + ask_withdraw
        if denom > 0:
            out[15] = (bid_withdraw - ask_withdraw) / denom
        else:
            out[15] = 0.0
    else:
        out[15] = 0.0
    # Record per-snapshot side totals for next snapshot's lookback.
    history.side_totals.append(
        (snapshot.timestamp_ms, float(total_bid), float(total_ask)),
    )

    # [16] current_streak_velocity — signed levels-per-second of the current
    # streak (cup_flip's depth/duration metric, signed by streak direction).
    # Bounded by physical price grid + clock; cannot blow up like a raw
    # count.
    # [17] current_streak_vw_strength — recency-decayed volume sum (cup_flip's
    # 0.9-decay factor caps the asymptotic sum at ~10×max_size).
    out[16] = float(streak_velocity)
    out[17] = float(streak_vw)

    history.append(snapshot)
    return out


def compute_l2_series(
    snapshots: list[L2Snapshot],
    *,
    streak_min_length: int = DEFAULT_STREAK_MIN_LENGTH,
    streak_gap_tolerance: int = DEFAULT_STREAK_GAP_TOLERANCE,
) -> pd.DataFrame:
    """Bulk computation. Returns DataFrame indexed by timestamp_ms.

    Parameters
    ----------
    snapshots : list[L2Snapshot]
        Ordered sequence of snapshots to feature-ize.
    streak_min_length : int
        Forwarded to ``StreakDetector(min_streak_length=...)``. Streaks
        shorter than this don't get emitted as "completed" by cup_flip
        (affects downstream consumers; not the current_streak read).
    streak_gap_tolerance : int
        Forwarded to ``StreakDetector(gap_tolerance=...)``. Number of
        opposing-side fills the detector absorbs before closing the
        current streak. Default 1 matches cascade_taxonomy.
    """
    if not snapshots:
        return pd.DataFrame(columns=L2FeatureNames.ALL)
    history = L2History(
        streak_detector=StreakDetector(
            min_streak_length=streak_min_length,
            gap_tolerance=streak_gap_tolerance,
        ),
    )
    rows = np.zeros((len(snapshots), L2_FEATURE_DIM), dtype=np.float64)
    for i, snap in enumerate(snapshots):
        rows[i] = compute_l2_features(snap, history)
    df = pd.DataFrame(rows, columns=L2FeatureNames.ALL)
    df.index = [s.timestamp_ms for s in snapshots]
    df.index.name = "timestamp_ms"
    return df
