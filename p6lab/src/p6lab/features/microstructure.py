"""Wave 4 Phase 2 — additional Level-2 microstructure features.

Five features bundled here, each with its own rolling-state dataclass
and update function. Called per-snapshot from NB06 §01 and from the
live ``FeatureAccumulator``.

Features
--------
1. **OFI** (Order Flow Imbalance, signed trade-volume over rolling
   window). Simplified bridge from ``p6v2.ofi.OFITracker``. Emitted as
   ``ofi_1s``, ``ofi_5s``, ``ofi_30s``.
2. **Realized variance** over 30s: ``sum(log(mid_{i+1}/mid_i)²)``.
   Scalar ``realized_variance_30s``.
3. **Roll spread** estimator: ``2√(-cov(Δmid, lag(Δmid)))`` over 60s.
   Scalar ``roll_spread_bps``.
4. **Kyle's λ**: OLS slope of ``Δprice ~ signed_volume`` over 5min.
   Scalar ``kyles_lambda``.
5. **Tick-rule PIN**: ``|buy_ticks - sell_ticks| / total`` over 60s.
   Scalar ``tick_rule_pin``.

Each `update_*` function takes its state + new input(s), returns the
current scalar value (or 0.0 on warm-up).

Design notes
------------
These are stateful *rolling* features — callers create one state per
symbol and call update per snapshot (or per trade). Windows are
millisecond-bounded, not count-bounded, so sparse snapshots don't
warp the window length.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import log, sqrt
from typing import Deque

import numpy as np


# ---------------------------------------------------------------------------
# Shared rolling buffer utility
# ---------------------------------------------------------------------------


def _trim_deque_by_ts(buf: Deque, now_ms: int, window_ms: int) -> None:
    """Drop entries older than ``now_ms - window_ms`` (entries are
    tuples whose first element is timestamp_ms)."""
    cutoff = now_ms - window_ms
    while buf and buf[0][0] < cutoff:
        buf.popleft()


# ---------------------------------------------------------------------------
# 1. OFI — Order Flow Imbalance
# ---------------------------------------------------------------------------


@dataclass
class OFIState:
    """Rolling signed-trade-volume buffer. Entries: (ts_ms, signed_vol)."""
    trades: Deque[tuple[int, float]] = field(default_factory=deque)

    def update(self, ts_ms: int, side: str, volume: float) -> None:
        signed = volume if side.lower() == "buy" else -volume
        self.trades.append((int(ts_ms), float(signed)))
        # Cap buffer at 30s by default — longest window we compute
        _trim_deque_by_ts(self.trades, ts_ms, 30_000)

    def ofi(self, now_ms: int, window_ms: int) -> float:
        """Sum of signed volumes over ``[now - window_ms, now]``."""
        cutoff = now_ms - window_ms
        return sum(v for ts, v in self.trades if ts >= cutoff)


# ---------------------------------------------------------------------------
# 2. Realized variance over 30s
# ---------------------------------------------------------------------------


@dataclass
class RealizedVarianceState:
    """Rolling mid-price buffer for RV computation."""
    mids: Deque[tuple[int, float]] = field(default_factory=deque)

    def update(self, ts_ms: int, mid: float) -> None:
        self.mids.append((int(ts_ms), float(mid)))
        _trim_deque_by_ts(self.mids, ts_ms, 30_000)

    def value(self) -> float:
        """Sum of squared log-returns over the buffer."""
        if len(self.mids) < 2:
            return 0.0
        prev_mid = self.mids[0][1]
        s = 0.0
        for _, m in list(self.mids)[1:]:
            if prev_mid > 0 and m > 0:
                r = log(m / prev_mid)
                s += r * r
            prev_mid = m
        return s


# ---------------------------------------------------------------------------
# 3. Roll spread estimator
# ---------------------------------------------------------------------------


@dataclass
class RollSpreadState:
    """Buffers last N mid-price Δs for Roll's covariance spread."""
    dmids: Deque[tuple[int, float]] = field(default_factory=deque)
    _last_mid: float = 0.0

    def update(self, ts_ms: int, mid: float) -> None:
        if self._last_mid > 0:
            self.dmids.append((int(ts_ms), float(mid) - self._last_mid))
            _trim_deque_by_ts(self.dmids, ts_ms, 60_000)
        self._last_mid = float(mid)

    def value(self, mid: float | None = None) -> float:
        """Roll spread ≈ 2√(-cov(Δmid, lag Δmid)). Returns 0 if +cov.

        Convert to bps via ``mid`` if provided; else absolute tick units.
        """
        if len(self.dmids) < 3:
            return 0.0
        arr = np.asarray([d for _, d in self.dmids], dtype=float)
        if arr.size < 3:
            return 0.0
        lag = arr[:-1]
        cur = arr[1:]
        cov = float(np.mean((lag - lag.mean()) * (cur - cur.mean())))
        if cov >= 0:
            return 0.0
        spread = 2.0 * sqrt(-cov)
        if mid and mid > 0:
            return (spread / mid) * 10_000.0
        return spread


# ---------------------------------------------------------------------------
# 4. Kyle's λ — price impact per unit signed flow
# ---------------------------------------------------------------------------


@dataclass
class KyleLambdaState:
    """Buffers (ts, Δmid, signed_vol) pairs for OLS slope."""
    samples: Deque[tuple[int, float, float]] = field(default_factory=deque)
    _last_mid: float = 0.0

    def update(self, ts_ms: int, mid: float, signed_vol: float) -> None:
        if self._last_mid > 0:
            dm = float(mid) - self._last_mid
            self.samples.append((int(ts_ms), dm, float(signed_vol)))
            _trim_deque_by_ts(self.samples, ts_ms, 5 * 60_000)   # 5min window
        self._last_mid = float(mid)

    def value(self) -> float:
        """Slope of Δmid ~ signed_vol via OLS. Returns 0 on too-few samples."""
        if len(self.samples) < 10:
            return 0.0
        x = np.asarray([s[2] for s in self.samples], dtype=float)
        y = np.asarray([s[1] for s in self.samples], dtype=float)
        if x.std() < 1e-9:
            return 0.0
        num = float(np.mean((x - x.mean()) * (y - y.mean())))
        den = float(np.var(x))
        if den < 1e-12:
            return 0.0
        return num / den


# ---------------------------------------------------------------------------
# 5. Tick-rule PIN — informed trading fingerprint
# ---------------------------------------------------------------------------


@dataclass
class TickRulePINState:
    """Counts buy-tick vs sell-tick trades over a rolling window."""
    events: Deque[tuple[int, int]] = field(default_factory=deque)   # (ts, +1/-1)
    _prev_price: float = 0.0

    def update(self, ts_ms: int, trade_price: float) -> None:
        if self._prev_price > 0:
            if trade_price > self._prev_price:
                self.events.append((int(ts_ms), 1))
            elif trade_price < self._prev_price:
                self.events.append((int(ts_ms), -1))
            # tied price → ignore (not informative)
        self._prev_price = float(trade_price)
        _trim_deque_by_ts(self.events, ts_ms, 60_000)

    def value(self) -> float:
        """|buy - sell| / total over the last 60s."""
        if not self.events:
            return 0.0
        total = len(self.events)
        signed_sum = sum(e for _, e in self.events)
        return abs(signed_sum) / total


# ---------------------------------------------------------------------------
# Aggregator — convenience bundle
# ---------------------------------------------------------------------------


@dataclass
class MicrostructureState:
    """One-stop container for all 5 states, keyed by symbol in production."""
    ofi: OFIState = field(default_factory=OFIState)
    rv: RealizedVarianceState = field(default_factory=RealizedVarianceState)
    roll: RollSpreadState = field(default_factory=RollSpreadState)
    kyle: KyleLambdaState = field(default_factory=KyleLambdaState)
    pin: TickRulePINState = field(default_factory=TickRulePINState)


def update_microstructure(
    state: MicrostructureState,
    *,
    ts_ms: int,
    mid: float,
    trades: list[dict] | None = None,
) -> None:
    """Update all 5 rolling states with one snapshot's info.

    ``trades`` is a list of dicts with keys: ``price``, ``volume``,
    ``side`` ('buy'|'sell'). Empty list is fine (no trades this tick).
    """
    state.rv.update(ts_ms, mid)
    state.roll.update(ts_ms, mid)
    signed_total = 0.0
    for t in trades or []:
        px = float(t.get("price") or 0.0)
        vol = float(t.get("volume") or t.get("size") or 0.0)
        side = str(t.get("side") or "").lower()
        if vol <= 0 or side not in ("buy", "sell"):
            continue
        state.ofi.update(ts_ms, side, vol)
        state.pin.update(ts_ms, px)
        signed_total += vol if side == "buy" else -vol
    state.kyle.update(ts_ms, mid, signed_total)


def snapshot_features(
    state: MicrostructureState,
    now_ms: int,
    mid: float,
) -> dict[str, float]:
    """Return the 5 feature scalars for downstream tabular use.

    Keys match column names used in the NB06 feature matrix.
    """
    return {
        "ofi_1s":  state.ofi.ofi(now_ms, 1_000),
        "ofi_5s":  state.ofi.ofi(now_ms, 5_000),
        "ofi_30s": state.ofi.ofi(now_ms, 30_000),
        "realized_variance_30s": state.rv.value(),
        "roll_spread_bps": state.roll.value(mid),
        "kyles_lambda": state.kyle.value(),
        "tick_rule_pin": state.pin.value(),
    }


MICROSTRUCTURE_FEATURE_NAMES: list[str] = [
    "ofi_1s", "ofi_5s", "ofi_30s",
    "realized_variance_30s",
    "roll_spread_bps",
    "kyles_lambda",
    "tick_rule_pin",
]
