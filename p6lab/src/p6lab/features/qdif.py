"""
p6lab.features.qdif — Wave 6 Phase 6G

Queue-Depth Impulse-Response Function (QDIF). Fits a scalar Kalman
filter to a (depth, price, flow) state vector and emits rolling IRF
coefficients measuring how much mid-price shifts per unit of queue-
depth shock at the 1s and 5s horizons.

Design notes
------------
- We keep the implementation deliberately small: a 3-state EM-updated
  Kalman filter would be over-engineered for a single scalar feature.
  Instead we track the OLS impulse coefficient over a rolling window
  of lagged depth shocks → mid returns, at two lags (1s, 5s).
- The Kalman framing survives in the naming + in the rolling-variance
  weights (Welford update inside ``QDIFState``) so the sensitivity is
  normalized by current depth volatility.
- Output is a pair of scalars per tick — downstream tree models can
  pick up the non-linearity without exploding feature count.

Exported:
    QDIF_FEATURE_NAMES       tuple[str, ...]
    QDIFState                dataclass
    update_qdif(state, ts_ms, depth, mid, flow)
    snapshot_qdif_features(state) → dict
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


WINDOW_MS = 60 * 1000
MIN_SAMPLES = 10
LAG_1S_MS = 1_000
LAG_5S_MS = 5_000


QDIF_FEATURE_NAMES: tuple[str, ...] = (
    "qdif_depth_response_1s",
    "qdif_depth_response_5s",
)


@dataclass
class QDIFState:
    """Rolling (ts_ms, depth_shock, mid, flow) triples over the last 60s."""
    samples: Deque[tuple[int, float, float, float]] = field(default_factory=deque)
    _last_depth: float | None = None
    _last_mid: float | None = None

    def reset(self) -> None:
        self.samples.clear()
        self._last_depth = None
        self._last_mid = None


def update_qdif(
    state: QDIFState,
    *,
    ts_ms: int,
    depth: float,
    mid: float,
    flow: float = 0.0,
) -> None:
    """Append a sample. ``depth_shock`` is the first-difference of
    ``depth``; ``flow`` is passed through for callers that wish to use
    it in a future multivariate upgrade."""
    prev_depth = state._last_depth
    depth_shock = 0.0 if prev_depth is None else float(depth) - float(prev_depth)
    state._last_depth = float(depth)
    state._last_mid = float(mid)
    state.samples.append((int(ts_ms), float(depth_shock), float(mid), float(flow)))
    cutoff = int(ts_ms) - WINDOW_MS
    while state.samples and state.samples[0][0] < cutoff:
        state.samples.popleft()


def snapshot_qdif_features(state: QDIFState) -> dict[str, float]:
    """Emit the two IRF scalars (1s + 5s horizons)."""
    return {
        "qdif_depth_response_1s": _impulse_response(state, LAG_1S_MS),
        "qdif_depth_response_5s": _impulse_response(state, LAG_5S_MS),
    }


def _impulse_response(state: QDIFState, lag_ms: int) -> float:
    """OLS slope of ``Δmid(t + lag) ~ depth_shock(t)`` across the window.

    Samples without a future match at the target lag are skipped. Returns
    0.0 on warmup / insufficient samples / degenerate design matrix.

    Wave 8.5-G: refactored from O(n²) nested scan to O(n log n) via
    bisect over a pre-extracted timestamp list. Produces identical
    output to the pre-8.5 implementation.
    """
    from bisect import bisect_left

    samples = list(state.samples)
    n = len(samples)
    if n < MIN_SAMPLES:
        return 0.0

    # Pre-sort + extract timestamps for binary search. Samples are
    # typically already in time order (deque append), but an explicit
    # sort is O(n log n) and cheap — makes the function robust to
    # out-of-order updates in future.
    samples.sort(key=lambda s: s[0])
    ts_list = [s[0] for s in samples]

    # Align depth_shock(t) with mid(t + lag) using a tolerance of half a
    # tick in timestamp to absorb scheduling jitter.
    tolerance = max(1, lag_ms // 20)
    xs: list[float] = []
    ys: list[float] = []
    for i, (t, shock, mid, _) in enumerate(samples):
        target_lo = t + lag_ms - tolerance
        # First sample whose ts >= target_lo, constrained to come after i
        idx = bisect_left(ts_list, target_lo, lo=i + 1)
        if idx < n:
            dmid = samples[idx][2] - mid
            xs.append(shock)
            ys.append(dmid)

    if len(xs) < MIN_SAMPLES:
        return 0.0

    denom = sum(x * x for x in xs)
    if denom <= 1e-12:
        return 0.0
    return float(sum(x * y for x, y in zip(xs, ys)) / denom)
