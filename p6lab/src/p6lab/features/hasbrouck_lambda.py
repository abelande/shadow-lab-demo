"""
p6lab.features.hasbrouck_lambda — Wave 6 Phase 6F

Hasbrouck's λ — a GMM-robust variant of Kyle's price-impact coefficient.
The sign-and-magnitude of signed order flow still drives the mid; the
estimator trades OLS (Wave 4) for a 2-step GMM with Δmid's lagged
absolute value as the instrument. Less sensitive to microstructure
noise and asymmetric information events.

Reference: Hasbrouck (1991), "Measuring the Information Content of Stock
Trades", Journal of Finance.

State is a rolling ``(ts, Δmid, signed_vol)`` ring buffer, identical in
shape to ``KyleLambdaState`` in microstructure.py so callers can swap
the two without rewiring.

Exported:
    HasbroucksLambdaState (dataclass)
    update_hasbrouck_lambda(state, ts_ms, mid, signed_vol)
    compute_hasbrouck_lambda(state) → float
    HASBROUCK_FEATURE_NAMES (tuple)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


WINDOW_MS = 5 * 60 * 1000        # keep last 5 min of trades
MIN_SAMPLES = 8                  # GMM is noisy below this
HASBROUCK_FEATURE_NAMES: tuple[str, ...] = ("hasbroucks_lambda",)


@dataclass
class HasbroucksLambdaState:
    """(ts, Δmid, signed_vol) triples bounded by a 5-min wall clock."""
    samples: Deque[tuple[int, float, float]] = field(default_factory=deque)
    _last_mid: float = 0.0

    def reset(self) -> None:
        self.samples.clear()
        self._last_mid = 0.0


def update_hasbrouck_lambda(
    state: HasbroucksLambdaState,
    *,
    ts_ms: int,
    mid: float,
    signed_vol: float,
) -> None:
    """Append a new sample and trim to the last 5 minutes."""
    prev = state._last_mid
    state._last_mid = float(mid)
    if prev > 0.0:
        dmid = float(mid) - prev
        state.samples.append((int(ts_ms), dmid, float(signed_vol)))
    cutoff = int(ts_ms) - WINDOW_MS
    while state.samples and state.samples[0][0] < cutoff:
        state.samples.popleft()


def compute_hasbrouck_lambda(state: HasbroucksLambdaState) -> float:
    """Two-step GMM estimator of λ in ``Δmid = λ · signed_vol + ε``.

    Instrument for step 2 is ``|Δmid_{t-1}|`` which is correlated with
    true flow but not with the iid noise in ε. Falls back to the Kyle
    OLS slope when the GMM system is ill-conditioned (e.g. constant
    instrument). Returns 0.0 on warmup or degenerate inputs.
    """
    samples = list(state.samples)
    n = len(samples)
    if n < MIN_SAMPLES:
        return 0.0

    # ts is unused here but kept for signature symmetry with Kyle.
    dmids = [s[1] for s in samples]
    flows = [s[2] for s in samples]

    # Instrument: |Δmid_{t-1}| aligned with (Δmid_t, flow_t) → drop first row
    instr = [abs(dmids[i - 1]) for i in range(1, n)]
    y = dmids[1:]
    x = flows[1:]
    if not instr or not y:
        return 0.0

    # Step 1: project flow onto instrument to get fitted-flow.
    # coef = <instr, x> / <instr, instr>
    denom = sum(z * z for z in instr)
    if denom <= 1e-12:
        return _ols_fallback(dmids, flows)
    coef = sum(z * xi for z, xi in zip(instr, x)) / denom
    x_hat = [coef * z for z in instr]

    # Step 2: regress y on x_hat → λ
    denom2 = sum(v * v for v in x_hat)
    if denom2 <= 1e-12:
        return _ols_fallback(dmids, flows)
    lam = sum(v * yi for v, yi in zip(x_hat, y)) / denom2
    return float(lam)


def snapshot_hasbrouck_features(
    state: HasbroucksLambdaState,
) -> dict[str, float]:
    return {"hasbroucks_lambda": compute_hasbrouck_lambda(state)}


def _ols_fallback(dmids: list[float], flows: list[float]) -> float:
    """Straight OLS slope of Δmid on signed_vol when the GMM system is
    degenerate. Matches KyleLambdaState.value()."""
    n = len(flows)
    if n < 2:
        return 0.0
    denom = sum(f * f for f in flows)
    if denom <= 1e-12:
        return 0.0
    return float(sum(f * d for f, d in zip(flows, dmids)) / denom)
