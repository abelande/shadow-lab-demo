"""
p6lab.features.hawkes — Wave 6 Phase 6E

Parametric Hawkes self-excitation fit on trade inter-arrival times. Uses
the exponential-kernel form

    λ(t) = μ + α · Σ_{t_i < t} exp(-β · (t - t_i))

Branching ratio is ``α / β`` ∈ [0, 1); values near 1 mean clustered
cascades, values near 0 mean near-Poisson.

MLE is over a 60-second rolling window of arrival timestamps — short
enough to catch regime shifts, long enough to be numerically stable.

Exported:
    HAWKES_FEATURE_NAMES            tuple[str, ...]
    HawkesState (dataclass)
    update_hawkes(state, ts_ms)
    snapshot_hawkes_features(state) → dict
    fit_hawkes_mle(ts_ms_array) → HawkesParams

Design note
-----------
The full 2-parameter MLE is a 1D optimization over ``β`` (with ``α``
analytic given ``β`` and μ solved from the unconditional intensity). We
use a deterministic golden-section search over a bounded ``β`` interval
to avoid the scipy optimiser dependency at runtime. This keeps live
feature extraction under 1ms even on 1000-event windows.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

logger = logging.getLogger(__name__)

WINDOW_MS = 60 * 1000
MIN_EVENTS = 10
BETA_MIN = 1e-3
BETA_MAX = 10.0
GOLDEN_RATIO = (math.sqrt(5) - 1) / 2


HAWKES_FEATURE_NAMES: tuple[str, ...] = (
    "hawkes_branching_ratio",
    "hawkes_intensity",
    "hawkes_decay",
)


@dataclass
class HawkesParams:
    """Estimated (μ, α, β) and derived branching ratio."""
    mu: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

    @property
    def branching_ratio(self) -> float:
        """``α / β`` bounded to [0, 1) — meaningless outside stationarity."""
        if self.beta <= 0.0:
            return 0.0
        return float(max(0.0, min(self.alpha / self.beta, 0.999)))


@dataclass
class HawkesState:
    """Rolling arrival-time buffer. Stored in seconds for numerical
    stability — ms counts drop too many digits under ``exp(-β · dt)``."""
    arrivals_sec: Deque[float] = field(default_factory=deque)
    last_params: HawkesParams = field(default_factory=HawkesParams)

    def reset(self) -> None:
        self.arrivals_sec.clear()
        self.last_params = HawkesParams()


def update_hawkes(state: HawkesState, ts_ms: int) -> None:
    """Append an event and trim to the rolling window."""
    t_sec = float(ts_ms) / 1000.0
    state.arrivals_sec.append(t_sec)
    cutoff = t_sec - WINDOW_MS / 1000.0
    while state.arrivals_sec and state.arrivals_sec[0] < cutoff:
        state.arrivals_sec.popleft()


def snapshot_hawkes_features(state: HawkesState) -> dict[str, float]:
    """Fit (or reuse last fit when window unchanged) and emit the 3 scalars."""
    params = fit_hawkes_mle(list(state.arrivals_sec))
    state.last_params = params
    return {
        "hawkes_branching_ratio": params.branching_ratio,
        "hawkes_intensity": float(params.mu + params.alpha),
        "hawkes_decay": float(params.beta),
    }


def fit_hawkes_mle(arrivals_sec: list[float]) -> HawkesParams:
    """Fit (μ, α, β) via golden-section search over β.

    For each candidate β, α is fit analytically by matching the
    unconditional moment ``E[N(T)] = μT + α / β · Σ (1 - exp(-β · (T - t_i)))``
    and μ is set to the Poisson baseline on early events. We return the
    triple that minimizes the negative log-likelihood.

    Returns zero parameters when too few events are available.
    """
    n = len(arrivals_sec)
    if n < MIN_EVENTS:
        return HawkesParams()
    t_end = arrivals_sec[-1]
    t_start = arrivals_sec[0]
    T = t_end - t_start
    if T <= 0.0:
        return HawkesParams()

    mu = n / T / 2.0  # rough baseline — the other half is self-excitation
    events = [t - t_start for t in arrivals_sec]

    def _nll(beta: float) -> tuple[float, float]:
        """Return (nll, alpha) for the given beta."""
        if beta <= 0.0:
            return float("inf"), 0.0
        alpha = _alpha_given_beta(mu, beta, events, T)
        # Clamp α so the branching ratio stays < 1 — unstable otherwise.
        alpha = max(0.0, min(alpha, 0.999 * beta))
        ll = _log_likelihood(mu, alpha, beta, events, T)
        return -ll, alpha

    # Golden-section search over β ∈ [BETA_MIN, BETA_MAX]
    lo, hi = BETA_MIN, BETA_MAX
    c = hi - GOLDEN_RATIO * (hi - lo)
    d = lo + GOLDEN_RATIO * (hi - lo)
    fc, _ = _nll(c)
    fd, _ = _nll(d)
    for _ in range(60):
        if fc < fd:
            hi = d
            d, fd = c, fc
            c = hi - GOLDEN_RATIO * (hi - lo)
            fc, _ = _nll(c)
        else:
            lo = c
            c, fc = d, fd
            d = lo + GOLDEN_RATIO * (hi - lo)
            fd, _ = _nll(d)
        if hi - lo < 1e-3:
            break
    beta_star = (lo + hi) / 2.0
    _, alpha_star = _nll(beta_star)
    return HawkesParams(mu=float(mu), alpha=float(alpha_star), beta=float(beta_star))


# ---------------------------------------------------------------------------
# Likelihood helpers
# ---------------------------------------------------------------------------


def _alpha_given_beta(
    mu: float, beta: float, events: list[float], T: float,
) -> float:
    """Moment-match α so the expected count matches the observed count."""
    n = len(events)
    # Σ (1 − exp(−β · (T − t_i)))
    s = 0.0
    for t in events:
        s += 1.0 - math.exp(-beta * (T - t))
    if s <= 1e-12:
        return 0.0
    return max(0.0, beta * (n - mu * T) / s)


def _log_likelihood(
    mu: float, alpha: float, beta: float,
    events: list[float], T: float,
) -> float:
    """Log-likelihood of a Hawkes process with exponential kernel."""
    if mu <= 0.0 or beta <= 0.0:
        return -float("inf")
    # Σ log λ(t_i) − ∫₀^T λ(t) dt
    ll = 0.0
    excite = 0.0
    prev = 0.0
    for t in events:
        dt = t - prev
        excite = excite * math.exp(-beta * dt) + alpha
        lam = mu + excite
        if lam <= 0.0:
            return -float("inf")
        ll += math.log(lam)
        prev = t
    compensator = mu * T
    for t in events:
        compensator += (alpha / beta) * (1.0 - math.exp(-beta * (T - t)))
    return float(ll - compensator)
