"""
p6lab.features.markov_lob — Wave 7 Phase 7E

Markov model over discretized LOB states. Ported from
``p4-clones/mesh/markov_lob.py``. Produces two scalar features:

  1. ``markov_p_price_up_given_state`` — P(next-tick mid goes up |
     current discretized LOB state). Picked up at snapshot time by
     looking up the current state in the fitted transition table.
  2. ``markov_state_entropy`` — the Shannon entropy of the outgoing
     transition row for the current state, normalized to [0, 1]. Low
     entropy ⇒ deterministic successor state (regime); high entropy ⇒
     mixed successors (chop).

State discretization mirrors the source: depth_ratio ∈ {bid-heavy,
balanced, ask-heavy} × spread ∈ {tight, wide} → 6 states.

Exported:
    MARKOV_FEATURE_NAMES    tuple[str, ...]
    MarkovLOBState          dataclass (rolling)
    update_markov_lob(state, ts_ms, bid_depth, ask_depth, spread, mid)
    snapshot_markov_features(state) → dict
    classify_lob_state(bid_depth, ask_depth, spread) → int
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


N_DEPTH_BINS = 3
N_SPREAD_BINS = 2
N_STATES = N_DEPTH_BINS * N_SPREAD_BINS
SPREAD_TIGHT_CUTOFF = 0.03
WINDOW_SAMPLES = 500


MARKOV_FEATURE_NAMES: tuple[str, ...] = (
    "markov_p_price_up_given_state",
    "markov_state_entropy",
    "markov_current_state",
)


@dataclass
class MarkovLOBState:
    states: Deque[int] = field(default_factory=lambda: deque(maxlen=WINDOW_SAMPLES))
    mids: Deque[float] = field(default_factory=lambda: deque(maxlen=WINDOW_SAMPLES))
    transition_counts: list[list[int]] = field(
        default_factory=lambda: [[0] * N_STATES for _ in range(N_STATES)]
    )
    price_up_counts: list[int] = field(default_factory=lambda: [0] * N_STATES)
    state_visits: list[int] = field(default_factory=lambda: [0] * N_STATES)

    def reset(self) -> None:
        self.states.clear()
        self.mids.clear()
        self.transition_counts = [[0] * N_STATES for _ in range(N_STATES)]
        self.price_up_counts = [0] * N_STATES
        self.state_visits = [0] * N_STATES


# ---------------------------------------------------------------------------
# State discretization
# ---------------------------------------------------------------------------


def classify_lob_state(
    bid_depth: float, ask_depth: float, spread: float,
) -> int:
    """Map (bid_depth, ask_depth, spread) → integer state ∈ [0, 5]."""
    bid = max(float(bid_depth), 0.0)
    ask = max(float(ask_depth), 0.0)
    total = bid + ask
    ratio = 0.5 if total <= 0.0 else bid / total
    if ratio < 0.4:
        depth_bin = 2   # ask-heavy
    elif ratio < 0.6:
        depth_bin = 1   # balanced
    else:
        depth_bin = 0   # bid-heavy
    spread_bin = 0 if float(spread) < SPREAD_TIGHT_CUTOFF else 1
    return depth_bin * N_SPREAD_BINS + spread_bin


# ---------------------------------------------------------------------------
# Update + snapshot
# ---------------------------------------------------------------------------


def update_markov_lob(
    state: MarkovLOBState,
    *,
    ts_ms: int,
    bid_depth: float,
    ask_depth: float,
    spread: float,
    mid: float,
) -> None:
    """Ingest one (bid_depth, ask_depth, spread, mid) snapshot."""
    s = classify_lob_state(bid_depth, ask_depth, spread)
    prev_state = state.states[-1] if state.states else None
    prev_mid = state.mids[-1] if state.mids else None
    state.states.append(s)
    state.mids.append(float(mid))
    state.state_visits[s] += 1
    if prev_state is not None:
        state.transition_counts[prev_state][s] += 1
        if prev_mid is not None and mid > prev_mid:
            state.price_up_counts[prev_state] += 1


def snapshot_markov_features(state: MarkovLOBState) -> dict[str, float]:
    """Return the 3 scalar features at the current state."""
    if not state.states:
        return {
            "markov_p_price_up_given_state": 0.5,
            "markov_state_entropy": 1.0,
            "markov_current_state": 0.0,
        }
    s = state.states[-1]
    row = state.transition_counts[s]
    total = sum(row)
    if total <= 0:
        return {
            "markov_p_price_up_given_state": 0.5,
            "markov_state_entropy": 1.0,
            "markov_current_state": float(s),
        }
    p_up = state.price_up_counts[s] / max(state.state_visits[s], 1)
    # Shannon entropy of outgoing row, normalized to [0, 1].
    probs = [c / total for c in row if c > 0]
    if not probs:
        entropy = 0.0
    else:
        ent = -sum(p * math.log(p) for p in probs)
        entropy = ent / math.log(len(row)) if len(row) > 1 else 0.0
    return {
        "markov_p_price_up_given_state": float(p_up),
        "markov_state_entropy": float(entropy),
        "markov_current_state": float(s),
    }
