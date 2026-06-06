"""Online Markov LOB state tracker with learned transition probabilities.

Adapted from p4-clones/mesh/markov_lob.py. Rewritten for p6-v2 types
and online (incremental) learning — fits the transition matrix from
each snapshot rather than requiring a pre-fit batch.

Classifies each LOB snapshot into one of K discrete states based on
(bid_depth / ask_depth ratio, normalized spread). Maintains a transition
matrix and outputs P(price_up | current_state).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...models import OrderBookSnapshot


class MarkovStateTracker:
    """Online Markov state machine for LOB microstructure.

    States are defined by a 2D grid: (depth_ratio_bin, spread_bin).
    Transition matrix and P(up|state) are learned incrementally.
    """

    def __init__(
        self,
        n_depth_bins: int = 3,
        n_spread_bins: int = 2,
        min_observations: int = 50,
    ) -> None:
        self._n_depth = n_depth_bins
        self._n_spread = n_spread_bins
        self._n_states = n_depth_bins * n_spread_bins
        self._min_obs = min_observations

        # Depth ratio thresholds: [0, 0.4, 0.6, 1.0] for 3 bins
        step = 1.0 / n_depth_bins
        self._depth_thresholds = [i * step for i in range(n_depth_bins + 1)]

        # Transition counts: (from_state, to_state)
        self._transition_counts = np.zeros(
            (self._n_states, self._n_states), dtype=np.float64
        )
        # Price-up counts per state
        self._up_counts = np.zeros(self._n_states, dtype=np.float64)
        self._state_counts = np.zeros(self._n_states, dtype=np.float64)

        self._prev_state: Optional[int] = None
        self._prev_mid: Optional[float] = None
        self._total_observations: int = 0

    def update(self, snapshot: OrderBookSnapshot) -> float:
        """Process a snapshot. Returns P(price_up | current_state) in [0, 1].

        Returns 0.5 (neutral) until min_observations have been seen.
        """
        state = self._classify(snapshot)
        mid = snapshot.mid_price

        # Learn from transition
        if self._prev_state is not None and self._prev_mid is not None and mid is not None:
            self._transition_counts[self._prev_state, state] += 1.0
            self._state_counts[self._prev_state] += 1.0
            if mid > self._prev_mid:
                self._up_counts[self._prev_state] += 1.0

        self._prev_state = state
        self._prev_mid = mid
        self._total_observations += 1

        # Return learned probability or neutral
        if self._total_observations < self._min_obs:
            return 0.5
        count = self._state_counts[state]
        if count < 5:
            return 0.5
        return float(self._up_counts[state] / count)

    def _classify(self, snapshot: OrderBookSnapshot) -> int:
        """Map snapshot to discrete state index."""
        bid_vol = sum(lv.volume for lv in snapshot.bids[:5]) if snapshot.bids else 0.0
        ask_vol = sum(lv.volume for lv in snapshot.asks[:5]) if snapshot.asks else 0.0
        total = bid_vol + ask_vol
        depth_ratio = bid_vol / total if total > 0 else 0.5

        spread = snapshot.spread or 0.0
        mid = snapshot.mid_price or 1.0
        spread_bps = (spread / mid) * 10000.0 if mid > 0 else 0.0

        # Bin depth ratio
        depth_bin = 0
        for i, threshold in enumerate(self._depth_thresholds[1:]):
            if depth_ratio <= threshold:
                depth_bin = i
                break
        else:
            depth_bin = self._n_depth - 1

        # Bin spread: simple binary — tight vs wide relative to 2 bps
        spread_bin = 0 if spread_bps <= 2.0 else min(int(spread_bps / 2.0), self._n_spread - 1)

        return depth_bin * self._n_spread + spread_bin

    @property
    def transition_matrix(self) -> np.ndarray:
        """Normalized transition probability matrix."""
        row_sums = self._transition_counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return self._transition_counts / row_sums

    def reset(self) -> None:
        self._transition_counts[:] = 0
        self._up_counts[:] = 0
        self._state_counts[:] = 0
        self._prev_state = None
        self._prev_mid = None
        self._total_observations = 0
