"""Shannon entropy of rolling pressure values for confidence gating.

Adapted from p2-dedup/paired_systems/cairo_entropy_pair.py. Pure Python.

When pressure values are choppy / uniformly distributed → high entropy →
state machine confidence should be reduced (the pressure signal is
unreliable noise). When pressure values cluster in a narrow range →
low entropy → the signal is coherent.

Output range: [0.0, 1.0]. 0 = perfectly predictable, 1 = maximum disorder.
"""
from __future__ import annotations

import math
from collections import deque


class EntropyGate:
    """Rolling Shannon entropy of binned pressure values."""

    def __init__(self, window: int = 30, n_bins: int = 10) -> None:
        self._window = window
        self._n_bins = n_bins
        self._values: deque[float] = deque(maxlen=window)

    def update(self, pressure: float) -> float:
        """Feed a pressure value in [-1, +1]. Returns entropy in [0, 1]."""
        self._values.append(pressure)
        if len(self._values) < self._window:
            return 0.5  # warmup: moderately uncertain

        return self._compute_entropy()

    def _compute_entropy(self) -> float:
        # Bin the pressure values into n_bins equally spaced bins over [-1, 1]
        counts = [0] * self._n_bins
        for v in self._values:
            # Clamp to [-1, 1] then map to bin index
            clamped = max(-1.0, min(1.0, v))
            idx = int((clamped + 1.0) / 2.0 * (self._n_bins - 1))
            idx = min(idx, self._n_bins - 1)
            counts[idx] += 1

        n = len(self._values)
        max_entropy = math.log2(self._n_bins)
        if max_entropy <= 0:
            return 0.0

        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / n
                entropy -= p * math.log2(p)

        return entropy / max_entropy  # normalize to [0, 1]

    def reset(self) -> None:
        self._values.clear()
