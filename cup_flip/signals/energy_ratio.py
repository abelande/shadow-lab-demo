"""Short-term vs long-term RMS energy ratio for acceleration detection.

Adapted from p5-dsp-signal-lab/features/energy.py. Pure Python, no numpy.

Ratio > 1.0 = energy (pressure) is accelerating (breakout).
Ratio < 1.0 = energy is decelerating (exhaustion / consolidation).
Ratio ≈ 1.0 = steady state.
"""
from __future__ import annotations

import math
from collections import deque


class EnergyRatio:
    """Rolling RMS energy ratio: short_rms / long_rms."""

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self._short_window = short_window
        self._long_window = long_window
        self._values: deque[float] = deque(maxlen=long_window)

    def update(self, value: float) -> float:
        """Feed a new pressure / fill-rate value. Returns the energy ratio."""
        self._values.append(value)
        if len(self._values) < self._long_window:
            return 1.0  # warmup: neutral

        short = list(self._values)[-self._short_window:]
        short_rms = math.sqrt(sum(v * v for v in short) / len(short))

        all_vals = list(self._values)
        long_rms = math.sqrt(sum(v * v for v in all_vals) / len(all_vals))

        if long_rms < 1e-12:
            return 1.0
        return short_rms / long_rms

    def reset(self) -> None:
        self._values.clear()
