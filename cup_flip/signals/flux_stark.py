"""FLUX + STARK regime shift detection for streak exhaustion.

Based on STARK σ.3 (Wasserstein distance) from Armory indicator analysis.
Rewritten for online operation on a rolling mid-price buffer.

FLUX  = Wasserstein-1 distance between recent and baseline log-return
        distributions. High FLUX = structural shift in the return
        generating process. Includes transport direction (signed: positive
        = distribution shifting bullish) and tail asymmetry.
STARK = d(FLUX)/dt — rate of change of FLUX.
        STARK > 0 = shift still accelerating.
        STARK < 0 = shift decelerating (exhaustion).

The cup_flip exhaustion_detector consumes (flux, stark) to detect when
a streak is running out of momentum.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class FluxStarkTracker:
    """Online FLUX (Wasserstein distance) and STARK (d(FLUX)/dt) computation."""

    def __init__(
        self,
        flux_window: int = 30,
        baseline_window: int = 120,
        stark_smoothing: int = 5,
    ) -> None:
        self._flux_window = flux_window
        self._baseline_window = baseline_window
        self._stark_smooth = stark_smoothing

        self._mid_prices: deque[float] = deque(maxlen=baseline_window + flux_window + 1)
        self._flux_history: deque[float] = deque(maxlen=stark_smoothing + 1)
        self._transport_direction: float = 0.0
        self._tail_asymmetry: float = 0.0

    @property
    def transport_direction(self) -> float:
        """Signed shift direction. Positive = distribution moving bullish."""
        return self._transport_direction

    @property
    def tail_asymmetry(self) -> float:
        """Upper vs lower tail migration differential."""
        return self._tail_asymmetry

    def update(self, mid_price: float) -> tuple[float, float]:
        """Feed a mid price. Returns (flux, stark).

        Returns (0.0, 0.0) during warmup (need baseline_window + flux_window + 1 prices).
        """
        self._mid_prices.append(mid_price)
        # Need N+1 prices to get N log returns
        total_needed = self._baseline_window + self._flux_window + 1
        if len(self._mid_prices) < total_needed:
            return 0.0, 0.0

        prices = np.array(self._mid_prices)
        log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))

        baseline = log_returns[: self._baseline_window]
        recent = log_returns[-self._flux_window :]

        flux = self._wasserstein_1d(recent, baseline)
        self._flux_history.append(flux)

        stark = 0.0
        if len(self._flux_history) >= 2:
            diffs = [
                self._flux_history[i] - self._flux_history[i - 1]
                for i in range(1, len(self._flux_history))
            ]
            stark = sum(diffs) / len(diffs)

        return flux, stark

    def _wasserstein_1d(self, recent: np.ndarray, baseline: np.ndarray) -> float:
        """Wasserstein-1 distance between two 1-D empirical distributions.

        Also computes transport direction and tail asymmetry as side effects.
        W_1 for 1-D empirical distributions = integral of |F_recent - F_baseline|,
        computed via sorted quantile matching.
        """
        r_sorted = np.sort(recent)
        b_sorted = np.sort(baseline)

        # Interpolate both to common quantile grid for comparison
        n_quantiles = max(len(r_sorted), len(b_sorted))
        quantiles = np.linspace(0, 1, n_quantiles, endpoint=False) + 0.5 / n_quantiles

        r_interp = np.interp(quantiles, np.linspace(0, 1, len(r_sorted), endpoint=False) + 0.5 / len(r_sorted), r_sorted)
        b_interp = np.interp(quantiles, np.linspace(0, 1, len(b_sorted), endpoint=False) + 0.5 / len(b_sorted), b_sorted)

        # Wasserstein-1: mean absolute difference of quantile functions
        transport = r_interp - b_interp
        w1 = float(np.mean(np.abs(transport)))

        # Transport direction: signed mean shift (positive = recent > baseline)
        self._transport_direction = float(np.mean(transport))

        # Tail asymmetry: upper tail shift vs lower tail shift
        n10 = max(1, len(quantiles) // 10)
        upper_shift = float(np.mean(transport[-n10:]))
        lower_shift = float(np.mean(transport[:n10]))
        self._tail_asymmetry = upper_shift - lower_shift

        return w1

    def reset(self) -> None:
        self._mid_prices.clear()
        self._flux_history.clear()
        self._transport_direction = 0.0
        self._tail_asymmetry = 0.0
