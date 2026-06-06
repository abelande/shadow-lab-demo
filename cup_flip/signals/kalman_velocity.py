"""Simplified Kalman filter for price velocity estimation.

Adapted from p5-dsp-signal-lab/filtering/kalman.py. Stripped to the
essential 2×2 state-space [price, velocity] without bands or adaptive
noise (those are refinements for later).

State: x = [price, velocity]
Observation: z = mid_price
Velocity output tells cup_flip whether price momentum is accelerating
or decelerating — a direct input to streak exhaustion and stop-run
confirmation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class KalmanVelocity:
    """Online 1D Kalman filter tracking [price, velocity].

    Call update(mid_price) per snapshot. Returns (filtered_price, velocity).
    Velocity > 0 = price rising, velocity < 0 = price falling.
    The magnitude indicates rate of change in price units per snapshot interval.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        self._dt = dt

        # State transition: constant velocity model
        self._F = np.array([[1.0, dt], [0.0, 1.0]])
        # Observation matrix: we only see price
        self._H = np.array([[1.0, 0.0]])

        # Process noise
        self._Q = process_noise * np.array([
            [dt**3 / 3, dt**2 / 2],
            [dt**2 / 2, dt],
        ])
        # Measurement noise
        self._R = np.array([[measurement_noise]])

        # State estimate and covariance
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None

    def update(self, price: float) -> tuple[float, float]:
        """Observe a new price, return (filtered_price, velocity)."""
        z = np.array([[price]])

        if self._x is None:
            # Initialize on first observation
            self._x = np.array([[price], [0.0]])
            self._P = np.eye(2) * 100.0  # high initial uncertainty
            return price, 0.0

        # Predict
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # Update
        y = z - self._H @ x_pred                          # innovation
        S = self._H @ P_pred @ self._H.T + self._R        # innovation covariance
        K = P_pred @ self._H.T @ np.linalg.inv(S)         # Kalman gain
        self._x = x_pred + K @ y
        I = np.eye(2)
        self._P = (I - K @ self._H) @ P_pred

        filtered_price = float(self._x[0, 0])
        velocity = float(self._x[1, 0])
        return filtered_price, velocity

    @property
    def velocity_std(self) -> float:
        """Standard deviation of the velocity estimate from the covariance matrix."""
        if self._P is None:
            return 1.0
        return float(np.sqrt(max(self._P[1, 1], 0.0)))

    def reset(self) -> None:
        self._x = None
        self._P = None
