"""Detect stop-runs: one side overwhelms and clears M levels without resistance."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from ..models import Side
from .streak_detector import Streak


@dataclass
class StopRunSignal:
    aggressor_side: Side
    levels_cleared: int
    velocity: float
    confidence: float


class StopRunDetector:
    def __init__(self, levels_threshold: int = 3, velocity_threshold: float = 2.0):
        self.levels_threshold = levels_threshold
        self.velocity_threshold = velocity_threshold

    def detect(
        self,
        streak: Streak | None,
        kalman_velocity: Optional[float] = None,
        kalman_velocity_std: Optional[float] = None,
    ) -> StopRunSignal | None:
        if streak is None:
            return None
        if streak.depth < self.levels_threshold:
            return None
        if streak.velocity < self.velocity_threshold:
            return None
        # If ask side consumed => buyer aggression => bullish stop run
        aggressor = Side.BID if streak.side == Side.ASK else Side.ASK
        depth_score = min(1.0, streak.depth / (self.levels_threshold * 2))
        vel_score = min(1.0, streak.velocity / (self.velocity_threshold * 2))
        conf = 0.6 * depth_score + 0.4 * vel_score

        # Kalman velocity gate: only applies when real Kalman data is provided.
        # Both kalman_velocity and kalman_velocity_std must be non-None.
        # If the Kalman filter says velocity is within 2σ of normal, this is
        # gradual unwinding rather than a real stop run — reduce confidence.
        if (
            kalman_velocity is not None
            and kalman_velocity_std is not None
            and kalman_velocity_std > 0
            and abs(kalman_velocity) < 2.0 * kalman_velocity_std
        ):
            conf *= 0.5

        return StopRunSignal(aggressor_side=aggressor, levels_cleared=streak.depth, velocity=streak.velocity, confidence=conf)
