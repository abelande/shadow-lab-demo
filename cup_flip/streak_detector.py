"""
Streak Detector — consecutive fills on one side of the book.

A streak means one side is getting eaten: consecutive ask fills = bull streak,
consecutive bid fills = bear streak. Track length, velocity, depth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from ..models import Order, Side, OrderAction


@dataclass
class Streak:
    side: Side            # which side is being consumed (ASK = bull, BID = bear)
    fills: List[Order] = field(default_factory=list)
    start_price: float = 0.0
    end_price: float = 0.0
    start_ms: int = 0
    end_ms: int = 0

    @property
    def length(self) -> int:
        return len(self.fills)

    @property
    def depth(self) -> int:
        """Number of distinct price levels consumed."""
        return len(set(f.price for f in self.fills))

    @property
    def velocity(self) -> float:
        """Levels per second consumed."""
        duration_s = (self.end_ms - self.start_ms) / 1000.0
        if duration_s <= 0:
            return 999.0 if self.depth > 0 else 0.0
        return self.depth / duration_s

    @property
    def total_volume(self) -> float:
        return sum(f.size for f in self.fills)

    @property
    def volume_weighted_strength(self) -> float:
        """Size-weighted streak strength with recency decay.

        Recent heavy fills contribute more than old small fills.
        Decay factor = 0.9^(distance_from_latest).
        """
        if not self.fills:
            return 0.0
        n = len(self.fills)
        return sum(f.size * (0.9 ** (n - 1 - i)) for i, f in enumerate(self.fills))


class StreakDetector:
    """Detects consecutive fills on one side of the book.

    Supports a gap_tolerance so N opposite-side fills don't immediately
    reset the streak (noise immunity).
    """

    def __init__(self, min_streak_length: int = 3, gap_tolerance: int = 1):
        self.min_streak_length = min_streak_length
        self.gap_tolerance = gap_tolerance
        self._current: Optional[Streak] = None
        self._completed: List[Streak] = []
        self._gap_count: int = 0  # consecutive opposing fills since last main-side fill

    @property
    def current_streak(self) -> Optional[Streak]:
        return self._current

    @property
    def completed_streaks(self) -> List[Streak]:
        return self._completed

    def process_fill(self, fill: Order) -> Optional[Streak]:
        """
        Process a fill event. Returns a completed streak if one just ended,
        or None if the streak continues / a new one starts.
        """
        fill_side = fill.side  # side of the resting order that got filled

        if self._current is None:
            # Start new streak
            self._current = Streak(
                side=fill_side,
                fills=[fill],
                start_price=fill.price,
                end_price=fill.price,
                start_ms=fill.timestamp_ms,
                end_ms=fill.timestamp_ms,
            )
            self._gap_count = 0
            return None

        if fill_side == self._current.side:
            # Continue streak — reset gap counter
            self._current.fills.append(fill)
            self._current.end_price = fill.price
            self._current.end_ms = fill.timestamp_ms
            self._gap_count = 0
            return None
        else:
            # Opposing fill — check gap tolerance
            self._gap_count += 1
            if self._gap_count <= self.gap_tolerance:
                # Tolerate this opposing fill: absorb it into the streak
                self._current.fills.append(fill)
                return None

            # Tolerance exceeded — close current streak
            completed = None
            if self._current.length >= self.min_streak_length:
                completed = self._current
                self._completed.append(completed)

            # Start new streak on opposite side
            self._current = Streak(
                side=fill_side,
                fills=[fill],
                start_price=fill.price,
                end_price=fill.price,
                start_ms=fill.timestamp_ms,
                end_ms=fill.timestamp_ms,
            )
            self._gap_count = 0
            return completed

    def reset(self):
        self._current = None
        self._completed = []
        self._gap_count = 0
