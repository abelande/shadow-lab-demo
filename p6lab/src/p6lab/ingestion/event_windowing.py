"""
p6lab.ingestion.event_windowing — Pluggable event window iterator.

Spec: p6-notebook-lab-spec.md §3.2

WindowIterator with three anchoring strategies:
  - burst_anchored   → pattern mining (notebook 04 §02)
  - fixed_horizon    → forward-outcome labeling (§5.3)
  - fixed_length     → correlation grids (notebook 06)

Ref: OB-reference.md burst windowing patterns (L470-487)
"""

from __future__ import annotations

import bisect
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WindowAnchorStrategy(Enum):
    BURST_ANCHORED = "burst_anchored"
    FIXED_HORIZON = "fixed_horizon"
    FIXED_LENGTH = "fixed_length"


@dataclass
class Window:
    window_id: int
    start_ms: int
    end_ms: int
    anchor_ms: int
    events: list
    anchor_strategy: WindowAnchorStrategy
    metadata: dict


@dataclass
class BurstDetectorConfig:
    min_events_per_100ms: int = 5
    lookback_ms: int = 500
    lookahead_ms: int = 2_000
    min_burst_gap_ms: int = 1_000


def _event_ts(e) -> int:
    """Extract timestamp_ms from event (dict, attribute, or (ts, ...) tuple)."""
    if isinstance(e, tuple):
        return int(e[0])
    if isinstance(e, dict):
        return int(e.get("timestamp_ms", 0))
    return int(getattr(e, "timestamp_ms", 0))


class BurstDetector:
    """Detects activity bursts in the MBO event stream."""

    def __init__(self, config: BurstDetectorConfig | None = None) -> None:
        self.config = config or BurstDetectorConfig()
        self._last_burst_ms: int = -10**18  # far past

    def reset(self) -> None:
        self._last_burst_ms = -10**18

    def is_burst(self, events_in_window, window_start_ms: int) -> bool:
        """True if ``events_in_window`` has enough events AND we're past
        the gap-suppression window. ``events_in_window`` may be a list
        OR an int count — callers in hot loops should pass the count
        directly to avoid materializing the list."""
        count = events_in_window if isinstance(events_in_window, int) else len(events_in_window)
        if count < self.config.min_events_per_100ms:
            return False
        if (window_start_ms - self._last_burst_ms) < self.config.min_burst_gap_ms:
            return False
        self._last_burst_ms = window_start_ms
        return True


class WindowIterator(ABC):
    """Abstract base for all windowing strategies. Spec §3.2."""

    @classmethod
    def create(
        cls,
        strategy: WindowAnchorStrategy,
        events: list,
        *,
        window_ms: int = 2_000,
        stride_ms: int = 500,
        burst_config: BurstDetectorConfig | None = None,
        horizon_ms: int | None = None,
    ) -> "WindowIterator":
        if strategy == WindowAnchorStrategy.BURST_ANCHORED:
            return BurstAnchoredIterator(events, burst_config or BurstDetectorConfig())
        if strategy == WindowAnchorStrategy.FIXED_HORIZON:
            return FixedHorizonIterator(events, horizon_ms or 60_000)
        if strategy == WindowAnchorStrategy.FIXED_LENGTH:
            return FixedLengthIterator(events, window_ms, stride_ms)
        raise ValueError(f"Unknown strategy: {strategy}")

    @abstractmethod
    def __iter__(self) -> Iterator[Window]:
        ...


class BurstAnchoredIterator(WindowIterator):
    """
    Burst-anchored windowing for pattern mining (notebook 04 §02).

    Implementation:
      - Maintain a 100ms sliding deque of recent events by timestamp.
      - On each event, evaluate burst condition. If it fires, emit
        a Window covering [anchor - lookback_ms, anchor + lookahead_ms].
      - Gap suppression handled inside BurstDetector.
    """

    def __init__(self, events: list, burst_config: BurstDetectorConfig) -> None:
        # sort to guarantee monotonic scan
        self._events = sorted(events, key=_event_ts)
        self._burst_config = burst_config
        self._burst_detector = BurstDetector(burst_config)
        self._timestamps = [_event_ts(e) for e in self._events]

    def __iter__(self) -> Iterator[Window]:
        self._burst_detector.reset()
        deq: deque[int] = deque()  # indices into self._events
        window_id = 0
        burst_window_ms = 100  # intensity measured over a rolling 100ms
        cfg = self._burst_config

        timestamps = self._timestamps
        events = self._events
        # Hot loop — avoid per-event list allocation. Pass deque LENGTH
        # to the burst detector, only materialize event slices when a
        # burst actually fires.
        for i in range(len(events)):
            ts = timestamps[i]
            deq.append(i)
            while deq and (ts - timestamps[deq[0]]) > burst_window_ms:
                deq.popleft()
            if self._burst_detector.is_burst(len(deq), ts):
                start_ms = ts - cfg.lookback_ms
                end_ms = ts + cfg.lookahead_ms
                lo = bisect.bisect_left(timestamps, start_ms)
                hi = bisect.bisect_right(timestamps, end_ms)
                yield Window(
                    window_id=window_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    anchor_ms=ts,
                    events=events[lo:hi],
                    anchor_strategy=WindowAnchorStrategy.BURST_ANCHORED,
                    metadata={
                        "burst_intensity": len(deq) / (burst_window_ms / 1000.0),
                        "anchor_index": i,
                    },
                )
                window_id += 1


class FixedHorizonIterator(WindowIterator):
    """
    Fixed-horizon windowing for forward-outcome labeling (spec §5.3).
    Anchored at each event; window extends horizon_ms into the future.
    """

    def __init__(self, events: list, horizon_ms: int) -> None:
        self._events = sorted(events, key=_event_ts)
        self._horizon_ms = horizon_ms
        self._timestamps = [_event_ts(e) for e in self._events]

    def __iter__(self) -> Iterator[Window]:
        for i, ev in enumerate(self._events):
            anchor = self._timestamps[i]
            end_ms = anchor + self._horizon_ms
            hi = bisect.bisect_right(self._timestamps, end_ms)
            yield Window(
                window_id=i,
                start_ms=anchor,
                end_ms=end_ms,
                anchor_ms=anchor,
                events=list(self._events[i:hi]),
                anchor_strategy=WindowAnchorStrategy.FIXED_HORIZON,
                metadata={"anchor_index": i},
            )


class FixedLengthIterator(WindowIterator):
    """
    Fixed-length sliding window for correlation feature grids (notebook 06).
    All windows are exactly window_ms long; stride_ms controls overlap.
    """

    def __init__(self, events: list, window_ms: int, stride_ms: int) -> None:
        if window_ms <= 0:
            raise ValueError(f"window_ms must be positive, got {window_ms}")
        if stride_ms <= 0:
            raise ValueError(f"stride_ms must be positive, got {stride_ms}")
        self._events = sorted(events, key=_event_ts)
        self._window_ms = window_ms
        self._stride_ms = stride_ms
        self._timestamps = [_event_ts(e) for e in self._events]

    def __iter__(self) -> Iterator[Window]:
        if not self._events:
            return
        start_ms = self._timestamps[0]
        end_cutoff = self._timestamps[-1]
        window_id = 0
        cur = start_ms
        # Emit all windows whose start is within the observed event span;
        # windows may extend past the last timestamp (fewer events, same length).
        while cur <= end_cutoff:
            win_end = cur + self._window_ms
            lo = bisect.bisect_left(self._timestamps, cur)
            hi = bisect.bisect_left(self._timestamps, win_end)
            yield Window(
                window_id=window_id,
                start_ms=cur,
                end_ms=win_end,
                anchor_ms=cur,
                events=list(self._events[lo:hi]),
                anchor_strategy=WindowAnchorStrategy.FIXED_LENGTH,
                metadata={},
            )
            window_id += 1
            cur += self._stride_ms
