"""Wave 9 §H.1.a / A5 — tests for OnlineActivityDetector."""
from __future__ import annotations

import pytest

from p6lab.live.activity_detector import (
    ActivityDetectorConfig, OnlineActivityDetector,
)


class TestOnlineActivityDetector:
    def test_first_update_returns_false(self) -> None:
        """No prior price → CUSUM has nothing to compute → not active."""
        det = OnlineActivityDetector(ActivityDetectorConfig())
        assert det.update(100.0, 0) is False

    def test_inactive_until_event_fires(self) -> None:
        """Below-threshold drift should not trip the detector."""
        det = OnlineActivityDetector(ActivityDetectorConfig(
            cusum_threshold=1.0, lookback_ms=60_000,
        ))
        # Tiny price changes accumulate but stay below threshold
        active_states = [
            det.update(100.0 + 0.01 * i, i * 100) for i in range(50)
        ]
        assert not any(active_states)
        assert det.event_count == 0

    def test_event_fires_on_cumulative_drift(self) -> None:
        """A monotonic rise crossing threshold fires an event."""
        det = OnlineActivityDetector(ActivityDetectorConfig(
            cusum_threshold=1.0, lookback_ms=60_000,
        ))
        # Step by 0.1 each time; threshold 1.0 → event at step ~10
        first_active_idx = None
        for i in range(20):
            active = det.update(100.0 + 0.1 * i, i * 100)
            if active and first_active_idx is None:
                first_active_idx = i
        assert det.event_count >= 1
        assert first_active_idx is not None
        assert first_active_idx <= 12  # within a few of expected boundary

    def test_active_window_expires_after_lookback(self) -> None:
        """After lookback_ms with no new events, snapshots become inactive."""
        det = OnlineActivityDetector(ActivityDetectorConfig(
            cusum_threshold=1.0, lookback_ms=5_000,  # 5s window
        ))
        # Cause a single event at t≈1000ms
        for i in range(15):
            det.update(100.0 + 0.1 * i, i * 100)
        assert det.event_count == 1

        # Immediately after the event: active
        assert det.is_active_at(1_500) is True
        # Within window: active
        assert det.is_active_at(5_500) is True
        # Past lookback: inactive
        assert det.is_active_at(10_000) is False

    def test_negative_drift_fires_negative_arm(self) -> None:
        det = OnlineActivityDetector(ActivityDetectorConfig(
            cusum_threshold=1.0, lookback_ms=60_000,
        ))
        # Falling series — should trigger via the -threshold arm
        for i in range(15):
            det.update(100.0 - 0.1 * i, i * 100)
        assert det.event_count >= 1

    def test_alternating_signs_no_event(self) -> None:
        """Cancelling diffs should leave both accumulators near zero."""
        det = OnlineActivityDetector(ActivityDetectorConfig(
            cusum_threshold=1.0, lookback_ms=60_000,
        ))
        for i in range(30):
            sign = 1 if i % 2 == 0 else -1
            det.update(100.0 + 0.05 * sign, i * 100)
        assert det.event_count == 0

    def test_reset_clears_state(self) -> None:
        det = OnlineActivityDetector(ActivityDetectorConfig(
            cusum_threshold=1.0, lookback_ms=60_000,
        ))
        for i in range(15):
            det.update(100.0 + 0.1 * i, i * 100)
        assert det.event_count >= 1
        det.reset()
        assert det.event_count == 0
        assert det.update(100.0, 9999) is False  # like first call again

    def test_is_active_at_pure_query_does_not_advance_state(self) -> None:
        """is_active_at() must not affect CUSUM accumulators."""
        det = OnlineActivityDetector(ActivityDetectorConfig(
            cusum_threshold=1.0, lookback_ms=60_000,
        ))
        # Drive an event
        for i in range(15):
            det.update(100.0 + 0.1 * i, i * 100)
        events_before = det.event_count
        # Many is_active_at calls
        for ts in range(0, 100_000, 100):
            det.is_active_at(ts)
        assert det.event_count == events_before  # unchanged

    def test_event_count_grows_with_repeated_drifts(self) -> None:
        """Event count tracks the number of CUSUM crossings."""
        det = OnlineActivityDetector(ActivityDetectorConfig(
            cusum_threshold=0.5, lookback_ms=60_000,
        ))
        # Step 0.1 per row, threshold 0.5 → ~event every 5 rows over 50 rows
        for i in range(50):
            det.update(100.0 + 0.1 * i, i * 100)
        assert det.event_count >= 5  # multiple events fired
