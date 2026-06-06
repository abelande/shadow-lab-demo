"""Tests for p6lab.ingestion.event_windowing — pluggable window iterators."""
from __future__ import annotations

from dataclasses import dataclass

from p6lab.ingestion.event_windowing import (
    BurstDetector,
    BurstDetectorConfig,
    BurstAnchoredIterator,
    FixedHorizonIterator,
    FixedLengthIterator,
    WindowAnchorStrategy,
    WindowIterator,
)


@dataclass
class _E:
    timestamp_ms: int


def _events(*times) -> list[_E]:
    return [_E(t) for t in times]


class TestBurstDetector:
    def test_fires_when_threshold_met(self):
        d = BurstDetector(BurstDetectorConfig(min_events_per_100ms=5, min_burst_gap_ms=0))
        # 5 events in window
        assert d.is_burst([1, 2, 3, 4, 5], window_start_ms=1000) is True

    def test_does_not_fire_below_threshold(self):
        d = BurstDetector(BurstDetectorConfig(min_events_per_100ms=5))
        assert d.is_burst([1, 2], 1000) is False

    def test_gap_suppression(self):
        d = BurstDetector(BurstDetectorConfig(min_events_per_100ms=3, min_burst_gap_ms=1000))
        # first burst at t=0 fires
        assert d.is_burst([1, 2, 3], 0) is True
        # second burst at t=500 suppressed (gap=500 < 1000)
        assert d.is_burst([1, 2, 3], 500) is False
        # burst at t=2000 fires (gap=2000)
        assert d.is_burst([1, 2, 3], 2000) is True


class TestBurstAnchored:
    def test_emits_windows_around_burst(self):
        # Dense burst: 10 events in 100ms
        events = _events(*range(1000, 1091, 10))  # 10 events at 10ms intervals → fits in 100ms
        # sparse before/after
        events = _events(100) + events + _events(5000)
        cfg = BurstDetectorConfig(min_events_per_100ms=5, min_burst_gap_ms=0,
                                   lookback_ms=200, lookahead_ms=500)
        it = BurstAnchoredIterator(events, cfg)
        windows = list(it)
        assert len(windows) >= 1
        w = windows[0]
        # window should span anchor ± lookback/lookahead
        assert w.start_ms < w.anchor_ms <= w.end_ms

    def test_no_burst_no_windows(self):
        events = _events(0, 10_000, 20_000, 30_000)  # sparse
        cfg = BurstDetectorConfig(min_events_per_100ms=5)
        it = BurstAnchoredIterator(events, cfg)
        assert list(it) == []


class TestFixedHorizon:
    def test_one_window_per_event(self):
        events = _events(0, 100, 200)
        it = FixedHorizonIterator(events, horizon_ms=1000)
        windows = list(it)
        assert len(windows) == 3

    def test_window_includes_forward_events(self):
        events = _events(0, 500, 1500)
        it = FixedHorizonIterator(events, horizon_ms=1000)
        windows = list(it)
        # anchor=0, horizon=1000 → events at 0, 500 (both ≤1000)
        assert len(windows[0].events) == 2


class TestFixedLength:
    def test_produces_correct_count(self):
        events = _events(*range(0, 10_000, 100))  # 100 events over 10s
        it = FixedLengthIterator(events, window_ms=1000, stride_ms=1000)
        windows = list(it)
        # 10s / 1s stride → 9 full non-overlapping windows (last start at 9000 → end 10000 == last ts)
        assert len(windows) >= 9

    def test_overlapping_stride(self):
        events = _events(*range(0, 2000, 100))
        it = FixedLengthIterator(events, window_ms=1000, stride_ms=500)
        windows = list(it)
        # starts at 0, 500, 1000, 1500 (last window extends past last ts = 1900)
        assert len(windows) == 4
        assert [w.start_ms for w in windows] == [0, 500, 1000, 1500]
        # first window contains 10 events (0..900), stride ensures overlap
        assert len(windows[0].events) == 10
        assert len(windows[1].events) == 10  # 500..1400

    def test_empty_events(self):
        assert list(FixedLengthIterator([], 1000, 500)) == []


class TestFactory:
    def test_create_burst(self):
        it = WindowIterator.create(WindowAnchorStrategy.BURST_ANCHORED, _events(0))
        assert isinstance(it, BurstAnchoredIterator)

    def test_create_fixed_length(self):
        it = WindowIterator.create(
            WindowAnchorStrategy.FIXED_LENGTH, _events(0),
            window_ms=1000, stride_ms=500,
        )
        assert isinstance(it, FixedLengthIterator)
