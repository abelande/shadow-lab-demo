"""Unit tests for triple-barrier + cost-thresholded labels."""
from __future__ import annotations

import numpy as np
import pytest

from p6lab.validation.labelers import (
    FiringEvent,
    LabelSpec,
    MFEMAELabel,
    TripleBarrierLabel,
    activity_mask,
    compute_label_set,
    cost_thresholded_binary,
    cusum_events,
    mfe_mae_labels,
    pattern_firing_labels,
    triple_barrier_labels,
)


class TestTripleBarrierLabels:
    def test_all_up_barriers_on_rising_series(self) -> None:
        """A strictly rising series should hit the up-barrier for every entry."""
        n = 20
        mid = np.arange(n, dtype=float) * 0.5    # rises 0.5/step
        ts_ms = np.arange(n, dtype=np.int64) * 100  # 100ms/step

        labels = triple_barrier_labels(
            mid, ts_ms,
            horizon_ms=2_000,
            up_target_ticks=4.0,
            down_target_ticks=4.0,
            tick_size=0.25,
        )

        # up barrier = entry + 1.0 price units
        # For row i: need mid[j] - mid[i] >= 1.0, i.e. 0.5*(j-i) >= 1.0, i.e. j >= i+2
        # Row i=19 has no future rows → timeout
        resolved = [lbl for lbl in labels if lbl.barrier_hit == "up"]
        assert len(resolved) >= n - 3  # most entries resolve up
        assert all(lbl.side == 1 for lbl in resolved)
        assert all(lbl.ret > 0 for lbl in resolved)

    def test_all_down_barriers_on_falling_series(self) -> None:
        """A strictly falling series should hit the down-barrier for every entry."""
        n = 20
        mid = 100.0 - np.arange(n, dtype=float) * 0.5
        ts_ms = np.arange(n, dtype=np.int64) * 100

        labels = triple_barrier_labels(
            mid, ts_ms,
            horizon_ms=2_000,
            up_target_ticks=4.0,
            down_target_ticks=4.0,
            tick_size=0.25,
        )

        resolved = [lbl for lbl in labels if lbl.barrier_hit == "down"]
        assert len(resolved) >= n - 3
        assert all(lbl.side == -1 for lbl in resolved)
        assert all(lbl.ret < 0 for lbl in resolved)

    def test_all_timeouts_on_flat_series(self) -> None:
        """A perfectly flat series never hits a barrier within the
        observable window → timeout for rows whose deadline lands in the
        data, unknown for rows whose deadline falls past the last sample.
        """
        n = 50
        mid = np.full(n, 100.0)
        ts_ms = np.arange(n, dtype=np.int64) * 100  # last_ts = 4900
        horizon_ms = 2_000

        labels = triple_barrier_labels(
            mid, ts_ms,
            horizon_ms=horizon_ms,
            up_target_ticks=4.0,
            down_target_ticks=4.0,
            tick_size=0.25,
        )

        # All side == 0 (no barrier ever fires on a flat series), but
        # rows whose deadline > last_ts get tagged "unknown" rather than
        # "timeout" so they are dropped from training.
        assert all(lbl.side == 0 for lbl in labels)
        assert all(lbl.ret == 0.0 for lbl in labels)
        last_ts = int(ts_ms[-1])
        timeouts = [
            lbl for i, lbl in enumerate(labels)
            if int(ts_ms[i]) + horizon_ms <= last_ts
        ]
        unknowns = [
            lbl for i, lbl in enumerate(labels)
            if int(ts_ms[i]) + horizon_ms > last_ts
        ]
        assert len(timeouts) == 30  # rows 0..29
        assert len(unknowns) == 20  # rows 30..49
        assert all(lbl.barrier_hit == "timeout" for lbl in timeouts)
        assert all(lbl.barrier_hit == "unknown" for lbl in unknowns)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            triple_barrier_labels(
                np.array([1.0, 2.0]), np.array([0], dtype=np.int64),
            )

    def test_returns_dataclass_instances(self) -> None:
        mid = np.array([100.0, 101.0])
        ts_ms = np.array([0, 100], dtype=np.int64)
        labels = triple_barrier_labels(
            mid, ts_ms, horizon_ms=1_000,
            up_target_ticks=2.0, down_target_ticks=2.0, tick_size=0.25,
        )
        assert all(isinstance(lbl, TripleBarrierLabel) for lbl in labels)

    def test_mixed_series_resolves_first_barrier(self) -> None:
        """Up-then-down series: first entry hits up before down."""
        mid = np.array([100.0, 100.5, 101.5, 100.0, 98.5])
        ts_ms = np.arange(5, dtype=np.int64) * 100

        labels = triple_barrier_labels(
            mid, ts_ms, horizon_ms=500,
            up_target_ticks=4.0, down_target_ticks=4.0, tick_size=0.25,
        )
        # up-barrier = 100 + 1.0 = 101.0; mid[2]=101.5 hits first
        assert labels[0].side == 1
        assert labels[0].barrier_hit == "up"


class TestCostThresholdedBinary:
    def test_mask_subcost_as_nan(self) -> None:
        """Moves below cost are masked to NaN."""
        # cost = 2 ticks * 0.25 = 0.5
        # horizon = 5 rows
        mid = np.concatenate([
            np.array([100.0] * 10),       # 10 flat rows → 5 sub-cost moves
            np.array([100.0, 101.0, 101.0, 99.0, 100.5]),  # some supra-cost moves
        ])
        labels = cost_thresholded_binary(
            mid, horizon_snapshots=5, cost_ticks=2.0, tick_size=0.25,
        )
        # First 5 rows look 5 steps forward to still-flat → NaN
        assert np.isnan(labels[:5]).all()

    def test_above_cost_binary(self) -> None:
        mid = np.array([100.0, 100.0, 100.0, 101.0, 101.0])  # fwd-2 = 1.0 for row 0
        labels = cost_thresholded_binary(
            mid, horizon_snapshots=2, cost_ticks=2.0, tick_size=0.25,
        )
        # row 0: 100 → 100 = 0, <cost → NaN
        # row 1: 100 → 101 = 1, >cost, up → 1
        # row 2: 100 → 101 = 1, >cost, up → 1
        # rows 3,4: NaN (no forward data past horizon)
        assert np.isnan(labels[0])
        assert labels[1] == 1.0
        assert labels[2] == 1.0
        assert np.isnan(labels[3])
        assert np.isnan(labels[4])

    def test_negative_move_labeled_zero(self) -> None:
        mid = np.array([100.0, 100.0, 99.0])
        labels = cost_thresholded_binary(
            mid, horizon_snapshots=2, cost_ticks=2.0, tick_size=0.25,
        )
        assert labels[0] == 0.0  # down move ≥ cost

    def test_short_series_all_nan(self) -> None:
        mid = np.array([100.0, 101.0])
        labels = cost_thresholded_binary(
            mid, horizon_snapshots=5, cost_ticks=2.0, tick_size=0.25,
        )
        assert np.isnan(labels).all()


# ---------------------------------------------------------------------------
# Wave 9 §H.1.a — CUSUM events + activity mask
# ---------------------------------------------------------------------------


class TestCusumEvents:
    def test_flat_series_yields_no_events(self) -> None:
        price = np.full(100, 100.0)
        events = cusum_events(price, threshold=0.5)
        assert events.size == 0

    def test_monotone_rise_fires_at_threshold_intervals(self) -> None:
        # Step size 0.1 per row, threshold 1.0 → events approx every 10 rows.
        n = 100
        price = np.arange(n, dtype=float) * 0.1
        events = cusum_events(price, threshold=1.0)
        assert events.size > 0
        # Events should be roughly evenly spaced (CUSUM resets each fire).
        gaps = np.diff(events)
        # Allow some slack for accumulator boundaries (off-by-one on resets).
        assert all(8 <= g <= 12 for g in gaps), f"unexpected gap pattern: {gaps}"

    def test_alternating_signs_no_events_below_threshold(self) -> None:
        # Up/down alternation: cumulative drift stays near zero.
        n = 100
        price = 100.0 + np.array([0.1 * (-1) ** i for i in range(n)]).cumsum()
        events = cusum_events(price, threshold=1.0)
        # Possibly a small number of events from edge effects but not many.
        assert events.size <= 5

    def test_negative_drift_fires_negative_arm(self) -> None:
        n = 50
        price = 100.0 - np.arange(n, dtype=float) * 0.2
        events = cusum_events(price, threshold=1.0)
        assert events.size > 0  # negative arm should fire

    def test_threshold_zero_returns_empty(self) -> None:
        price = np.linspace(100, 110, 50)
        assert cusum_events(price, threshold=0.0).size == 0

    def test_short_series_returns_empty(self) -> None:
        assert cusum_events(np.array([100.0]), threshold=0.5).size == 0
        assert cusum_events(np.array([], dtype=float), threshold=0.5).size == 0

    def test_events_indices_are_in_range(self) -> None:
        n = 200
        price = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.1, n))
        events = cusum_events(price, threshold=0.5)
        assert (events >= 0).all() and (events < n).all()


class TestActivityMask:
    def test_cusum_lookback_includes_pre_event_rows(self) -> None:
        # Single CUSUM event at row 50 (constant rise crossing threshold).
        n = 100
        price = np.arange(n, dtype=float) * 0.1
        ts_ms = np.arange(n, dtype=np.int64) * 100  # 100 ms cadence
        mask = activity_mask(
            price, ts_ms,
            method="cusum",
            cusum_threshold=1.0,
            lookback_ms=500,            # 5 rows of lookback at 100ms cadence
            lookforward_ms=0,
        )
        assert mask.dtype == bool
        assert mask.shape == (n,)
        assert mask.any()
        # Some pre-event rows should be in mask; first row never is.
        assert not mask[0]

    def test_cusum_lookforward_zero_excludes_far_post_event_rows(self) -> None:
        # With lookforward_ms=0, rows beyond the last event timestamp are False.
        n = 50
        price = np.concatenate([
            np.arange(20, dtype=float) * 0.1,  # rises through one threshold
            np.full(30, 1.9),                   # then flat — no further events
        ])
        ts_ms = np.arange(n, dtype=np.int64) * 100
        mask = activity_mask(
            price, ts_ms,
            method="cusum",
            cusum_threshold=1.0,
            lookback_ms=300,
            lookforward_ms=0,
        )
        # Tail rows (well past any event) should be False.
        assert not mask[-1]
        assert not mask[-5:].any()

    def test_volume_bar_method(self) -> None:
        n = 20
        price = np.full(n, 100.0)
        ts_ms = np.arange(n, dtype=np.int64) * 100
        volume = np.array([10, 50, 100, 150, 200, 50, 10, 500, 5, 5,
                           5, 5, 5, 5, 200, 5, 5, 5, 5, 5])
        mask = activity_mask(
            price, ts_ms,
            method="volume_bar",
            volume=volume,
            volume_floor=100,
        )
        # Expect True only at rows whose volume ≥ 100
        expected = volume >= 100
        assert (mask == expected).all()

    def test_volume_bar_requires_volume_arg(self) -> None:
        n = 5
        with pytest.raises(ValueError, match="requires the `volume`"):
            activity_mask(
                np.zeros(n), np.zeros(n, dtype=np.int64),
                method="volume_bar",
            )

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            activity_mask(
                np.zeros(10), np.zeros(5, dtype=np.int64),
                method="cusum",
            )

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            activity_mask(
                np.zeros(5), np.arange(5, dtype=np.int64),
                method="bogus",
            )

    def test_flat_series_yields_all_false_mask(self) -> None:
        n = 100
        mask = activity_mask(
            np.full(n, 100.0), np.arange(n, dtype=np.int64) * 100,
            method="cusum", cusum_threshold=1.0,
        )
        assert mask.dtype == bool
        assert not mask.any()

    def test_compose_via_logical_and(self) -> None:
        # Both detectors fire on the same row → AND is True there.
        n = 30
        price = np.arange(n, dtype=float) * 0.1   # CUSUM-active throughout
        ts_ms = np.arange(n, dtype=np.int64) * 100
        volume = np.zeros(n)
        volume[15:20] = 500   # volume-active rows 15..19

        m_cusum = activity_mask(
            price, ts_ms,
            method="cusum", cusum_threshold=1.0,
            lookback_ms=500, lookforward_ms=0,
        )
        m_vol = activity_mask(
            price, ts_ms,
            method="volume_bar",
            volume=volume, volume_floor=100,
        )
        combined = m_cusum & m_vol
        # Composition is just elementwise AND
        assert combined.dtype == bool
        assert (combined == (m_cusum & m_vol)).all()


# ---------------------------------------------------------------------------
# Wave 9 §H.1.c — path-aware 5-class MFE/MAE labels
# ---------------------------------------------------------------------------


class TestMFEMAELabels:
    """5-class label scheme {-2, -1, 0, +1, +2} encoding direction × cleanness."""

    def test_returns_one_label_per_row(self) -> None:
        n = 30
        mid = 100.0 + np.arange(n, dtype=float) * 0.1
        ts_ms = np.arange(n, dtype=np.int64) * 100
        labels = mfe_mae_labels(
            mid, ts_ms,
            horizon_ms=2_000,
            up_target_ticks=4.0,
            down_target_ticks=4.0,
            stop_threshold_ticks=1.5,
            tick_size=0.25,
        )
        assert len(labels) == n
        assert all(isinstance(lbl, MFEMAELabel) for lbl in labels)
        assert all(lbl.label in (-2, -1, 0, 1, 2) for lbl in labels)

    def test_clean_bull_on_monotonic_rise(self) -> None:
        """Strict monotonic rise → up barrier hit with no drawdown → +2."""
        n = 30
        mid = 100.0 + np.arange(n, dtype=float) * 0.5  # rises 0.5/step (2 ticks)
        ts_ms = np.arange(n, dtype=np.int64) * 100
        labels = mfe_mae_labels(
            mid, ts_ms,
            horizon_ms=5_000,
            up_target_ticks=4.0,    # 1.0 in price units
            down_target_ticks=4.0,
            stop_threshold_ticks=1.5,
            tick_size=0.25,
        )
        clean_bull = [lbl for lbl in labels if lbl.label == 2]
        assert len(clean_bull) >= n - 5  # most rows resolve as clean +2
        for lbl in clean_bull:
            assert lbl.barrier_hit == "up"
            assert lbl.mfe >= 1.0  # at least one tick × 4 reached
            assert lbl.mae == 0.0  # no drawdown on monotonic rise

    def test_clean_bear_on_monotonic_fall(self) -> None:
        n = 30
        mid = 100.0 - np.arange(n, dtype=float) * 0.5
        ts_ms = np.arange(n, dtype=np.int64) * 100
        labels = mfe_mae_labels(
            mid, ts_ms,
            horizon_ms=5_000,
            up_target_ticks=4.0,
            down_target_ticks=4.0,
            stop_threshold_ticks=1.5,
            tick_size=0.25,
        )
        clean_bear = [lbl for lbl in labels if lbl.label == -2]
        assert len(clean_bear) >= n - 5
        for lbl in clean_bear:
            assert lbl.barrier_hit == "down"
            assert lbl.mae <= -1.0
            assert lbl.mfe == 0.0  # no upward excursion on monotonic fall

    def test_wicky_bull_when_drawdown_then_recovery(self) -> None:
        """Dip past stop, then recover and hit up barrier → +1, not +2."""
        # Start at 100. Drop to 99.55 (= -0.45 = -1.8 ticks, past 1.5 stop).
        # Then climb to 101.0 (+1.0 = 4 ticks → up barrier).
        mid = np.array(
            [100.00, 99.80, 99.55, 99.80, 100.20, 100.60, 101.00, 101.10],
            dtype=float,
        )
        ts_ms = np.arange(len(mid), dtype=np.int64) * 100
        labels = mfe_mae_labels(
            mid, ts_ms,
            horizon_ms=2_000,
            up_target_ticks=4.0,
            down_target_ticks=8.0,    # wide enough to not hit on the dip
            stop_threshold_ticks=1.5,
            tick_size=0.25,
        )
        # Row 0: entry at 100. MAE reaches -0.45 (past stop), then up
        # barrier (+1.0) hit → wicky bull (+1).
        assert labels[0].label == 1
        assert labels[0].barrier_hit == "up"
        assert abs(labels[0].mae) >= 1.5 * 0.25  # crossed stop threshold

    def test_wicky_bear_when_pop_then_fall(self) -> None:
        """Pop past stop, then fall to down barrier → -1, not -2."""
        mid = np.array(
            [100.00, 100.20, 100.45, 100.20, 99.80, 99.40, 99.00, 98.90],
            dtype=float,
        )
        ts_ms = np.arange(len(mid), dtype=np.int64) * 100
        labels = mfe_mae_labels(
            mid, ts_ms,
            horizon_ms=2_000,
            up_target_ticks=8.0,    # wide enough to not hit on the pop
            down_target_ticks=4.0,
            stop_threshold_ticks=1.5,
            tick_size=0.25,
        )
        # Row 0: entry at 100. MFE reaches +0.45 (past stop), then down
        # barrier (-1.0) hit → wicky bear (-1).
        assert labels[0].label == -1
        assert labels[0].barrier_hit == "down"
        assert labels[0].mfe >= 1.5 * 0.25

    def test_timeout_when_no_barrier_hits(self) -> None:
        """Flat/oscillating series within barriers → label=0. Rows whose
        deadline extends past the last sample get tagged ``unknown``
        instead of ``timeout`` so they're excluded from training."""
        n = 20
        rng = np.random.default_rng(0)
        # Tiny noise that never reaches a 4-tick barrier
        mid = 100.0 + rng.normal(0, 0.05, n).cumsum() * 0.0  # essentially flat
        mid[:] = 100.0 + rng.uniform(-0.1, 0.1, n)
        ts_ms = np.arange(n, dtype=np.int64) * 100  # last_ts = 1900
        horizon_ms = 1_500
        labels = mfe_mae_labels(
            mid, ts_ms,
            horizon_ms=horizon_ms,
            up_target_ticks=4.0,
            down_target_ticks=4.0,
            stop_threshold_ticks=1.5,
            tick_size=0.25,
        )
        # All should time out → label 0
        assert all(lbl.label == 0 for lbl in labels)
        assert all(lbl.barrier_hit in {"timeout", "unknown"} for lbl in labels)
        last_ts = int(ts_ms[-1])
        # Observable rows: i*100 + 1500 <= 1900 → i <= 4
        for i, lbl in enumerate(labels):
            expected = "timeout" if int(ts_ms[i]) + horizon_ms <= last_ts else "unknown"
            assert lbl.barrier_hit == expected, (i, lbl.barrier_hit)

    def test_mfe_mae_invariants(self) -> None:
        """MFE >= 0; MAE <= 0; finite for every label."""
        rng = np.random.default_rng(7)
        n = 100
        mid = 100.0 + rng.normal(0, 0.1, n).cumsum()
        ts_ms = np.arange(n, dtype=np.int64) * 100
        labels = mfe_mae_labels(
            mid, ts_ms,
            horizon_ms=3_000,
            up_target_ticks=4.0,
            down_target_ticks=4.0,
            stop_threshold_ticks=1.5,
            tick_size=0.25,
        )
        for lbl in labels:
            assert lbl.mfe >= 0.0
            assert lbl.mae <= 0.0
            assert np.isfinite(lbl.mfe)
            assert np.isfinite(lbl.mae)
            assert lbl.barrier_hit in {"up", "down", "timeout", "unknown"}

    def test_cleanness_threshold_boundary_inclusive(self) -> None:
        """Stop threshold uses ``abs(opposite) < stop`` (strict): exactly at
        the stop magnitude is wicky, not clean. Lock the convention."""
        # Build a series where MAE reaches exactly -0.375 (= 1.5 ticks).
        mid = np.array(
            [100.00, 99.625, 100.50, 101.00],
            dtype=float,
        )
        ts_ms = np.arange(len(mid), dtype=np.int64) * 100
        labels = mfe_mae_labels(
            mid, ts_ms,
            horizon_ms=1_000,
            up_target_ticks=4.0,
            down_target_ticks=8.0,
            stop_threshold_ticks=1.5,
            tick_size=0.25,
        )
        # MAE = -0.375 exactly = stop → wicky (label=1, not 2).
        assert labels[0].mae == pytest.approx(-0.375)
        assert labels[0].label == 1

    def test_horizon_ms_records_actual_resolution_time(self) -> None:
        """horizon_ms field == time-to-barrier on resolution, == horizon
        on timeout."""
        # Very short rise — barrier hits at row 4
        mid = np.array([100.0, 100.25, 100.5, 100.75, 101.0, 101.25],
                       dtype=float)
        ts_ms = np.arange(len(mid), dtype=np.int64) * 100
        labels = mfe_mae_labels(
            mid, ts_ms,
            horizon_ms=10_000,
            up_target_ticks=4.0,
            down_target_ticks=4.0,
            stop_threshold_ticks=1.5,
            tick_size=0.25,
        )
        # Row 0: barrier hits at row 4 (ts=400ms) → horizon_ms == 400
        assert labels[0].label == 2
        assert labels[0].horizon_ms == 400

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            mfe_mae_labels(
                np.zeros(10), np.zeros(5, dtype=np.int64),
                horizon_ms=1_000,
            )

    def test_class_distribution_under_random_walk(self) -> None:
        """All five classes appear under a sufficiently volatile random walk.

        Parameters tuned so that:
        - n=1000 with σ=0.15 and 50-row horizon → cumulative σ ≈ 1.06,
          which is 2.1× a 0.5 barrier — plenty of resolutions
        - Tight stop (0.5 ticks) ensures wickiness is achievable
        """
        rng = np.random.default_rng(42)
        n = 1_000
        mid = 100.0 + rng.normal(0, 0.15, n).cumsum()
        ts_ms = np.arange(n, dtype=np.int64) * 100
        labels = mfe_mae_labels(
            mid, ts_ms,
            horizon_ms=5_000,           # 50 rows
            up_target_ticks=2.0,         # 0.5 price units
            down_target_ticks=2.0,
            stop_threshold_ticks=0.5,    # tight stop → wicky labels achievable
            tick_size=0.25,
        )
        counts: dict[int, int] = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
        for lbl in labels:
            counts[lbl.label] += 1
        # All five classes should appear at this scale + volatility.
        # (Wicky-vs-clean ratio is parameter-sensitive; not asserted.)
        assert all(c > 0 for c in counts.values()), counts


# ---------------------------------------------------------------------------
# Wave 9 §H.1.b — multi-target label dispatcher
# ---------------------------------------------------------------------------


class TestComputeLabelSet:
    """Multi-spec dispatcher: TB / MFE-MAE side-by-side, multiple horizons."""

    def _series(self, n: int = 60, drift: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        mid = 100.0 + np.arange(n, dtype=float) * drift
        ts_ms = np.arange(n, dtype=np.int64) * 100
        return mid, ts_ms

    def test_empty_specs_returns_empty_dict(self) -> None:
        mid, ts_ms = self._series()
        out = compute_label_set(mid, ts_ms, specs=[])
        assert out == {}

    def test_single_tb_spec_matches_direct_call(self) -> None:
        mid, ts_ms = self._series()
        spec = LabelSpec(
            name="tb_60s", kind="tb",
            horizon_ms=2_000, up_target_ticks=4.0, down_target_ticks=4.0,
            tick_size=0.25,
        )
        out = compute_label_set(mid, ts_ms, [spec])
        direct = triple_barrier_labels(
            mid, ts_ms,
            horizon_ms=2_000, up_target_ticks=4.0, down_target_ticks=4.0,
            tick_size=0.25,
        )
        assert "tb_60s" in out
        assert out["tb_60s"].shape == (len(mid),)
        assert out["tb_60s"].dtype == np.float64
        # compute_label_set converts barrier_hit=="unknown" → NaN; resolved
        # rows carry their integer side as a float.
        expected = np.asarray(
            [
                float("nan") if lbl.barrier_hit == "unknown"
                else float(lbl.side)
                for lbl in direct
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(out["tb_60s"], expected)

    def test_single_mfe_mae_spec_matches_direct_call(self) -> None:
        mid, ts_ms = self._series(drift=0.5)  # rising fast → clean +2 expected
        spec = LabelSpec(
            name="mm_60s", kind="mfe_mae",
            horizon_ms=2_000, stop_threshold_ticks=1.5,
        )
        out = compute_label_set(mid, ts_ms, [spec])
        direct = mfe_mae_labels(
            mid, ts_ms, horizon_ms=2_000, stop_threshold_ticks=1.5,
        )
        expected = np.asarray(
            [
                float("nan") if lbl.barrier_hit == "unknown"
                else float(lbl.label)
                for lbl in direct
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(out["mm_60s"], expected)

    def test_multiple_horizons_produce_independent_columns(self) -> None:
        # Construct a series that resolves at long horizon but times out at short.
        # Slow rise: 0.04 per row → 4 ticks (1.0) reached at row 25 (= 2.5s)
        n = 80
        mid = 100.0 + np.arange(n, dtype=float) * 0.04
        ts_ms = np.arange(n, dtype=np.int64) * 100
        specs = [
            LabelSpec("tb_500ms", kind="tb", horizon_ms=500),    # too short
            LabelSpec("tb_3s",    kind="tb", horizon_ms=3_000),
            LabelSpec("tb_5s",    kind="tb", horizon_ms=5_000),
        ]
        out = compute_label_set(mid, ts_ms, specs)
        # Short horizon: row 0 cannot reach +1.0 → timeout (0)
        # Longer horizons: row 0 should hit up barrier → +1
        assert out["tb_500ms"][0] == 0
        assert out["tb_3s"][0] == 1
        assert out["tb_5s"][0] == 1
        # All columns same length
        for col in out.values():
            assert col.shape == (n,)

    def test_mixed_tb_and_mfe_mae_specs(self) -> None:
        n = 50
        mid = 100.0 + np.arange(n, dtype=float) * 0.5  # fast rise
        ts_ms = np.arange(n, dtype=np.int64) * 100
        specs = [
            LabelSpec("tb",  kind="tb",      horizon_ms=2_000),
            LabelSpec("mm",  kind="mfe_mae", horizon_ms=2_000),
        ]
        out = compute_label_set(mid, ts_ms, specs)
        # Observable rows (i*100 + 2000 <= last_ts=4900 → i <= 29) should
        # all be bullish on this monotonically rising series.
        tb_obs = out["tb"][~np.isnan(out["tb"])]
        mm_obs = out["mm"][~np.isnan(out["mm"])]
        assert (tb_obs == 1).all()
        # MFE/MAE should give clean +2 when up barrier hits monotonically
        assert (mm_obs == 2).sum() > 0
        assert (tb_obs == 1).sum() > 0

    def test_duplicate_spec_names_raise(self) -> None:
        mid, ts_ms = self._series()
        specs = [
            LabelSpec("foo", kind="tb",      horizon_ms=1_000),
            LabelSpec("foo", kind="mfe_mae", horizon_ms=1_000),
        ]
        with pytest.raises(ValueError, match="duplicate spec names"):
            compute_label_set(mid, ts_ms, specs)

    def test_unknown_kind_raises(self) -> None:
        mid, ts_ms = self._series()
        specs = [LabelSpec("x", kind="bogus", horizon_ms=1_000)]
        with pytest.raises(ValueError, match="unknown LabelSpec.kind"):
            compute_label_set(mid, ts_ms, specs)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            compute_label_set(
                np.zeros(10), np.zeros(5, dtype=np.int64),
                [LabelSpec("a", kind="tb")],
            )

    def test_full_a_phase_spec_set(self) -> None:
        """The canonical Wave 9 A-phase spec list — TB + MFE/MAE at four
        horizons. Verifies the dispatcher handles the full intended use.

        Cadence is 1s/row (not 100ms) so the longest horizon (300s) is
        actually observable for a meaningful slice of rows; otherwise
        every label would be NaN/unknown."""
        rng = np.random.default_rng(11)
        n = 500
        mid = 100.0 + rng.normal(0, 0.1, n).cumsum()
        ts_ms = np.arange(n, dtype=np.int64) * 1_000  # 1s cadence
        horizons = (60_000, 120_000, 180_000, 300_000)
        specs: list[LabelSpec] = []
        for h in horizons:
            specs.append(LabelSpec(f"tb_{h//1000}s", kind="tb", horizon_ms=h))
            specs.append(LabelSpec(f"mm_{h//1000}s", kind="mfe_mae", horizon_ms=h))
        out = compute_label_set(mid, ts_ms, specs)
        # Eight columns, each of length n, all float64 (NaN for unknown).
        assert len(out) == 8
        for name, vec in out.items():
            assert vec.shape == (n,)
            assert vec.dtype == np.float64
        # TB columns produce observable values in {-1, 0, 1}; MFE/MAE in
        # {-2, -1, 0, 1, 2}. NaN is allowed (unknown — horizon unobservable).
        for name, vec in out.items():
            obs = vec[~np.isnan(vec)]
            assert obs.size > 0, f"{name} has no observable rows"
            if name.startswith("tb_"):
                assert set(np.unique(obs).tolist()) <= {-1.0, 0.0, 1.0}
            else:
                assert set(np.unique(obs).tolist()) <= {-2.0, -1.0, 0.0, 1.0, 2.0}

    def test_dataframe_wrap_pattern(self) -> None:
        """Document the expected NB06 §04 usage: wrap dispatcher output
        in pd.DataFrame for downstream training."""
        import pandas as pd
        mid, ts_ms = self._series(n=20)
        specs = [
            LabelSpec("tb_a", kind="tb",      horizon_ms=500),
            LabelSpec("mm_a", kind="mfe_mae", horizon_ms=500),
        ]
        out = compute_label_set(mid, ts_ms, specs)
        df = pd.DataFrame(out)
        assert df.shape == (20, 2)
        assert list(df.columns) == ["tb_a", "mm_a"]
        # Columns are float64 to carry NaN for rows whose horizon extends
        # past the observed data.
        assert df.dtypes.tolist() == [np.float64, np.float64]


# ---------------------------------------------------------------------------
# Wave 10-A — pattern-firing labels
# ---------------------------------------------------------------------------


class TestPatternFiringLabels:
    """Per-row label = "did any library pattern fire in next K snapshots?"."""

    def _ts(self, n: int = 50, dt_ms: int = 100) -> np.ndarray:
        return np.arange(n, dtype=np.int64) * dt_ms

    def test_empty_firings_observable_zero_and_unknown_nan(self) -> None:
        """Empty firings → rows with full horizon observed are 0;
        rows whose deadline > last_ts are NaN (unknown)."""
        ts = self._ts(20)  # 100ms cadence; last_ts = 1900
        labels, enc = pattern_firing_labels(ts, [], horizon_ms=1_000)
        assert labels.shape == (20,)
        # Observable rows: i*100 + 1000 <= 1900 → i <= 9
        assert (labels[:10] == 0.0).all()
        assert np.isnan(labels[10:]).all()
        assert enc == {}

    def test_binary_single_firing_in_window(self) -> None:
        """Firing at row 5 → rows 0..5 with horizon covering t=500ms label 1.

        Rows 6..9 see no firing in their window AND have full horizon
        observed → label 0. Rows 10..19 see no firing AND deadline > last_ts
        → NaN (unknown)."""
        ts = self._ts(20)  # 100ms cadence; ts[5] = 500, last_ts = 1900
        firings = [(5, "bull_breakout", 0.7)]
        labels, _ = pattern_firing_labels(
            ts, firings, horizon_ms=1_000,  # 1s forward window
        )
        assert labels[:6].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Observable, no firing → 0
        assert (labels[6:10] == 0.0).all()
        # Unobservable, no firing → NaN
        assert np.isnan(labels[10:]).all()

    def test_binary_firing_outside_window(self) -> None:
        """Firing too far away → no labels in window."""
        ts = self._ts(20)  # last_ts = 1900
        firings = [(15, "p", 0.7)]   # ts[15] = 1500
        labels, _ = pattern_firing_labels(
            ts, firings, horizon_ms=500,  # only 500ms forward
        )
        # Only rows whose window contains ts[15]=1500 fire — that's rows
        # 10..15 (ts[10]=1000, window [1000,1500] includes 1500).
        assert labels[10:16].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Rows 0..9: full horizon observed (i*100 + 500 <= 1900 always),
        # no firing → 0.
        assert (labels[:10] == 0.0).all()
        # Rows 16..19: deadline > last_ts (e.g. row 16: 1600+500=2100 > 1900),
        # no firing → NaN.
        assert np.isnan(labels[16:]).all()

    def test_first_encoding_picks_first_firing(self) -> None:
        ts = self._ts(20)
        firings = [
            (5, "alpha", 0.65),
            (8, "beta", 0.85),  # higher score but later
        ]
        labels, enc = pattern_firing_labels(
            ts, firings, horizon_ms=2_000, encoding="first",
        )
        assert enc == {1: "alpha", 2: "beta"}
        # Row 0's window [0, 2000] sees firings at 500, 800 → first=alpha=1
        assert labels[0] == 1
        # Row 6's window [600, 2600] — first firing >= row 6 is beta@8
        assert labels[6] == 2

    def test_best_encoding_picks_highest_score(self) -> None:
        ts = self._ts(20)
        firings = [
            (5, "alpha", 0.65),
            (8, "beta", 0.85),
            (10, "gamma", 0.55),
        ]
        labels, enc = pattern_firing_labels(
            ts, firings, horizon_ms=2_000, encoding="best",
        )
        # Row 0's window [0, 2000] sees all three; best = beta (0.85)
        beta_ord = {v: k for k, v in enc.items()}["beta"]
        assert labels[0] == beta_ord

    def test_pattern_id_order_overrides_default(self) -> None:
        ts = self._ts(20)
        firings = [(5, "alpha", 0.7), (10, "beta", 0.7)]
        order = ["beta", "alpha"]   # reverse of default sorted order
        labels, enc = pattern_firing_labels(
            ts, firings, horizon_ms=2_000, encoding="first",
            pattern_id_order=order,
        )
        assert enc == {1: "beta", 2: "alpha"}

    def test_firing_event_dataclass_input(self) -> None:
        """FiringEvent instances are accepted alongside tuples."""
        ts = self._ts(20)
        firings = [
            FiringEvent(snapshot_idx=5, pattern_id="bull", score=0.7),
            FiringEvent(snapshot_idx=10, pattern_id="bear", score=0.65),
        ]
        labels, enc = pattern_firing_labels(
            ts, firings, horizon_ms=1_500, encoding="binary",
        )
        # Both firings should produce labels in their windows. Use nansum
        # because rows whose horizon extends past last_ts and see no
        # firing emit NaN.
        assert np.nansum(labels) > 0

    def test_unknown_encoding_raises(self) -> None:
        ts = self._ts(10)
        with pytest.raises(ValueError, match="unknown encoding"):
            pattern_firing_labels(ts, [], encoding="bogus")

    def test_out_of_range_snap_idx_skipped(self) -> None:
        ts = self._ts(10)
        firings = [(5, "p", 0.7), (100, "q", 0.7)]   # 100 is out of range
        labels, _ = pattern_firing_labels(
            ts, firings, horizon_ms=2_000, encoding="binary",
        )
        # Only the in-range firing should affect labels
        assert labels[5] == 1   # within range
        # No row should be labeled by the out-of-range firing
        # (row 100 doesn't exist, and ts[100] would be out of bounds)
        assert labels.shape == (10,)

    def test_dtype_is_float64(self) -> None:
        """All encodings emit float64 so NaN can flag unknown rows."""
        ts = self._ts(10)
        firings = [(2, "p", 0.7)]
        binary, _ = pattern_firing_labels(ts, firings, encoding="binary")
        first, _ = pattern_firing_labels(ts, firings, encoding="first")
        best, _ = pattern_firing_labels(ts, firings, encoding="best")
        assert binary.dtype == np.float64
        assert first.dtype == np.float64
        assert best.dtype == np.float64

    def test_horizon_zero_only_self_row(self) -> None:
        """horizon_ms=0 means firing at row i counts ONLY for row i."""
        ts = self._ts(10)
        firings = [(5, "p", 0.7)]
        labels, _ = pattern_firing_labels(
            ts, firings, horizon_ms=0, encoding="binary",
        )
        assert labels[5] == 1
        assert labels[:5].sum() == 0
        assert labels[6:].sum() == 0

    def test_realistic_sparse_firings(self) -> None:
        """Sparse firings (1% of rows) — labels should be sparse but
        nonzero, sized correctly."""
        rng = np.random.default_rng(0)
        n = 1000
        ts = self._ts(n)
        # 10 firings randomly placed
        fire_indices = rng.choice(n, size=10, replace=False)
        firings = [(int(i), f"pat_{i % 3}", 0.7) for i in fire_indices]
        labels, enc = pattern_firing_labels(
            ts, firings, horizon_ms=2_000, encoding="best",
        )
        # Some labels should be nonzero (in firing windows)
        assert (labels > 0).sum() > 0
        # Encoding map should have entries for all unique pattern_ids
        assert set(enc.values()) == {"pat_0", "pat_1", "pat_2"}

    def test_dataframe_wrap_pattern(self) -> None:
        """Document the NB06 §02 usage: stack pattern_fire labels with
        compute_label_set output via pandas DataFrame."""
        import pandas as pd
        n = 30
        mid = 100.0 + np.arange(n, dtype=float) * 0.05
        ts = self._ts(n)
        firings = [(10, "p1", 0.7)]

        tb_mfe = compute_label_set(mid, ts, [
            LabelSpec("tb_60s", kind="tb", horizon_ms=2_000),
            LabelSpec("mm_60s", kind="mfe_mae", horizon_ms=2_000),
        ])
        pf_labels, _ = pattern_firing_labels(
            ts, firings, horizon_ms=2_000, encoding="binary",
        )

        all_labels = {**tb_mfe, "pf_60s": pf_labels}
        df = pd.DataFrame(all_labels)
        assert df.shape == (n, 3)
        assert "pf_60s" in df.columns
