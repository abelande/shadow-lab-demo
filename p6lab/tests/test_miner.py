"""Tests for p6lab.patterns.miner — HDBSCAN pattern discovery."""
from __future__ import annotations

import numpy as np
import pytest

from p6lab.patterns.miner import (
    MIN_COSINE_DISTANCE_TO_EXISTING,
    MIN_HIT_RATE_5M,
    MIN_OCCURRENCES,
    MIN_SHARPE,
    MinedCandidate,
    SHAPE_VECTOR_DIM,
    extract_event_shape_vector,
    filter_candidates,
    run_hdbscan,
)


def _ev(ts: int, price: float = 100.0, size: float = 1.0,
        side: str = "bid", action: str = "add") -> dict:
    return {"timestamp_ms": ts, "price": price, "size": size,
            "side": side, "action": action, "order_id": f"o{ts}"}


class TestShapeVector:
    def test_returns_correct_dim(self):
        evs = [_ev(0), _ev(100), _ev(200)]
        v = extract_event_shape_vector(evs)
        assert v.shape == (SHAPE_VECTOR_DIM,)

    def test_empty_window_returns_zeros(self):
        v = extract_event_shape_vector([])
        assert v.shape == (SHAPE_VECTOR_DIM,)
        assert np.all(v == 0)

    def test_add_cancel_ratio(self):
        evs = [_ev(i, action="add") for i in range(10)]
        evs += [_ev(i, action="cancel") for i in range(10, 15)]
        v = extract_event_shape_vector(evs)
        # 10 adds / (10+5) = 0.666...
        assert v[0] == pytest.approx(10 / 15, abs=0.001)

    def test_side_asymmetry(self):
        evs = [_ev(i, side="bid") for i in range(8)]
        evs += [_ev(i, side="ask") for i in range(8, 10)]
        v = extract_event_shape_vector(evs)
        # (8-2)/10 = 0.6
        assert v[13] == pytest.approx(0.6, abs=0.01)

    def test_reserved_dims_are_zero(self):
        evs = [_ev(i, price=100 + i * 0.1, size=float(i)) for i in range(50)]
        v = extract_event_shape_vector(evs)
        assert np.all(v[15:30] == 0)

    def test_burst_intensity_scales_with_density(self):
        # 10 events in 100ms → 100 ev/s
        dense = [_ev(i * 10) for i in range(10)]
        # 10 events in 1000ms → 10 ev/s
        sparse = [_ev(i * 100) for i in range(10)]
        vd = extract_event_shape_vector(dense)
        vs = extract_event_shape_vector(sparse)
        assert vd[1] > vs[1]


class TestHDBSCAN:
    def test_clusters_distinct_groups(self):
        rng = np.random.default_rng(42)
        # Three well-separated groups in the first 2 dims
        g1 = rng.normal(loc=[0, 0], scale=0.1, size=(60, 2))
        g2 = rng.normal(loc=[10, 10], scale=0.1, size=(60, 2))
        g3 = rng.normal(loc=[-10, 10], scale=0.1, size=(60, 2))
        data = np.vstack([g1, g2, g3])
        # pad to SHAPE_VECTOR_DIM
        pad = np.zeros((len(data), SHAPE_VECTOR_DIM - 2))
        data = np.hstack([data, pad])
        labels, _ = run_hdbscan(data, min_cluster_size=20, min_samples=10)
        n_clusters = len(set(l for l in labels if l >= 0))
        assert n_clusters >= 2  # HDBSCAN should find the distinct groups

    def test_empty_input(self):
        labels, clusterer = run_hdbscan(np.zeros((0, SHAPE_VECTOR_DIM)))
        assert len(labels) == 0


class TestFilterCandidates:
    def _cand(self, n=500, hit=0.70, sharpe=0.5, cos=1.0,
              centroid=None) -> MinedCandidate:
        return MinedCandidate(
            cluster_id=0,
            centroid=centroid if centroid is not None else np.ones(SHAPE_VECTOR_DIM),
            member_count=n,
            exemplar_timestamps_ms=[0, 1, 2],
            outcome_stats={},
            hit_rate_5m=hit,
            sharpe=sharpe,
            cosine_distance_to_nearest_existing=cos,
            random_state=42,
        )

    def test_passes_all_filters(self):
        kept = filter_candidates([self._cand()])
        assert len(kept) == 1

    def test_rejects_low_n(self):
        kept = filter_candidates([self._cand(n=MIN_OCCURRENCES - 1)])
        assert kept == []

    def test_rejects_low_hit_rate(self):
        kept = filter_candidates([self._cand(hit=MIN_HIT_RATE_5M - 0.01)])
        assert kept == []

    def test_rejects_low_sharpe(self):
        kept = filter_candidates([self._cand(sharpe=MIN_SHARPE - 0.01)])
        assert kept == []

    def test_rejects_too_close_to_existing(self):
        centroid = np.ones(SHAPE_VECTOR_DIM)
        existing = np.array([np.ones(SHAPE_VECTOR_DIM)])  # identical → cos dist = 0
        kept = filter_candidates([self._cand(centroid=centroid)], existing_centroids=existing)
        assert kept == []

    def test_accepts_novel_centroid(self):
        centroid = np.zeros(SHAPE_VECTOR_DIM)
        centroid[0] = 1.0
        existing = np.zeros((1, SHAPE_VECTOR_DIM))
        existing[0, 1] = 1.0  # orthogonal → cos dist = 1.0
        kept = filter_candidates([self._cand(centroid=centroid)], existing_centroids=existing)
        assert len(kept) == 1
