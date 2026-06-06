"""Wave 9-A: tests for primary-model wiring through CorrelationEngine.

These tests verify the structural fix described in
``reports/P6LAB-WAVE-9-10-BUILD-PHASES.md`` §9-A: the engine extracts a
LightGBM-like model from ``model_dict["lightgbm_model"]`` in
``reload_model``, calls ``predict_proba`` per snapshot, and stamps
``primary_proba`` onto each ``MatchResult`` and ``PatternMatch``.

We use a stub model (no LightGBM dependency in the test) — the only
contract is ``predict_proba(X) -> np.ndarray of shape (n, 2)``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p6lab.correlation.engine import CorrelationEngine
from p6lab.correlation.scorer import EnsembleScorer
from p6lab.patterns.library import (
    OutcomeDistribution, PatternDefinition, PatternLibrary, PatternStatus,
)
from p6lab.patterns.template_matcher import (
    BOOK_SHAPE_DIM, MatchContext, PatternTemplate, TemplateMatcher,
)


# ---------------------------------------------------------------------------
# Stub model — mimics LightGBM/sklearn predict_proba contract
# ---------------------------------------------------------------------------


class _StubBinaryClassifier:
    """Tiny stand-in that emits proba = sigmoid(w · x)."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = X @ self.weights
        p1 = 1.0 / (1.0 + np.exp(-z))
        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ctx(instr: str = "NQ") -> MatchContext:
    return MatchContext(
        time_of_day_minutes=600, vix_level=18.0,
        vix_regime="normal", relative_volume=1.0, instrument=instr,
    )


def _l2_window(n: int = 10, bsv_value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "book_shape_vector": [
                np.ones(BOOK_SHAPE_DIM) * bsv_value for _ in range(n)
            ],
            "bid_ask_imbalance": np.zeros(n),
        },
        index=[i * 100 for i in range(n)],
    )


def _build_engine(tmp_path: Path) -> CorrelationEngine:
    lib = PatternLibrary(tmp_path / "library.yaml")
    lib.load()
    p = PatternDefinition(
        name="bull_breakout",
        l3_signature="burst_add",
        l2_manifestation="depth_lift",
        l1_footprint="spread_collapse",
        instruments=["NQ"],
        regime_specific=False,
        status=PatternStatus.ACTIVE,
        outcome_distribution={
            "5m": OutcomeDistribution(
                mean_atr=0.5, std=0.3, hit_rate=0.7, n=300,
            ),
        },
    )
    lib.add_pattern("bull_breakout", p)
    matcher = TemplateMatcher()
    matcher.templates["bull_breakout"] = PatternTemplate(
        pattern_id="bull_breakout",
        book_series=np.ones((10, BOOK_SHAPE_DIM)),
        feature_centroid=np.ones(12),
        pattern_context={"vix_regime": "normal"},
    )
    return CorrelationEngine(
        library=lib, matcher=matcher, scorer=EnsembleScorer(),
    )


def _write_model_pickle(
    path: Path,
    *,
    feature_names: list[str],
    weights: np.ndarray | None,
    include_lightgbm: bool = True,
) -> Path:
    """Write a pickle in the schema produced by NB06 §12 (Wave 9-A)."""
    model_dict: dict = {
        "version": "test_v1",
        "matcher_templates": {"bull_breakout": np.ones((10, BOOK_SHAPE_DIM))},
        "matcher_centroids": {"bull_breakout": np.ones(12)},
        "pattern_contexts": {"bull_breakout": {"vix_regime": "normal"}},
    }
    if include_lightgbm:
        clf = _StubBinaryClassifier(
            np.zeros(len(feature_names)) if weights is None else weights,
        )
        model_dict["lightgbm_model"] = pickle.dumps(clf)
        model_dict["feature_names"] = list(feature_names)
    with open(path, "wb") as f:
        pickle.dump(model_dict, f)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReloadModelExtractsPrimary:
    def test_extracts_lightgbm_when_present(self, tmp_path: Path):
        eng = _build_engine(tmp_path)
        feat_names = ["f0", "f1", "f2"]
        path = _write_model_pickle(
            tmp_path / "m.pkl",
            feature_names=feat_names,
            weights=np.array([1.0, -1.0, 0.5]),
        )
        eng.reload_model(str(path))
        assert eng._primary_model is not None
        assert eng._primary_feature_names == feat_names

    def test_legacy_pickle_without_lightgbm_is_warned_not_fatal(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        eng = _build_engine(tmp_path)
        path = _write_model_pickle(
            tmp_path / "legacy.pkl",
            feature_names=[],
            weights=None,
            include_lightgbm=False,
        )
        with caplog.at_level("WARNING"):
            eng.reload_model(str(path))
        assert eng._primary_model is None
        assert eng._primary_feature_names == []
        assert any(
            "no 'lightgbm_model'" in rec.message for rec in caplog.records
        )


class TestPrimaryProbaFlowsThroughMatch:
    def test_proba_stamped_on_pattern_match_with_dict_features(
        self, tmp_path: Path,
    ):
        eng = _build_engine(tmp_path)
        feat_names = ["f0", "f1", "f2"]
        # Weights chosen so different inputs give different probas.
        _write_model_pickle(
            tmp_path / "m.pkl",
            feature_names=feat_names,
            weights=np.array([2.0, -1.0, 0.5]),
        )
        eng.reload_model(str(tmp_path / "m.pkl"))

        feature_row = {"f0": 1.0, "f1": 0.5, "f2": -0.25}
        matches = eng.match(_l2_window(), None, _ctx(), feature_row=feature_row)
        assert len(matches) >= 1
        for m in matches:
            assert m.primary_proba is not None
            assert 0.0 <= m.primary_proba <= 1.0

    def test_proba_stamped_on_match_result_with_array_features(
        self, tmp_path: Path,
    ):
        eng = _build_engine(tmp_path)
        feat_names = ["a", "b"]
        _write_model_pickle(
            tmp_path / "m.pkl",
            feature_names=feat_names,
            weights=np.array([1.0, 1.0]),
        )
        eng.reload_model(str(tmp_path / "m.pkl"))
        # Pre-ordered ndarray (caller responsible for column order).
        row = np.array([0.7, -0.3])
        matches = eng.match(_l2_window(), None, _ctx(), feature_row=row)
        assert all(m.primary_proba is not None for m in matches)

    def test_proba_varies_across_feature_rows(self, tmp_path: Path):
        """Std > 0 across distinct feature rows — proves model is *running*,
        not stuck on a constant. Wave 9-A done-criterion proxy."""
        eng = _build_engine(tmp_path)
        feat_names = ["f0", "f1"]
        _write_model_pickle(
            tmp_path / "m.pkl",
            feature_names=feat_names,
            weights=np.array([3.0, -3.0]),
        )
        eng.reload_model(str(tmp_path / "m.pkl"))

        seen = []
        for x in np.linspace(-1.0, 1.0, 11):
            matches = eng.match(
                _l2_window(), None, _ctx(),
                feature_row={"f0": float(x), "f1": -float(x)},
            )
            if matches:
                seen.append(matches[0].primary_proba)
        assert len(seen) >= 5
        assert float(np.std(seen)) > 0.10

    def test_no_proba_when_feature_row_omitted(self, tmp_path: Path):
        eng = _build_engine(tmp_path)
        _write_model_pickle(
            tmp_path / "m.pkl",
            feature_names=["f0"],
            weights=np.array([1.0]),
        )
        eng.reload_model(str(tmp_path / "m.pkl"))
        matches = eng.match(_l2_window(), None, _ctx())
        assert all(m.primary_proba is None for m in matches)

    def test_no_proba_when_dict_missing_required_column(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        eng = _build_engine(tmp_path)
        _write_model_pickle(
            tmp_path / "m.pkl",
            feature_names=["needed_a", "needed_b"],
            weights=np.array([1.0, 1.0]),
        )
        eng.reload_model(str(tmp_path / "m.pkl"))
        with caplog.at_level("WARNING"):
            matches = eng.match(
                _l2_window(), None, _ctx(),
                feature_row={"needed_a": 0.5},  # needed_b missing
            )
        assert all(m.primary_proba is None for m in matches)
        assert any("missing column" in rec.message for rec in caplog.records)

    def test_no_proba_when_array_width_wrong(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        eng = _build_engine(tmp_path)
        _write_model_pickle(
            tmp_path / "m.pkl",
            feature_names=["f0", "f1", "f2"],
            weights=np.array([1.0, 1.0, 1.0]),
        )
        eng.reload_model(str(tmp_path / "m.pkl"))
        with caplog.at_level("WARNING"):
            matches = eng.match(
                _l2_window(), None, _ctx(),
                feature_row=np.array([0.5, 0.5]),  # wrong width
            )
        assert all(m.primary_proba is None for m in matches)
        assert any(
            "width" in rec.message and "expected" in rec.message
            for rec in caplog.records
        )

    def test_legacy_engine_without_model_has_none_proba(self, tmp_path: Path):
        eng = _build_engine(tmp_path)
        # No reload_model call → no primary model.
        matches = eng.match(
            _l2_window(), None, _ctx(),
            feature_row={"any": 1.0},
        )
        assert all(m.primary_proba is None for m in matches)


# ---------------------------------------------------------------------------
# Wave 9 A5 — soft-prior at inference (engine emits base_rate when
# is_active=False on a model-loaded engine)
# ---------------------------------------------------------------------------


class TestSoftPriorAtInference:
    def test_inactive_emits_base_rate_when_model_loaded(
        self, tmp_path: Path,
    ):
        """is_active=False with model loaded → primary_proba = base_rate."""
        eng = _build_engine(tmp_path)
        eng._base_rate = 0.5  # explicit for clarity
        _write_model_pickle(
            tmp_path / "m.pkl",
            feature_names=["f0", "f1"],
            weights=np.array([2.0, -1.0]),
        )
        eng.reload_model(str(tmp_path / "m.pkl"))

        matches = eng.match(
            _l2_window(), None, _ctx(),
            feature_row={"f0": 1.0, "f1": 0.5},
            is_active=False,
        )
        assert len(matches) >= 1
        for m in matches:
            assert m.primary_proba == pytest.approx(0.5)

    def test_inactive_emits_base_rate_even_without_feature_row(
        self, tmp_path: Path,
    ):
        """When is_active=False, the model is not consulted at all —
        the engine emits base_rate regardless of feature_row."""
        eng = _build_engine(tmp_path)
        _write_model_pickle(
            tmp_path / "m.pkl", feature_names=["a"], weights=np.array([1.0]),
        )
        eng.reload_model(str(tmp_path / "m.pkl"))
        matches = eng.match(
            _l2_window(), None, _ctx(),
            feature_row=None,    # no features but is_active=False overrides
            is_active=False,
        )
        for m in matches:
            assert m.primary_proba == pytest.approx(eng._base_rate)

    def test_inactive_returns_none_when_model_not_loaded(
        self, tmp_path: Path,
    ):
        """Soft prior requires model loaded; without it, behavior is
        unchanged from Wave 9-A — None on every match."""
        eng = _build_engine(tmp_path)
        # No reload_model call.
        matches = eng.match(
            _l2_window(), None, _ctx(),
            feature_row={"a": 1.0},
            is_active=False,
        )
        for m in matches:
            assert m.primary_proba is None

    def test_active_runs_model_normally(self, tmp_path: Path):
        """is_active=True is equivalent to the Wave 9-A path —
        proba comes from the model, not base_rate."""
        eng = _build_engine(tmp_path)
        # Weights large + opposite signs → strong proba signal away from 0.5
        _write_model_pickle(
            tmp_path / "m.pkl",
            feature_names=["f0"],
            weights=np.array([5.0]),
        )
        eng.reload_model(str(tmp_path / "m.pkl"))
        matches = eng.match(
            _l2_window(), None, _ctx(),
            feature_row={"f0": 1.0},
            is_active=True,
        )
        for m in matches:
            assert m.primary_proba is not None
            # Strong positive feature with weight=5 → proba near 1
            assert m.primary_proba > 0.9
            assert m.primary_proba != pytest.approx(eng._base_rate)

    def test_is_active_none_preserves_legacy_wave_9a_behavior(
        self, tmp_path: Path,
    ):
        """is_active=None (legacy default) → identical to Wave 9-A path:
        feature_row provided + model loaded → model output."""
        eng = _build_engine(tmp_path)
        _write_model_pickle(
            tmp_path / "m.pkl",
            feature_names=["f0", "f1"],
            weights=np.array([2.0, -1.0]),
        )
        eng.reload_model(str(tmp_path / "m.pkl"))

        # Without specifying is_active — default behavior
        m1 = eng.match(
            _l2_window(), None, _ctx(),
            feature_row={"f0": 0.5, "f1": -0.5},
        )
        # Same call with is_active=True
        m2 = eng.match(
            _l2_window(), None, _ctx(),
            feature_row={"f0": 0.5, "f1": -0.5},
            is_active=True,
        )
        # Both should produce the same proba (model-driven)
        assert m1[0].primary_proba == pytest.approx(m2[0].primary_proba)

    def test_custom_base_rate_emitted_when_inactive(
        self, tmp_path: Path,
    ):
        """Constructor's base_rate parameter overrides the default 0.5."""
        from p6lab.correlation.engine import CorrelationEngine
        from p6lab.correlation.scorer import EnsembleScorer
        from p6lab.patterns.library import (
            OutcomeDistribution, PatternDefinition, PatternLibrary,
            PatternStatus,
        )
        from p6lab.patterns.template_matcher import (
            BOOK_SHAPE_DIM, PatternTemplate, TemplateMatcher,
        )

        lib = PatternLibrary(tmp_path / "library.yaml")
        lib.load()
        lib.add_pattern("p1", PatternDefinition(
            name="p1", l3_signature="x", l2_manifestation="y",
            l1_footprint="z", instruments=["NQ"], regime_specific=False,
            status=PatternStatus.ACTIVE,
            outcome_distribution={"5m": OutcomeDistribution(
                mean_atr=0.5, std=0.3, hit_rate=0.7, n=300,
            )},
        ))
        matcher = TemplateMatcher()
        matcher.templates["p1"] = PatternTemplate(
            pattern_id="p1",
            book_series=np.ones((10, BOOK_SHAPE_DIM)),
            feature_centroid=np.ones(12),
            pattern_context={"vix_regime": "normal"},
        )
        eng = CorrelationEngine(
            library=lib, matcher=matcher, scorer=EnsembleScorer(),
            base_rate=0.43,  # custom prior
        )
        _write_model_pickle(
            tmp_path / "m.pkl", feature_names=["a"], weights=np.array([1.0]),
        )
        eng.reload_model(str(tmp_path / "m.pkl"))
        matches = eng.match(
            _l2_window(), None, _ctx(),
            feature_row={"a": 0.5},
            is_active=False,
        )
        for m in matches:
            assert m.primary_proba == pytest.approx(0.43)
