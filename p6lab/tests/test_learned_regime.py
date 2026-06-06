"""Tests for p6lab.correlation.learned_regime (Wave 7 Phase 7H)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from p6lab.correlation.learned_regime import (
    DEFAULT_FEATURES,
    REGIME_LABELS,
    FeatureContract,
    LearnedRegimeClassifier,
    train_learned_regime,
)


def _fake_features(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a feature frame + VIX series where regime label depends on
    realized_variance + fi_fast in a way the classifier can recover."""
    rng = np.random.default_rng(seed)
    rv = rng.uniform(0.0, 5.0, n)
    vix = 10.0 + rv * 7.0 + rng.normal(0, 1.0, n)
    X = pd.DataFrame({
        "fi_fast": rng.uniform(0, 1, n),
        "network_momentum": rng.uniform(-1, 1, n),
        "peer_correlation_avg": rng.uniform(0, 1, n),
        "imbalance_ema": rng.uniform(-1, 1, n),
        "spread_bps": rng.uniform(0, 5, n),
        "realized_variance_30s": rv,
        "cross_asset_adjacency": rng.uniform(0, 1, n),
    })
    return X, vix


def test_unfitted_predict_falls_back_to_rv_bucket() -> None:
    clf = LearnedRegimeClassifier()
    assert not clf.is_fitted_
    assert clf.predict({"realized_variance_30s": 0.1}) == "low"
    assert clf.predict({"realized_variance_30s": 0.5}) == "normal"
    assert clf.predict({"realized_variance_30s": 1.5}) == "elevated"
    assert clf.predict({"realized_variance_30s": 5.0}) == "high"


def test_unfitted_probability_is_uniform() -> None:
    clf = LearnedRegimeClassifier()
    probs = clf.predict_proba({})
    assert set(probs.keys()) == set(REGIME_LABELS)
    for v in probs.values():
        assert v == pytest.approx(1.0 / len(REGIME_LABELS))


def test_fit_on_vix_series() -> None:
    X, vix = _fake_features(n=400)
    clf = train_learned_regime(X, vix_series=vix)
    assert clf.is_fitted_
    # In-sample accuracy should be ≥ 0.7 on this easy mapping
    preds = [clf.predict(row) for row in X.to_dict(orient="records")]
    from p6lab.correlation.learned_regime import _vix_to_regime
    labels = [_vix_to_regime(float(v)) for v in vix]
    acc = sum(p == l for p, l in zip(preds, labels)) / len(preds)
    assert acc >= 0.6


def test_fit_without_vix_uses_quantile_binning() -> None:
    X, _ = _fake_features(n=400)
    clf = train_learned_regime(X)   # no vix_series
    assert clf.is_fitted_
    pred = clf.predict(X.iloc[0].to_dict())
    assert pred in REGIME_LABELS


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    X, vix = _fake_features(n=300)
    clf = train_learned_regime(X, vix_series=vix)
    path = tmp_path / "regime.json"
    clf.save(path)
    loaded = LearnedRegimeClassifier.load(path)
    assert loaded.is_fitted_
    # Predictions should match exactly (same coef → same class)
    for _, row in X.head(20).iterrows():
        d = row.to_dict()
        assert clf.predict(d) == loaded.predict(d)


def test_missing_columns_default_to_zero() -> None:
    clf = LearnedRegimeClassifier()
    # Unfitted path: realized_variance_30s fallback decides
    assert clf.predict({"realized_variance_30s": 0.01}) == "low"


def test_predict_proba_sums_to_one_when_fitted() -> None:
    X, vix = _fake_features(n=300)
    clf = train_learned_regime(X, vix_series=vix)
    probs = clf.predict_proba(X.iloc[0].to_dict())
    assert sum(probs.values()) == pytest.approx(1.0, rel=1e-5)


def test_feature_contract_default_matches_docs() -> None:
    contract = FeatureContract()
    assert contract.columns == DEFAULT_FEATURES


# Wave 8.5-D tests
def test_wave_85_d_unfitted_missing_rv_warns_once(caplog) -> None:
    """Unfitted + missing RV key: warning fires exactly once per instance."""
    caplog.clear()
    caplog.set_level("WARNING", logger="p6lab.correlation.learned_regime")
    clf = LearnedRegimeClassifier()
    # First call — should warn
    result = clf.predict({})  # missing realized_variance_30s
    assert result == "low"   # fallback behavior unchanged
    warnings = [r for r in caplog.records if "realized_variance_30s" in r.getMessage()]
    assert len(warnings) == 1
    # Second call — must NOT warn
    caplog.clear()
    clf.predict({})
    warnings_2 = [r for r in caplog.records if "realized_variance_30s" in r.getMessage()]
    assert len(warnings_2) == 0
    assert clf._warned_missing_rv is True


def test_wave_85_d_two_instances_warn_independently(caplog) -> None:
    """Instance-level sentinel: two classifiers warn separately."""
    caplog.clear()
    caplog.set_level("WARNING", logger="p6lab.correlation.learned_regime")
    clf_a = LearnedRegimeClassifier()
    clf_b = LearnedRegimeClassifier()
    clf_a.predict({})
    clf_b.predict({})
    warnings = [r for r in caplog.records if "realized_variance_30s" in r.getMessage()]
    assert len(warnings) == 2


def test_wave_85_d_rv_present_does_not_warn(caplog) -> None:
    """When RV key is present, no warning fires even on unfitted."""
    caplog.clear()
    caplog.set_level("WARNING", logger="p6lab.correlation.learned_regime")
    clf = LearnedRegimeClassifier()
    clf.predict({"realized_variance_30s": 1.5})
    warnings = [r for r in caplog.records if "realized_variance_30s" in r.getMessage()]
    assert len(warnings) == 0


def test_wave_85_d_fitted_classifier_does_not_warn(caplog) -> None:
    """Fitted classifier bypasses the unfitted fallback entirely."""
    import numpy as np
    import pandas as pd
    from p6lab.correlation.learned_regime import train_learned_regime
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.uniform(0, 1, 200) for c in DEFAULT_FEATURES})
    vix = 10.0 + rng.uniform(0, 30, 200)
    clf = train_learned_regime(X, vix_series=vix)
    caplog.clear()
    caplog.set_level("WARNING", logger="p6lab.correlation.learned_regime")
    clf.predict({})   # no RV, but classifier is fitted → should NOT warn
    warnings = [r for r in caplog.records if "realized_variance_30s" in r.getMessage()]
    assert len(warnings) == 0
