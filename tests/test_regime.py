"""Tests for regime classifier and weights (Layer 5)."""
from __future__ import annotations
import pytest
from p6.regime_context.regime_classifier import RegimeClassifier
from p6.regime_context.regime_weights import get_regime_weights
from p6.regime_context.abstain_policy import AbstainPolicy
from p6.models import RegimeType, RegimeWeights


def test_regime_classifier_none_returns_unknown():
    cls = RegimeClassifier()
    assert cls.classify(None) == RegimeType.UNKNOWN


def test_regime_classifier_empty_dict():
    cls = RegimeClassifier()
    assert cls.classify({}) == RegimeType.UNKNOWN


def test_regime_classifier_explicit_trending():
    cls = RegimeClassifier()
    assert cls.classify({"regime": "TRENDING"}) == RegimeType.TRENDING


def test_regime_classifier_explicit_ranging():
    cls = RegimeClassifier()
    assert cls.classify({"regime": "RANGING"}) == RegimeType.RANGING


def test_regime_classifier_explicit_volatile():
    cls = RegimeClassifier()
    assert cls.classify({"regime": "VOLATILE"}) == RegimeType.VOLATILE


def test_regime_classifier_heuristic_high_vol():
    cls = RegimeClassifier()
    # heuristic path triggered by unrecognized regime label
    result = cls.classify({"regime": "UNRECOGNIZED", "volatility": 0.9, "trend_strength": 0.3})
    assert result == RegimeType.VOLATILE


def test_regime_classifier_heuristic_high_trend():
    cls = RegimeClassifier()
    result = cls.classify({"regime": "UNRECOGNIZED", "trend_strength": 0.8, "volatility": 0.2})
    assert result == RegimeType.TRENDING


def test_get_regime_weights_trending_l2_highest():
    w = get_regime_weights(RegimeType.TRENDING)
    assert w.l2_weight >= w.l1_weight
    assert w.abstain is False


def test_get_regime_weights_ranging_l1_highest():
    w = get_regime_weights(RegimeType.RANGING)
    assert w.l1_weight >= w.l2_weight
    assert w.abstain is False


def test_get_regime_weights_volatile_abstain():
    w = get_regime_weights(RegimeType.VOLATILE)
    assert w.abstain is True


def test_get_regime_weights_unknown_balanced():
    w = get_regime_weights(RegimeType.UNKNOWN)
    assert w.l1_weight == w.l2_weight == w.l3_weight == w.l4_weight
    assert w.abstain is False


def test_abstain_policy_abstains_on_low_confidence():
    policy = AbstainPolicy()
    w = get_regime_weights(RegimeType.TRENDING)
    result = policy.should_abstain(
        regime_weights=w, confidence=0.1, authenticity_score=0.9, pressure_abs=0.5
    )
    assert result is True


def test_abstain_policy_no_abstain_on_good_signal():
    policy = AbstainPolicy()
    w = get_regime_weights(RegimeType.TRENDING)
    result = policy.should_abstain(
        regime_weights=w, confidence=0.8, authenticity_score=0.9, pressure_abs=0.7
    )
    assert result is False
