"""Tests for p6lab.validation.augmentation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p6lab.validation.augmentation import AugmentationEngine, AugmentedSample


@pytest.fixture
def feats() -> pd.DataFrame:
    return pd.DataFrame({
        "bid_size": np.linspace(10, 20, 50),
        "ask_size": np.linspace(15, 5, 50),
        "spread_ticks": np.ones(50),
        "tick_velocity": np.zeros(50),
    })


class TestTransforms:
    def test_time_stretch_increases_length(self, feats):
        out = AugmentationEngine().time_stretch_compress(feats, 1.5)
        assert len(out) > len(feats)

    def test_time_compress_decreases_length(self, feats):
        out = AugmentationEngine().time_stretch_compress(feats, 0.5)
        assert len(out) < len(feats)

    def test_depth_scale_modifies_depth_cols(self, feats):
        out = AugmentationEngine().depth_scale(feats, 2.0)
        assert out["bid_size"].iloc[0] == pytest.approx(20.0)
        assert out["spread_ticks"].iloc[0] == pytest.approx(1.0)  # not depth

    def test_volatility_shift_changes_vol_cols(self, feats):
        out = AugmentationEngine(random_state=0).volatility_shift(feats, atr_shift=1.0)
        assert not np.allclose(out["spread_ticks"], feats["spread_ticks"])

    def test_phase_jitter_preserves_columns(self, feats):
        out = AugmentationEngine().phase_duration_jitter(feats, 1.2)
        assert list(out.columns) == list(feats.columns)


class TestGenerate:
    def test_first_sample_is_original(self, feats):
        eng = AugmentationEngine()
        samples = eng.generate(feats, label=1, instrument="NQ", n_samples=5)
        assert samples[0].augmentation_method == "original"
        assert len(samples) == 5

    def test_methods_are_tagged(self, feats):
        samples = AugmentationEngine().generate(feats, label=1, instrument="NQ", n_samples=6)
        methods = {s.augmentation_method for s in samples}
        assert "original" in methods
        assert len(methods) > 1
