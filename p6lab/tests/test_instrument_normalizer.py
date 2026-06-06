"""Tests for p6lab.ingestion.instrument_normalizer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p6lab.ingestion.instrument_normalizer import (
    InstrumentNormalizer, NormalizerConfig, VIXRegime,
)


def _norm(tmp_path: Path) -> InstrumentNormalizer:
    return InstrumentNormalizer(NormalizerConfig(
        symbol="NQ", tick_size=0.25, atr_20d=2.0, median_depth_20d=100.0,
        cache_dir=tmp_path,
    ))


class TestRegime:
    def test_buckets(self):
        n = InstrumentNormalizer(NormalizerConfig(symbol="NQ", tick_size=0.25))
        assert n.classify_regime(10) == VIXRegime.LOW
        assert n.classify_regime(20) == VIXRegime.NORMAL
        assert n.classify_regime(30) == VIXRegime.ELEVATED
        assert n.classify_regime(50) == VIXRegime.HIGH


class TestNormalize:
    def test_depth_ratio(self, tmp_path: Path):
        n = _norm(tmp_path)
        assert n.normalize_depth(50.0) == pytest.approx(0.5)

    def test_spread_bps(self, tmp_path: Path):
        n = _norm(tmp_path)
        assert n.spread_to_bps(99.75, 100.25) == pytest.approx(50.0, abs=0.1)

    def test_price_move_atr(self, tmp_path: Path):
        n = _norm(tmp_path)
        assert n.normalize_price_move(4.0) == pytest.approx(2.0)

    def test_book_shape_each_side_sums_one(self, tmp_path: Path):
        n = _norm(tmp_path)
        v = np.arange(40, dtype=float) + 1
        out = n.normalize_book_shape_vector(v)
        assert out[:20].sum() == pytest.approx(1.0)
        assert out[20:].sum() == pytest.approx(1.0)


class TestCacheRoundTrip:
    def test_update_then_load(self, tmp_path: Path):
        n = _norm(tmp_path)
        depth = pd.Series(np.linspace(50, 150, 30))
        atr = pd.Series(np.linspace(1.0, 3.0, 30))
        n.update_calibration(depth, atr)
        cached = InstrumentNormalizer.from_cache("NQ", tmp_path)
        assert cached.config.median_depth_20d == pytest.approx(n.config.median_depth_20d)
        assert cached.config.atr_20d == pytest.approx(n.config.atr_20d)

    def test_from_cache_missing_returns_default(self, tmp_path: Path):
        n = InstrumentNormalizer.from_cache("XYZ", tmp_path)
        assert n.config.symbol == "XYZ"
        assert n.config.median_depth_20d == 1.0  # default
