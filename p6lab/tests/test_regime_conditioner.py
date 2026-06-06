"""Tests for p6lab.correlation.regime_conditioner."""
from __future__ import annotations

from pathlib import Path

import pytest

from p6lab.correlation.regime_conditioner import RegimeConditioner
from p6lab.patterns.library import (
    OutcomeDistribution, PatternDefinition, PatternLibrary, PatternStatus,
)


def _lib_with(patterns: dict[str, PatternDefinition], tmp_path: Path) -> PatternLibrary:
    lib = PatternLibrary(tmp_path / "library.yaml")
    lib.load()
    for name, p in patterns.items():
        lib.add_pattern(name, p)
    return lib


class TestClassify:
    def test_buckets(self):
        rc = RegimeConditioner()
        assert rc.classify_regime(10) == "low"
        assert rc.classify_regime(20) == "normal"
        assert rc.classify_regime(30) == "elevated"
        assert rc.classify_regime(50) == "high"


class TestSelect:
    def test_filters_inactive(self, tmp_path: Path):
        rc = RegimeConditioner()
        p1 = PatternDefinition(name="p1", l3_signature="x", l2_manifestation="y",
                               l1_footprint="z", instruments=["NQ"],
                               regime_specific=False, status=PatternStatus.ACTIVE)
        p2 = PatternDefinition(name="p2", l3_signature="x", l2_manifestation="y",
                               l1_footprint="z", instruments=["NQ"],
                               regime_specific=False, status=PatternStatus.CANDIDATE)
        lib = _lib_with({"p1": p1, "p2": p2}, tmp_path)
        sel = rc.select_patterns(lib, "NQ", "normal")
        assert "p1" in sel.selected_pattern_ids
        assert "p2" not in sel.selected_pattern_ids

    def test_filters_by_instrument(self, tmp_path: Path):
        rc = RegimeConditioner()
        p1 = PatternDefinition(name="p1", l3_signature="x", l2_manifestation="y",
                               l1_footprint="z", instruments=["ES"],
                               regime_specific=False, status=PatternStatus.ACTIVE)
        lib = _lib_with({"p1": p1}, tmp_path)
        sel = rc.select_patterns(lib, "NQ", "normal")
        assert "p1" in sel.rejected_pattern_ids
