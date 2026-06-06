"""Unit tests for FIConditioner and MatchContext.fi_bucket gating."""
from __future__ import annotations

from pathlib import Path

import pytest

from p6lab.correlation.regime_conditioner import FIConditioner
from p6lab.patterns.library import (
    OutcomeDistribution,
    PatternDefinition,
    PatternLibrary,
    PatternStatus,
)


@pytest.fixture
def library(tmp_path: Path) -> PatternLibrary:
    lib = PatternLibrary(tmp_path / "lib.yaml")
    lib.load()
    # Regime-agnostic pattern (no fi_bucket tag)
    lib.add_pattern(
        "unconditional",
        PatternDefinition(
            name="unconditional",
            l3_signature="foo", l2_manifestation="bar", l1_footprint="baz",
            instruments=["NQ"], regime_specific=False,
            status=PatternStatus.ACTIVE,
            outcome_distribution={"5m": OutcomeDistribution(
                mean_atr=0.5, std=0.3, hit_rate=0.6, n=500,
            )},
        ),
    )
    # Fragile-only pattern (outcome key tagged _fragile)
    lib.add_pattern(
        "fragile_only",
        PatternDefinition(
            name="fragile_only",
            l3_signature="foo", l2_manifestation="bar", l1_footprint="baz",
            instruments=["NQ"], regime_specific=False,
            status=PatternStatus.ACTIVE,
            outcome_distribution={"5m_fragile": OutcomeDistribution(
                mean_atr=0.7, std=0.4, hit_rate=0.8, n=500,
            )},
        ),
    )
    # Calm-only pattern (outcome key tagged _calm)
    lib.add_pattern(
        "calm_only",
        PatternDefinition(
            name="calm_only",
            l3_signature="foo", l2_manifestation="bar", l1_footprint="baz",
            instruments=["NQ"], regime_specific=False,
            status=PatternStatus.ACTIVE,
            outcome_distribution={"5m_calm": OutcomeDistribution(
                mean_atr=0.2, std=0.15, hit_rate=0.75, n=500,
            )},
        ),
    )
    return lib


class TestClassifyFI:
    @pytest.mark.parametrize("fi,expected", [
        (0.0, "calm"),
        (0.15, "calm"),
        (0.29, "calm"),
        (0.30, "elevated"),
        (0.45, "elevated"),
        (0.59, "elevated"),
        (0.60, "fragile"),
        (0.85, "fragile"),
        (1.00, "fragile"),
    ])
    def test_buckets(self, fi: float, expected: str) -> None:
        assert FIConditioner().classify_fi(fi) == expected


class TestSelectPatterns:
    def test_calm_matches_unconditional_and_calm_only(self, library: PatternLibrary) -> None:
        cond = FIConditioner()
        selected = set(cond.select_patterns(library, "NQ", "calm"))
        assert "unconditional" in selected      # no fi tag → always matches
        assert "calm_only" in selected
        assert "fragile_only" not in selected

    def test_fragile_matches_unconditional_and_fragile_only(self, library: PatternLibrary) -> None:
        cond = FIConditioner()
        selected = set(cond.select_patterns(library, "NQ", "fragile"))
        assert "unconditional" in selected
        assert "fragile_only" in selected
        assert "calm_only" not in selected

    def test_elevated_matches_only_unconditional(self, library: PatternLibrary) -> None:
        cond = FIConditioner()
        selected = set(cond.select_patterns(library, "NQ", "elevated"))
        assert selected == {"unconditional"}

    def test_instrument_filter_applies(self, library: PatternLibrary) -> None:
        cond = FIConditioner()
        selected = cond.select_patterns(library, "ES", "calm")   # wrong instrument
        assert selected == []
