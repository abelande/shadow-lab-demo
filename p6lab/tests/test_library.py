"""Tests for p6lab.patterns.library — YAML pattern registry."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from p6lab.patterns.library import (
    OutcomeDistribution,
    PatternDefinition,
    PatternLibrary,
    PatternStatus,
)


def _make_pattern(name: str = "test_pat", status: PatternStatus = PatternStatus.CANDIDATE) -> PatternDefinition:
    return PatternDefinition(
        name=name,
        l3_signature="burst_add_cancel",
        l2_manifestation="imbalance_shift",
        l1_footprint="spread_widen",
        instruments=["NQ"],
        status=status,
    )


class TestLoadSave:
    def test_load_missing_returns_empty_library(self, tmp_path: Path):
        lib = PatternLibrary(tmp_path / "nope.yaml")
        data = lib.load()
        assert data.version == 1
        assert data.patterns == {}

    def test_round_trip_preserves_pattern(self, tmp_path: Path):
        path = tmp_path / "library.yaml"
        lib = PatternLibrary(path)
        lib.load()
        lib.add_pattern("p1", _make_pattern("p1"))
        lib.save()
        assert path.exists()
        # reload into fresh library
        lib2 = PatternLibrary(path)
        data = lib2.load()
        assert "p1" in data.patterns
        assert data.patterns["p1"].l3_signature == "burst_add_cancel"

    def test_version_auto_increments(self, tmp_path: Path):
        path = tmp_path / "library.yaml"
        lib = PatternLibrary(path)
        lib.load()
        lib.add_pattern("p1", _make_pattern("p1"))
        lib.save()
        v1 = PatternLibrary(path).load().version
        lib.save()
        v2 = PatternLibrary(path).load().version
        assert v2 > v1

    def test_save_atomic_no_partial_on_interrupt(self, tmp_path: Path):
        """Saved file should either be complete or not exist — no .tmp leftovers."""
        path = tmp_path / "library.yaml"
        lib = PatternLibrary(path)
        lib.load()
        lib.add_pattern("p1", _make_pattern("p1"))
        lib.save()
        tmps = list(tmp_path.glob("*.tmp"))
        assert tmps == []


class TestValidationHash:
    def test_add_sets_validation_hash(self, tmp_path: Path):
        lib = PatternLibrary(tmp_path / "library.yaml")
        lib.load()
        p = _make_pattern()
        lib.add_pattern("p1", p)
        assert p.validation_hash.startswith("sha256:")

    def test_duplicate_name_raises(self, tmp_path: Path):
        lib = PatternLibrary(tmp_path / "library.yaml")
        lib.load()
        lib.add_pattern("p1", _make_pattern("p1"))
        with pytest.raises(ValueError):
            lib.add_pattern("p1", _make_pattern("p1"))


class TestStatusTransitions:
    def test_valid_transition_candidate_to_mined(self, tmp_path: Path):
        lib = PatternLibrary(tmp_path / "library.yaml")
        lib.load()
        lib.add_pattern("p1", _make_pattern("p1"))
        lib.promote("p1", PatternStatus.MINED_APPROVED)
        assert lib.get_active_patterns()["p1"].status == PatternStatus.MINED_APPROVED

    def test_invalid_transition_rejected_to_active(self, tmp_path: Path):
        lib = PatternLibrary(tmp_path / "library.yaml")
        lib.load()
        lib.add_pattern("p1", _make_pattern("p1", PatternStatus.REJECTED))
        with pytest.raises(ValueError):
            lib.promote("p1", PatternStatus.ACTIVE)

    def test_terminal_retired_blocks_further_transitions(self, tmp_path: Path):
        lib = PatternLibrary(tmp_path / "library.yaml")
        lib.load()
        lib.add_pattern("p1", _make_pattern("p1", PatternStatus.RETIRED))
        with pytest.raises(ValueError):
            lib.promote("p1", PatternStatus.ACTIVE)

    def test_promote_missing_pattern_raises(self, tmp_path: Path):
        lib = PatternLibrary(tmp_path / "library.yaml")
        lib.load()
        with pytest.raises(KeyError):
            lib.promote("nope", PatternStatus.ACTIVE)


class TestActivePatterns:
    def test_get_active_excludes_candidate_and_rejected(self, tmp_path: Path):
        lib = PatternLibrary(tmp_path / "library.yaml")
        lib.load()
        lib.add_pattern("cand", _make_pattern("cand", PatternStatus.CANDIDATE))
        lib.add_pattern("act", _make_pattern("act", PatternStatus.ACTIVE))
        lib.add_pattern("rej", _make_pattern("rej", PatternStatus.REJECTED))
        active = lib.get_active_patterns()
        assert "act" in active
        assert "cand" not in active
        assert "rej" not in active
