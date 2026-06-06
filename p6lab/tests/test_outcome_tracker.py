"""Tests for OutcomeTrackerRenderer (Wave 5 Phase 5B)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from p6lab.correlation.match_broker import MatchBroker
from p6lab.correlation.renderers.outcome_tracker import (
    DEFAULT_HORIZON_MS,
    OutcomeTrackerRenderer,
)
from p6lab.patterns.library import (
    OutcomeDistribution,
    PatternDefinition,
    PatternLibrary,
    PatternStatus,
)


@dataclass
class _FakeMatch:
    """Shape-compatible with engine.PatternMatch — only the fields the tracker reads."""
    pattern_id: str
    expected_direction: str
    expected_move_atr: float = 1.0
    match_window_end_ms: int = 0
    instrument: str = "NQ"
    confidence_tier: str = "B"
    regime: str = "normal"
    ensemble_score: float = 0.75


@pytest.fixture
def tmp_outcomes(tmp_path: Path) -> Path:
    return tmp_path / "outcomes.jsonl"


@pytest.fixture
def seeded_library(tmp_path: Path) -> PatternLibrary:
    """Library with one ACTIVE and one MINED_APPROVED pattern for retirement tests."""
    lib_path = tmp_path / "library.yaml"
    lib = PatternLibrary(lib_path)
    lib.load()
    lib.add_pattern("bull_flag_reload", PatternDefinition(
        name="bull_flag_reload",
        l3_signature="sig",
        l2_manifestation="manif",
        l1_footprint="foot",
        min_sample_size=10,
        status=PatternStatus.ACTIVE,
        outcome_distribution={"5m": OutcomeDistribution(
            mean_atr=0.5, std=0.1, hit_rate=0.6, n=300,
        )},
    ))
    lib.add_pattern("candidate_pat", PatternDefinition(
        name="candidate_pat",
        l3_signature="sig2",
        l2_manifestation="m2",
        l1_footprint="f2",
        min_sample_size=10,
        status=PatternStatus.MINED_APPROVED,
        outcome_distribution={"5m": OutcomeDistribution(
            mean_atr=0.2, std=0.05, hit_rate=0.55, n=200,
        )},
    ))
    lib.save()
    return lib


def test_noop_when_no_price(tmp_outcomes: Path) -> None:
    tracker = OutcomeTrackerRenderer(tmp_outcomes)
    tracker(_FakeMatch("p", "bull", match_window_end_ms=1_000))
    # No price seen → match is dropped
    assert tracker.matches_dropped_no_price == 1
    assert tracker.pending_count == 0
    assert tracker.outcomes_closed == 0


def test_hit_outcome_recorded_and_written(tmp_outcomes: Path) -> None:
    tracker = OutcomeTrackerRenderer(tmp_outcomes, horizon_ms=1_000)
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker(_FakeMatch("p1", "bull", match_window_end_ms=100))
    # Price ticks past horizon with a +10-tick move
    tracker.on_price("NQ", mid=20_010.0, ts_ms=2_000)

    assert tracker.outcomes_closed == 1
    assert tracker.pending_count == 0

    rows = [json.loads(l) for l in tmp_outcomes.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["pattern_id"] == "p1"
    assert row["hit"] is True
    assert row["realized_return"] == pytest.approx(10.0)


def test_miss_outcome_on_adverse_move(tmp_outcomes: Path) -> None:
    tracker = OutcomeTrackerRenderer(tmp_outcomes, horizon_ms=1_000)
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker(_FakeMatch("p_bear", "bear", match_window_end_ms=100))
    # Went UP when we expected DOWN — a miss
    tracker.on_price("NQ", mid=20_005.0, ts_ms=2_000)

    rows = [json.loads(l) for l in tmp_outcomes.read_text().splitlines() if l.strip()]
    assert rows[0]["hit"] is False
    assert rows[0]["realized_return"] == pytest.approx(-5.0)


def test_direction_sign_applied_for_bear(tmp_outcomes: Path) -> None:
    tracker = OutcomeTrackerRenderer(tmp_outcomes, horizon_ms=1_000)
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker(_FakeMatch("bear_pat", "bear", match_window_end_ms=0))
    # Price dropped → bear match should realize +positive return
    tracker.on_price("NQ", mid=19_990.0, ts_ms=2_000)
    row = json.loads(tmp_outcomes.read_text().splitlines()[0])
    assert row["hit"] is True
    assert row["realized_return"] == pytest.approx(10.0)


def test_exit_not_triggered_before_horizon(tmp_outcomes: Path) -> None:
    tracker = OutcomeTrackerRenderer(tmp_outcomes, horizon_ms=60_000)
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker(_FakeMatch("p", "bull", match_window_end_ms=0))
    # 100 ticks advance but still < 60s horizon — no close yet
    tracker.on_price("NQ", mid=20_050.0, ts_ms=10_000)
    assert tracker.outcomes_closed == 0
    assert tracker.pending_count == 1


def test_flush_closes_pending(tmp_outcomes: Path) -> None:
    tracker = OutcomeTrackerRenderer(tmp_outcomes, horizon_ms=60_000)
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker(_FakeMatch("p", "bull", match_window_end_ms=0))
    tracker(_FakeMatch("p", "bull", match_window_end_ms=500))
    # Flush uses the latest observed mid (20_005)
    tracker.on_price("NQ", mid=20_005.0, ts_ms=1_000)
    closed = tracker.flush()
    assert closed == 2
    assert tracker.pending_count == 0


def test_neutral_direction_ignored_as_hit(tmp_outcomes: Path) -> None:
    tracker = OutcomeTrackerRenderer(tmp_outcomes, horizon_ms=1_000)
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker(_FakeMatch("p_n", "neutral", match_window_end_ms=0))
    tracker.on_price("NQ", mid=20_100.0, ts_ms=2_000)
    row = json.loads(tmp_outcomes.read_text().splitlines()[0])
    # neutral direction → signed return is 0 → hit=False
    assert row["hit"] is False
    assert row["realized_return"] == pytest.approx(0.0)


def test_broker_wiring_end_to_end(tmp_outcomes: Path) -> None:
    broker = MatchBroker()
    tracker = OutcomeTrackerRenderer(tmp_outcomes, horizon_ms=1_000)
    broker.subscribe(tracker)
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    broker.emit(_FakeMatch("p_broker", "bull", match_window_end_ms=100))
    tracker.on_price("NQ", mid=20_010.0, ts_ms=2_000)
    assert tracker.outcomes_closed == 1


def test_reaggregate_updates_library_outcome(
    tmp_outcomes: Path, seeded_library: PatternLibrary
) -> None:
    tracker = OutcomeTrackerRenderer(
        tmp_outcomes, library=seeded_library, horizon_ms=500,
        reaggregate_every_n=0,   # disable automatic — we'll call manually
    )
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    # 3 hits in a row
    for i in range(3):
        tracker(_FakeMatch(
            "bull_flag_reload", "bull",
            match_window_end_ms=i * 100,
        ))
    tracker.on_price("NQ", mid=20_020.0, ts_ms=5_000)
    assert tracker.outcomes_closed == 3
    tracker.reaggregate()

    # Re-read library from disk — value should be rewritten
    reloaded = PatternLibrary(seeded_library.library_path)
    reloaded.load()
    od = reloaded._data.patterns["bull_flag_reload"].outcome_distribution["5m"]
    assert od.n == 3
    assert od.hit_rate == pytest.approx(1.0)
    # All three outcomes resolved to +20 → mean_atr (raw mean) ≈ 20
    assert od.mean_atr == pytest.approx(20.0)


def test_retirement_triggered_by_low_hit_rate(
    tmp_outcomes: Path, seeded_library: PatternLibrary
) -> None:
    tracker = OutcomeTrackerRenderer(
        tmp_outcomes, library=seeded_library,
        horizon_ms=500,
        reaggregate_every_n=0,
        retire_below_hit_rate=0.5,
    )
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    # bull_flag_reload has min_sample_size=10, so need ≥ 8 closes.
    # Emit 10 bull matches; price moves DOWN so every one is a miss.
    for i in range(10):
        tracker(_FakeMatch(
            "bull_flag_reload", "bull",
            match_window_end_ms=i * 100,
        ))
    tracker.on_price("NQ", mid=19_980.0, ts_ms=5_000)
    assert tracker.outcomes_closed == 10

    tracker.reaggregate()
    reloaded = PatternLibrary(seeded_library.library_path)
    reloaded.load()
    assert reloaded._data.patterns["bull_flag_reload"].status == PatternStatus.RETIRED
    assert tracker.retirements == 1


def test_retirement_skipped_below_min_sample(
    tmp_outcomes: Path, seeded_library: PatternLibrary
) -> None:
    tracker = OutcomeTrackerRenderer(
        tmp_outcomes, library=seeded_library,
        horizon_ms=500,
        reaggregate_every_n=0,
        retire_below_hit_rate=0.5,
    )
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    # Only 5 matches — under min_sample_size × 0.8 = 8
    for i in range(5):
        tracker(_FakeMatch(
            "bull_flag_reload", "bull",
            match_window_end_ms=i * 100,
        ))
    tracker.on_price("NQ", mid=19_980.0, ts_ms=5_000)
    tracker.reaggregate()

    reloaded = PatternLibrary(seeded_library.library_path)
    reloaded.load()
    # Status must remain ACTIVE (sample too small to retire)
    assert reloaded._data.patterns["bull_flag_reload"].status == PatternStatus.ACTIVE
    assert tracker.retirements == 0


def test_automatic_reaggregation_cadence(
    tmp_outcomes: Path, seeded_library: PatternLibrary
) -> None:
    tracker = OutcomeTrackerRenderer(
        tmp_outcomes, library=seeded_library,
        horizon_ms=1_000,
        reaggregate_every_n=2,   # every 2 closes
        retire_below_hit_rate=0.0,  # never retire — isolate reagg signal
    )
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    for i in range(4):
        tracker(_FakeMatch("bull_flag_reload", "bull", match_window_end_ms=i * 100))
    tracker.on_price("NQ", mid=20_050.0, ts_ms=5_000)
    # 4 closes with reaggregate_every_n=2 → exactly 2 library updates
    assert tracker.library_updates == 2


def test_library_none_skips_self_update(tmp_outcomes: Path) -> None:
    tracker = OutcomeTrackerRenderer(
        tmp_outcomes, library=None, horizon_ms=500,
    )
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker(_FakeMatch("p", "bull", match_window_end_ms=0))
    tracker.on_price("NQ", mid=20_010.0, ts_ms=2_000)
    assert tracker.outcomes_closed == 1
    # No library passed → no updates, but outcome still written
    assert tracker.library_updates == 0


def test_outcome_row_shape(tmp_outcomes: Path) -> None:
    tracker = OutcomeTrackerRenderer(tmp_outcomes, horizon_ms=500)
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker(_FakeMatch(
        "p_full", "bull", match_window_end_ms=100,
        confidence_tier="A", regime="high_vol", expected_move_atr=0.8,
    ))
    tracker.on_price("NQ", mid=20_010.0, ts_ms=2_000)
    row = json.loads(tmp_outcomes.read_text().splitlines()[0])
    for k in (
        "pattern_id", "symbol", "entry_ts_ms", "exit_ts_ms", "entry_mid",
        "exit_mid", "realized_return", "expected_direction",
        "confidence_tier", "regime", "hit",
    ):
        assert k in row, f"missing key {k}"
    assert row["confidence_tier"] == "A"
    assert row["regime"] == "high_vol"


def test_jsonl_is_append_only(tmp_outcomes: Path) -> None:
    """Ensure two separate tracker sessions don't overwrite the JSONL file."""
    tracker1 = OutcomeTrackerRenderer(tmp_outcomes, horizon_ms=500)
    tracker1.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker1(_FakeMatch("a", "bull", match_window_end_ms=0))
    tracker1.on_price("NQ", mid=20_010.0, ts_ms=1_000)

    tracker2 = OutcomeTrackerRenderer(tmp_outcomes, horizon_ms=500)
    tracker2.on_price("NQ", mid=20_010.0, ts_ms=2_000)
    tracker2(_FakeMatch("b", "bull", match_window_end_ms=2_500))
    tracker2.on_price("NQ", mid=20_020.0, ts_ms=4_000)

    rows = [json.loads(l) for l in tmp_outcomes.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    assert rows[0]["pattern_id"] == "a"
    assert rows[1]["pattern_id"] == "b"


def test_pending_count_property(tmp_outcomes: Path) -> None:
    tracker = OutcomeTrackerRenderer(tmp_outcomes, horizon_ms=60_000)
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker(_FakeMatch("a", "bull", match_window_end_ms=0))
    tracker(_FakeMatch("b", "bull", match_window_end_ms=100))
    assert tracker.pending_count == 2
    tracker.flush()
    assert tracker.pending_count == 0
