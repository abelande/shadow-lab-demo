"""Tests for Type-D (cross-instrument contagion) cascade detector (Wave 7 Phase 7G)."""
from __future__ import annotations

import pytest

from p6lab.patterns.cascade_taxonomy import (
    CascadeClassifier,
    CascadeEvent,
    CascadeType,
)


def _mk_event(ts_ms: int, cascade_type: CascadeType = CascadeType.MOMENTUM_IGNITION) -> CascadeEvent:
    return CascadeEvent(
        cascade_type=cascade_type,
        anchor_ts_ms=ts_ms,
        end_ts_ms=ts_ms + 100,
        confidence=0.8,
    )


def test_no_events_yields_no_type_d() -> None:
    clf = CascadeClassifier()
    out = clf.detect_cross_instrument_contagion(events_by_symbol={})
    assert out == []


def test_single_symbol_does_not_fire_type_d() -> None:
    clf = CascadeClassifier()
    out = clf.detect_cross_instrument_contagion(
        events_by_symbol={"NQ": [_mk_event(1_000), _mk_event(2_000)]}
    )
    assert out == []


def test_two_symbols_fire_within_window() -> None:
    clf = CascadeClassifier()
    out = clf.detect_cross_instrument_contagion(events_by_symbol={
        "NQ": [_mk_event(1_000)],
        "ES": [_mk_event(1_500)],
    })
    # Without gating dicts, the pair passes by default
    assert len(out) == 1
    assert out[0].cascade_type == CascadeType.CROSS_INSTRUMENT_CONTAGION
    assert "NQ" in out[0].metadata["symbols"]
    assert "ES" in out[0].metadata["symbols"]


def test_coherence_gate_blocks_uncorrelated_pair() -> None:
    clf = CascadeClassifier()
    # Low coherence, low adjacency → pair rejected
    out = clf.detect_cross_instrument_contagion(
        events_by_symbol={
            "NQ": [_mk_event(1_000)],
            "YM": [_mk_event(1_500)],
        },
        coherence_matrix={("NQ", "YM"): 0.10},
        adjacency_matrix={("NQ", "YM"): 0.10},
    )
    assert out == []


def test_coherence_gate_allows_correlated_pair() -> None:
    clf = CascadeClassifier()
    out = clf.detect_cross_instrument_contagion(
        events_by_symbol={
            "NQ": [_mk_event(1_000)],
            "ES": [_mk_event(1_500)],
        },
        coherence_matrix={("ES", "NQ"): 0.85},
    )
    assert len(out) == 1


def test_cooldown_prevents_duplicates() -> None:
    clf = CascadeClassifier()
    # Three co-firing events within cooldown
    out = clf.detect_cross_instrument_contagion(events_by_symbol={
        "NQ": [_mk_event(1_000), _mk_event(3_000)],
        "ES": [_mk_event(1_200), _mk_event(3_200)],
    })
    # With cooldown of 15s, only the first cluster should produce an event
    assert len(out) == 1


def test_three_symbol_contagion_has_higher_confidence() -> None:
    clf = CascadeClassifier()
    out = clf.detect_cross_instrument_contagion(events_by_symbol={
        "NQ": [_mk_event(1_000)],
        "ES": [_mk_event(1_100)],
        "YM": [_mk_event(1_200)],
    })
    assert len(out) == 1
    assert out[0].metadata["cluster_size"] == 3
    assert out[0].confidence > 0.6


def test_single_instrument_classify_snapshots_never_emits_type_d() -> None:
    """Batch API must not synthesize Type D from single-instrument data."""
    clf = CascadeClassifier()
    events = clf.classify_snapshots([])
    for e in events:
        assert e.cascade_type != CascadeType.CROSS_INSTRUMENT_CONTAGION
