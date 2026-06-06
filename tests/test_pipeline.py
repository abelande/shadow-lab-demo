"""Tests for the full OrderBookMetaPipeline."""
from __future__ import annotations
import pytest
from p6.pipeline import OrderBookMetaPipeline
from p6.models import (
    DepthIndicatorFrame, StaircaseProfile, GameState, ForceVector,
    AuthenticityProfile, RegimeWeights, RegimeType,
)


def test_pipeline_run_returns_frame(sample_snapshot):
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(sample_snapshot)
    assert isinstance(frame, DepthIndicatorFrame)


def test_pipeline_frame_has_all_layers(sample_snapshot):
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(sample_snapshot)
    assert frame.staircase is not None
    assert frame.game_state is not None
    assert frame.force_vector is not None
    assert frame.authenticity is not None
    assert frame.regime_weights is not None


def test_pipeline_direction_in_range(sample_snapshot):
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(sample_snapshot)
    assert -1.0 <= frame.direction <= 1.0


def test_pipeline_confidence_in_range(sample_snapshot):
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(sample_snapshot)
    assert 0.0 <= frame.confidence <= 1.0


def test_pipeline_urgency_in_range(sample_snapshot):
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(sample_snapshot)
    assert 0.0 <= frame.urgency <= 1.0


def test_pipeline_size_multiplier_nonnegative(sample_snapshot):
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(sample_snapshot)
    assert frame.size_multiplier >= 0.0


def test_pipeline_with_regime_output(sample_snapshot):
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(
        sample_snapshot,
        combined_regime_output={"regime": "TRENDING", "trend_strength": 0.8, "volatility": 0.3},
    )
    assert frame.regime_weights.regime == RegimeType.TRENDING


def test_pipeline_with_no_regime_output(sample_snapshot):
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(sample_snapshot, combined_regime_output=None)
    assert frame.regime_weights.regime == RegimeType.UNKNOWN


def test_pipeline_symbol_preserved(sample_snapshot):
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(sample_snapshot)
    assert frame.symbol == "SYNTH"


def test_pipeline_timestamp_preserved(sample_snapshot):
    pipe = OrderBookMetaPipeline()
    frame = pipe.run(sample_snapshot)
    assert frame.timestamp_ms == sample_snapshot.timestamp_ms
