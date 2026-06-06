"""Enrichment signals for the Cup Flip L2 layer.

Each module provides a stateful tracker that updates per-snapshot and
returns a scalar signal. All are optional supplements to the existing
threshold-based logic — every consumer accepts a default value when
the signal is unavailable.
"""
from .energy_ratio import EnergyRatio
from .kalman_velocity import KalmanVelocity
from .entropy_gate import EntropyGate
from .markov_state import MarkovStateTracker
from .flux_stark import FluxStarkTracker

__all__ = [
    "EnergyRatio",
    "KalmanVelocity",
    "EntropyGate",
    "MarkovStateTracker",
    "FluxStarkTracker",
]
