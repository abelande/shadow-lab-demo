"""
p6lab.features.spectral_force — Wave 6 Phase 6A

Port of p5-dsp-signal-lab's per-band ``E_k(t)`` energy trajectories,
wrapped so callers can request either:

  1. a full rolling series per band (for notebook visualization), or
  2. a 4-scalar snapshot suitable for the live feature matrix
     (one scalar per institutional / fund / daytrading / HFT band +
     a total-energy scalar).

The underlying FFT / band-split / per-band-energy primitives already
live in ``p6-v2/spectral_force/``. This module is a thin-layer wrapper
that (a) handles import-fallback across repo layouts, (b) keeps a
timestamped ring buffer, and (c) exposes the canonical feature names.

Exported:
    SPECTRAL_FORCE_FEATURE_NAMES   tuple[str, ...]
    SpectralForceState             dataclass
    update_spectral_force(state, ts_ms, volume_delta_series)
    snapshot_spectral_force_features(state) → dict
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

logger = logging.getLogger(__name__)


BUFFER_LEN = 1024


SPECTRAL_FORCE_FEATURE_NAMES: tuple[str, ...] = (
    "force_band_energy_institutional",
    "force_band_energy_fund",
    "force_band_energy_daytrading",
    "force_band_energy_hft",
    "force_total_energy",
)


@dataclass
class SpectralForceState:
    """Rolling (ts, per-band-energy) history. ``history`` stores the last
    ``BUFFER_LEN`` frames as dict[band_name -> float]; notebook callers
    can iterate it directly for E_k(t) plots."""
    history: Deque[tuple[int, dict[str, float]]] = field(
        default_factory=lambda: deque(maxlen=BUFFER_LEN)
    )
    last_energies: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.history.clear()
        self.last_energies = {}


def update_spectral_force(
    state: SpectralForceState,
    *,
    ts_ms: int,
    volume_delta_series: list[float],
) -> None:
    """Recompute per-band energies from a ``volume_delta_series`` snapshot.

    ``volume_delta_series`` is the usual p6v2 input — a list of floats
    representing the net buy-minus-sell volume per bar. A short or empty
    input yields an empty energy dict (but still appends to history so
    callers can track the full cadence).
    """
    energies = _compute_band_energies(volume_delta_series)
    state.last_energies = energies
    state.history.append((int(ts_ms), energies))


def snapshot_spectral_force_features(
    state: SpectralForceState,
) -> dict[str, float]:
    """Emit the 4 per-band scalars + total energy."""
    latest = state.last_energies or {}
    inst = float(latest.get("INSTITUTIONAL", 0.0))
    fund = float(latest.get("FUND", 0.0))
    day = float(latest.get("DAYTRADING", 0.0))
    hft = float(latest.get("HFT", 0.0))
    total = inst + fund + day + hft
    return {
        "force_band_energy_institutional": inst,
        "force_band_energy_fund": fund,
        "force_band_energy_daytrading": day,
        "force_band_energy_hft": hft,
        "force_total_energy": float(total),
    }


# ---------------------------------------------------------------------------
# Core: reuse the p6-v2 primitives
# ---------------------------------------------------------------------------


def _compute_band_energies(volume_delta_series: list[float]) -> dict[str, float]:
    """FFT → band-split → per-band energy. Returns ``{band_name: float}``.

    Falls back to an empty dict when the p6-v2 modules aren't importable —
    keeps p6lab unit tests runnable without the sibling repo on
    ``PYTHONPATH``.
    """
    if not volume_delta_series:
        return {}

    components = _load_p6v2_components()
    if components is None:
        return {}

    FFTDecomposer, BandSplitter, EnergyPerBand = components
    series = list(volume_delta_series)
    freqs, coeffs = FFTDecomposer().decompose(series)
    bands = BandSplitter().split(freqs)
    energy_by_band = EnergyPerBand().compute(bands, coeffs, series)
    return {
        _band_name(band): float(val.get("energy", 0.0))
        for band, val in energy_by_band.items()
    }


def _load_p6v2_components() -> tuple[Any, Any, Any] | None:
    """Import p6-v2 spectral_force primitives. Handles two layouts:
       * installed-package layout (``p6v2`` on ``sys.path``)
       * repo-relative layout (``../..`` off this file)"""
    try:
        from p6v2.spectral_force.fft_decomposer import FFTDecomposer
        from p6v2.spectral_force.band_splitter import BandSplitter
        from p6v2.spectral_force.energy_per_band import EnergyPerBand
        return FFTDecomposer, BandSplitter, EnergyPerBand
    except Exception:
        logger.debug("p6v2 import failed; spectral_force will emit zeros")
        return None


def _band_name(band: Any) -> str:
    """Normalize either a FrequencyBand enum or a plain string to upper."""
    if hasattr(band, "name"):
        return str(band.name).upper()
    if hasattr(band, "value"):
        return str(band.value).upper()
    return str(band).upper()
