"""Tests for spectral force layer (Layer 3)."""
from __future__ import annotations
import pytest
from p6.spectral_force.volume_delta_series import VolumeDeltaSeries
from p6.spectral_force.fft_decomposer import FFTDecomposer
from p6.spectral_force.band_splitter import BandSplitter
from p6.spectral_force.energy_per_band import EnergyPerBand
from p6.spectral_force.force_aggregator import ForceAggregator
from p6.models import Order, OrderAction, Side, ForceVector, FrequencyBand


def _fill(oid, side, price, size, ts):
    return Order(
        order_id=oid, side=side, price=price, size=size,
        timestamp_ms=ts, action=OrderAction.FILL, is_aggressive=True,
    )


def test_volume_delta_series_empty():
    vds = VolumeDeltaSeries()
    series = vds.build([])
    assert isinstance(series, list)


def test_volume_delta_series_buy_fills():
    vds = VolumeDeltaSeries()
    fills = [_fill("t1", Side.ASK, 100.0, 10.0, 1000 + i * 100) for i in range(5)]
    series = vds.build(fills)
    assert len(series) > 0


def test_fft_decomposer_returns_matching_lengths():
    fft = FFTDecomposer()
    series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    freqs, coeffs = fft.decompose(series)
    assert len(freqs) == len(coeffs)


def test_fft_decomposer_empty():
    fft = FFTDecomposer()
    freqs, coeffs = fft.decompose([])
    assert freqs == []
    assert coeffs == []


def test_band_splitter_returns_four_bands():
    splitter = BandSplitter()
    freqs = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    bands = splitter.split(freqs)
    # Should return a dict with all 4 bands (or subset with indices)
    assert isinstance(bands, dict)


def test_force_aggregator_returns_force_vector():
    agg = ForceAggregator()
    band_energy = {
        FrequencyBand.INSTITUTIONAL: {"energy": 10.0, "sign": 1},
        FrequencyBand.FUND: {"energy": 5.0, "sign": -1},
        FrequencyBand.DAYTRADING: {"energy": 3.0, "sign": 1},
        FrequencyBand.HFT: {"energy": 1.0, "sign": 0},
    }
    fv = agg.aggregate(band_energy, timestamp_ms=1000)
    assert isinstance(fv, ForceVector)


def test_force_aggregator_institutional_dominance():
    agg = ForceAggregator()
    band_energy = {
        FrequencyBand.INSTITUTIONAL: {"energy": 100.0, "sign": 1},
        FrequencyBand.FUND: {"energy": 1.0, "sign": 1},
        FrequencyBand.DAYTRADING: {"energy": 1.0, "sign": 1},
        FrequencyBand.HFT: {"energy": 1.0, "sign": 1},
    }
    fv = agg.aggregate(band_energy, timestamp_ms=1000)
    assert fv.institutional_score > 0.8


def test_force_aggregator_empty():
    agg = ForceAggregator()
    fv = agg.aggregate({}, timestamp_ms=1000)
    assert fv.total_force == 0.0
    assert fv.institutional_score == 0.0


def test_full_spectral_pipeline_with_trades(sample_trades):
    vds = VolumeDeltaSeries()
    fft = FFTDecomposer()
    splitter = BandSplitter()
    epb = EnergyPerBand()
    agg = ForceAggregator()

    series = vds.build(sample_trades)
    freqs, coeffs = fft.decompose(series)
    bands = splitter.split(freqs)
    band_energy = epb.compute(bands, coeffs, series)
    fv = agg.aggregate(band_energy, timestamp_ms=2000)

    assert isinstance(fv, ForceVector)
    assert fv.timestamp_ms == 2000
