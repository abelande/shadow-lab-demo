"""Compute energy and signed delta per frequency band."""
from __future__ import annotations
from typing import Dict, List
from ..models import FrequencyBand


class EnergyPerBand:
    def compute(
        self,
        bands: Dict[FrequencyBand, list[int]],
        coeffs: List[complex],
        series: List[float],
    ) -> Dict[FrequencyBand, Dict[str, float]]:
        out: Dict[FrequencyBand, Dict[str, float]] = {}
        series_sign = 1.0 if sum(series) > 0 else (-1.0 if sum(series) < 0 else 0.0)

        for b, idxs in bands.items():
            e = 0.0
            for i in idxs:
                if i < len(coeffs):
                    c = coeffs[i]
                    e += (c.real * c.real + c.imag * c.imag)
            out[b] = {"energy": e, "sign": series_sign}
        return out
