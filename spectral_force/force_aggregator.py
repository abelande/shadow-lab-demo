"""Aggregate force: Σ[(1/f_k) × E_k × sign(Δv_k)]."""
from __future__ import annotations
from typing import Dict
from ..models import FrequencyBand, BandEnergy, ForceVector


class ForceAggregator:
    # nominal center frequencies (lower => slower/institutional)
    FREQ = {
        FrequencyBand.INSTITUTIONAL: 0.125,
        FrequencyBand.FUND: 0.375,
        FrequencyBand.DAYTRADING: 0.625,
        FrequencyBand.HFT: 0.875,
    }

    def aggregate(self, band_energy: Dict[FrequencyBand, Dict[str, float]], timestamp_ms: int) -> ForceVector:
        bands: list[BandEnergy] = []
        total = 0.0
        dom = None
        dom_e = -1.0
        for b, v in band_energy.items():
            e = float(v.get("energy", 0.0))
            s = int(v.get("sign", 0.0))
            fk = self.FREQ.get(b, 1.0)
            w = (1.0 / max(fk, 1e-6)) * e * s
            total += w
            bands.append(BandEnergy(band=b, energy=e, sign=s, weighted_force=w))
            if e > dom_e:
                dom_e = e
                dom = b

        inst = float(band_energy.get(FrequencyBand.INSTITUTIONAL, {}).get("energy", 0.0))
        total_e = sum(float(x.get("energy", 0.0)) for x in band_energy.values())
        institutional_score = (inst / total_e) if total_e > 0 else 0.0

        return ForceVector(
            total_force=total,
            bands=bands,
            institutional_score=institutional_score,
            dominant_band=dom,
            timestamp_ms=timestamp_ms,
        )
