"""Institutional participation score from low-frequency dominance."""
from __future__ import annotations
from ..models import ForceVector, FrequencyBand


class InstitutionalScore:
    def score(self, fv: ForceVector) -> float:
        low = 0.0
        total = 0.0
        for b in fv.bands:
            total += b.energy
            if b.band in (FrequencyBand.INSTITUTIONAL, FrequencyBand.FUND):
                low += b.energy
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, low / total))
