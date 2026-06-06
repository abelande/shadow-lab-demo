"""Split spectrum into 4 bands: institutional/fund/daytrading/HFT."""
from __future__ import annotations
from typing import Dict, List
from ..models import FrequencyBand


class BandSplitter:
    """Map frequency bins to semantic bands."""

    def split(self, freqs: List[float]) -> Dict[FrequencyBand, list[int]]:
        bands = {
            FrequencyBand.INSTITUTIONAL: [],
            FrequencyBand.FUND: [],
            FrequencyBand.DAYTRADING: [],
            FrequencyBand.HFT: [],
        }
        if not freqs:
            return bands

        # Quantile-based split to stay adaptive regardless of series length
        nz = [f for f in freqs if f > 0]
        if not nz:
            return bands
        fmax = max(nz)
        q1 = 0.25 * fmax
        q2 = 0.50 * fmax
        q3 = 0.75 * fmax

        for i, f in enumerate(freqs):
            if f == 0:
                bands[FrequencyBand.INSTITUTIONAL].append(i)
            elif f <= q1:
                bands[FrequencyBand.INSTITUTIONAL].append(i)
            elif f <= q2:
                bands[FrequencyBand.FUND].append(i)
            elif f <= q3:
                bands[FrequencyBand.DAYTRADING].append(i)
            else:
                bands[FrequencyBand.HFT].append(i)
        return bands
