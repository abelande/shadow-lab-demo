"""FFT decomposition with numpy fallback."""
from __future__ import annotations
from typing import List, Tuple


class FFTDecomposer:
    def decompose(self, series: List[float]) -> Tuple[list[float], list[complex]]:
        if not series:
            return [], []
        n = len(series)
        try:
            import numpy as np
            arr = np.array(series, dtype=float)
            fftv = np.fft.rfft(arr)
            freqs = np.fft.rfftfreq(n, d=1.0)
            return freqs.tolist(), [complex(x) for x in fftv]
        except Exception:
            # simple DFT fallback
            import cmath
            freqs: list[float] = []
            coeffs: list[complex] = []
            for k in range(n // 2 + 1):
                s = 0j
                for t, x in enumerate(series):
                    s += x * cmath.exp(-2j * cmath.pi * k * t / n)
                coeffs.append(s)
                freqs.append(k / n)
            return freqs, coeffs
