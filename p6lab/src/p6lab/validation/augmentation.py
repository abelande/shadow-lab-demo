"""
p6lab.validation.augmentation
=============================
Synthetic data augmentation — §8.2 of the P6 Lab Spec.

Transforms (OB-reference §L1336-L1537)
--------------------------------------
1. Time stretch/compress
2. Depth scaling ±20%
3. Volatility shift ±1 ATR
4. Cross-instrument transfer
5. Phase duration jitter

Each augmented sample must be tagged with ``augmentation_method``.
Notebook 07 enforces: augmented training must beat original-only baseline
(OB-reference §L1504-L1509).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEPTH_SCALE_MIN: float = 0.8
DEPTH_SCALE_MAX: float = 1.2
VOL_SHIFT_ATR: float = 1.0
PHASE_JITTER_MIN: float = 0.8
PHASE_JITTER_MAX: float = 1.2


AugMethod = Literal[
    "original",
    "time_stretch",
    "time_compress",
    "depth_scale",
    "volatility_shift",
    "cross_instrument_transfer",
    "phase_duration_jitter",
]


@dataclass
class AugmentedSample:
    """Container for one augmented sample."""

    features: pd.DataFrame
    label: int | float | str
    augmentation_method: AugMethod
    source_instrument: str
    target_instrument: str | None = None


_DEPTH_COL_HINTS = ("size", "depth", "volume", "bid_sz", "ask_sz", "imbalance")
_VOLATILITY_COL_HINTS = ("velocity", "spread", "return", "tick", "compression")


def _matches_any(col: str, hints: tuple[str, ...]) -> bool:
    return any(h in col.lower() for h in hints)


class AugmentationEngine:
    """Apply synthetic augmentation transforms to sequence features."""

    def __init__(self, random_state: int = 42) -> None:
        self.rng = np.random.default_rng(random_state)

    def time_stretch_compress(self, features: pd.DataFrame, factor: float) -> pd.DataFrame:
        """Resample sequence by ``factor`` (>1 stretches, <1 compresses).

        Linear interpolation per column. Index becomes a fresh integer range.
        """
        if factor <= 0:
            raise ValueError(f"factor must be > 0, got {factor}")
        n = len(features)
        if n < 2:
            return features.copy()
        new_n = max(1, int(round(n * factor)))
        old_x = np.linspace(0.0, 1.0, n)
        new_x = np.linspace(0.0, 1.0, new_n)
        out = {}
        for col in features.columns:
            try:
                vals = pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=float)
                out[col] = np.interp(new_x, old_x, vals)
            except Exception:
                # non-numeric — repeat to match new length
                out[col] = features[col].iloc[
                    np.minimum((np.arange(new_n) * n / new_n).astype(int), n - 1)
                ].to_numpy()
        return pd.DataFrame(out)

    def depth_scale(self, features: pd.DataFrame, scale: float) -> pd.DataFrame:
        """Multiply depth-related columns by ``scale``."""
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        out = features.copy()
        for col in out.columns:
            if _matches_any(col, _DEPTH_COL_HINTS):
                out[col] = pd.to_numeric(out[col], errors="coerce") * scale
        return out

    def volatility_shift(
        self, features: pd.DataFrame, atr_shift: float = VOL_SHIFT_ATR,
    ) -> pd.DataFrame:
        """Add ATR-scaled gaussian noise to volatility-sensitive columns."""
        out = features.copy()
        for col in out.columns:
            if _matches_any(col, _VOLATILITY_COL_HINTS):
                col_std = float(pd.to_numeric(out[col], errors="coerce").std() or 1.0)
                noise = self.rng.normal(0.0, atr_shift * col_std, size=len(out))
                out[col] = pd.to_numeric(out[col], errors="coerce") + noise
        return out

    def cross_instrument_transfer(
        self, features: pd.DataFrame, source_symbol: str, target_symbol: str,
    ) -> pd.DataFrame:
        """Re-scale depth columns from source to target instrument units.

        Without an actual normalizer this simulates the transfer by
        applying a fixed per-symbol scale (NQ→ES ratio ≈ 0.5, etc.).
        Real callers should inject their InstrumentNormalizer.
        """
        if source_symbol == target_symbol:
            return features.copy()
        # Heuristic ratio table — replace with InstrumentNormalizer in production.
        ratios = {("NQ", "ES"): 0.5, ("ES", "NQ"): 2.0, ("NQ", "RTY"): 0.3, ("ES", "RTY"): 0.6}
        scale = ratios.get((source_symbol, target_symbol), 1.0)
        return self.depth_scale(features, scale)

    def phase_duration_jitter(self, features: pd.DataFrame, jitter: float) -> pd.DataFrame:
        """Warp local segments by ``jitter`` factor (preserves overall length)."""
        if jitter <= 0:
            raise ValueError(f"jitter must be > 0, got {jitter}")
        n = len(features)
        if n < 4:
            return features.copy()
        # Split into 4 segments, jitter each, then resample back to n
        segs = np.array_split(np.arange(n), 4)
        new_segments = []
        lo, hi = sorted([1.0 / jitter, jitter])
        for seg in segs:
            seg_factor = float(self.rng.uniform(lo, hi))
            sub = features.iloc[seg]
            new_segments.append(self.time_stretch_compress(sub, seg_factor))
        joined = pd.concat(new_segments, ignore_index=True)
        # Resample back to original length
        return self.time_stretch_compress(joined, n / max(1, len(joined)))

    def generate(
        self,
        features: pd.DataFrame,
        label: int | float | str,
        instrument: str,
        n_samples: int = 10,
    ) -> list[AugmentedSample]:
        """Mix of augmentation methods. First sample is the original."""
        out: list[AugmentedSample] = [AugmentedSample(
            features=features.copy(), label=label, augmentation_method="original",
            source_instrument=instrument,
        )]
        methods: list[AugMethod] = [
            "time_stretch", "time_compress", "depth_scale",
            "volatility_shift", "phase_duration_jitter",
        ]
        for i in range(max(0, n_samples - 1)):
            method = methods[i % len(methods)]
            if method == "time_stretch":
                aug = self.time_stretch_compress(features, float(self.rng.uniform(1.05, 1.30)))
            elif method == "time_compress":
                aug = self.time_stretch_compress(features, float(self.rng.uniform(0.75, 0.95)))
            elif method == "depth_scale":
                aug = self.depth_scale(features, float(self.rng.uniform(DEPTH_SCALE_MIN, DEPTH_SCALE_MAX)))
            elif method == "volatility_shift":
                aug = self.volatility_shift(features, atr_shift=float(self.rng.uniform(0.5, 1.5)))
            elif method == "phase_duration_jitter":
                aug = self.phase_duration_jitter(features, float(self.rng.uniform(PHASE_JITTER_MIN, PHASE_JITTER_MAX)))
            else:
                aug = features.copy()
            out.append(AugmentedSample(
                features=aug, label=label, augmentation_method=method,
                source_instrument=instrument,
            ))
        return out
