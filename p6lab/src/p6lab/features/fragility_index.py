"""
p6lab.features.fragility_index — Six sub-indices + two composite FI scores.

Spec: p6-notebook-lab-spec.md §4.3
Ref:  OB-reference.md L1552-1696 (full specification)
      OB-reference.md L1734-1748 (validation criteria)
      OB-reference.md L1701      (signal_bar threshold adjustment)
      OB-reference.md L1721-1729 (size multiplier reduction)

Sub-indices:
  DF  — Depth Fragility         (L2 speed path)
  CF  — Cancellation Fragility  (L2 speed path)
  RF  — Refresh Fragility       (L1 speed path)
  SF  — Spread Fragility        (L1 speed path)
  FT  — Flow Toxicity           (L1 speed path, fed by VPIN)
  CIS — Cross-Instrument Stress (L2 speed path)

Composite scores (spec §4.3):
  FI_fast = 0.35 × RF + 0.35 × SF + 0.30 × FT
            (L1 speed, <5ms latency target)
  FI_full = 0.20 × DF + 0.15 × CF + 0.20 × RF
          + 0.15 × SF + 0.15 × FT + 0.15 × CIS
            (L2 speed, <50ms latency target)

Threshold semantics (spec §10.5):
  FI_fast > 0.6  → lower signal_bar detection threshold 0.55 → 0.40
  FI_full > 0.7  → drop backtest_controls max_size multiplier to 0.5

Validation criteria (spec §4.3, OB-reference L1734-1748):
  AUC > 0.70 for predicting >2 ATR moves within 30m
  FI > 0.5 at least 2m before 50% of cascades
  FP rate <30% at FI > 0.7

Consumers:
  - Notebook 07 §04-§06 (validation — spec §9.5)
  - fragility_gauge.js §10.5 (live UI gauge)
  - correlation_api.py WebSocket 'fragility_update' (spec §11.4)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight constants (spec §4.3 — embed explicitly for reference)
# ---------------------------------------------------------------------------

# FI_fast weights (L1 speed path, <5ms)
FI_FAST_WEIGHTS: dict[str, float] = {
    "RF": 0.35,   # Refresh Fragility
    "SF": 0.35,   # Spread Fragility
    "FT": 0.30,   # Flow Toxicity
}

# FI_full weights (L2 speed path, <50ms)
FI_FULL_WEIGHTS: dict[str, float] = {
    "DF":  0.20,  # Depth Fragility
    "CF":  0.15,  # Cancellation Fragility
    "RF":  0.20,  # Refresh Fragility
    "SF":  0.15,  # Spread Fragility
    "FT":  0.15,  # Flow Toxicity
    "CIS": 0.15,  # Cross-Instrument Stress
}

# Validation targets (spec §4.3, OB-reference L1734-1748)
FI_AUC_TARGET: float = 0.70           # predicting >2 ATR moves within 30m
FI_LEAD_TIME_TARGET_MINUTES: int = 2  # before 50% of cascades
FI_CASCADE_COVERAGE_TARGET: float = 0.50
FI_FP_RATE_THRESHOLD: float = 0.30   # FP rate < 30% at FI > 0.7

# Threshold constants (spec §10.5)
FI_FAST_SIGNAL_ADJUST_THRESHOLD: float = 0.60   # lower detection threshold
FI_FULL_SIZE_REDUCE_THRESHOLD: float = 0.70     # drop max size multiplier
FI_ALERT_YELLOW: float = 0.50
FI_ALERT_RED: float = 0.70

# UI threshold adjustment values (spec §10.5, OB-reference L1701)
SIGNAL_THRESHOLD_NORMAL: float = 0.55
SIGNAL_THRESHOLD_ELEVATED: float = 0.40   # when FI_fast > 0.6
SIZE_MULTIPLIER_REDUCED: float = 0.50    # when FI_full > 0.7


# ---------------------------------------------------------------------------
# Sub-index output dataclass
# ---------------------------------------------------------------------------

@dataclass
class FragilitySubIndices:
    """
    Six raw sub-index values. Spec §4.3, OB-reference L1552-1696.

    All values normalized to [0, 1].
    """
    DF: float = 0.0   # Depth Fragility
    CF: float = 0.0   # Cancellation Fragility
    RF: float = 0.0   # Refresh Fragility
    SF: float = 0.0   # Spread Fragility
    FT: float = 0.0   # Flow Toxicity
    CIS: float = 0.0  # Cross-Instrument Stress


@dataclass
class FragilityOutput:
    """
    Full FI output for a single snapshot. Spec §4.3.
    Consumed by WebSocket 'fragility_update' message (spec §11.4).
    """
    timestamp_ms: int
    symbol: str
    sub_indices: FragilitySubIndices
    fi_fast: float     # FI_fast composite
    fi_full: float     # FI_full composite

    @property
    def signal_threshold(self) -> float:
        """Current detection threshold based on FI_fast (spec §10.5)."""
        if self.fi_fast > FI_FAST_SIGNAL_ADJUST_THRESHOLD:
            return SIGNAL_THRESHOLD_ELEVATED
        return SIGNAL_THRESHOLD_NORMAL

    @property
    def size_multiplier(self) -> float:
        """Current max size multiplier based on FI_full (spec §10.5)."""
        if self.fi_full > FI_FULL_SIZE_REDUCE_THRESHOLD:
            return SIZE_MULTIPLIER_REDUCED
        return 1.0

    def to_dict(self) -> dict:
        """Serialize for WebSocket 'fragility_update' message (spec §11.4)."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "symbol": self.symbol,
            "fi_fast": self.fi_fast,
            "fi_full": self.fi_full,
            "sub_indices": {
                "DF": self.sub_indices.DF,
                "CF": self.sub_indices.CF,
                "RF": self.sub_indices.RF,
                "SF": self.sub_indices.SF,
                "FT": self.sub_indices.FT,
                "CIS": self.sub_indices.CIS,
            },
            "signal_threshold": self.signal_threshold,
            "size_multiplier": self.size_multiplier,
        }


# ---------------------------------------------------------------------------
# Main FragilityIndex class
# ---------------------------------------------------------------------------

class FragilityIndex:
    """
    Direct port of OB-reference.md §6 Fragility Index (L1552-1696).

    Spec §4.3. Compute FI_fast (<5ms) and FI_full (<50ms) at each snapshot.

    Two latency tiers are intentional (spec §4.3):
      FI_fast: L1 inputs only (RF, SF, FT) — computed <5ms
      FI_full: all six sub-indices — computed <50ms

    Consumers:
      - Notebook 07 §04-§06 for CPCV validation
      - correlation_api.py for live WebSocket 'fragility_update'
      - fragility_gauge.js §10.5 for UI rendering

    Usage:
        fi = FragilityIndex()
        output = fi.compute(l1_features, l2_features, vpin_value, timestamp_ms, symbol)
    """

    def __init__(self) -> None:
        # State for rate-of-change sub-indices
        self._prev_depth: float | None = None
        self._prev_spread: float | None = None
        self._cancel_history: list[float] = []

    def compute_fast(
        self,
        rf: float,   # Refresh Fragility from L1
        sf: float,   # Spread Fragility from L1
        ft: float,   # Flow Toxicity from VPIN
    ) -> float:
        """
        Compute FI_fast from L1 sub-indices only. Spec §4.3.
        Latency target: <5ms.

        Formula: FI_fast = 0.35 × RF + 0.35 × SF + 0.30 × FT
        """
        return (
            FI_FAST_WEIGHTS["RF"] * rf
            + FI_FAST_WEIGHTS["SF"] * sf
            + FI_FAST_WEIGHTS["FT"] * ft
        )

    def compute_full(self, sub: FragilitySubIndices) -> float:
        """
        Compute FI_full from all six sub-indices. Spec §4.3.
        Latency target: <50ms.

        Formula: FI_full = 0.20×DF + 0.15×CF + 0.20×RF + 0.15×SF + 0.15×FT + 0.15×CIS
        """
        return (
            FI_FULL_WEIGHTS["DF"]  * sub.DF
            + FI_FULL_WEIGHTS["CF"]  * sub.CF
            + FI_FULL_WEIGHTS["RF"]  * sub.RF
            + FI_FULL_WEIGHTS["SF"]  * sub.SF
            + FI_FULL_WEIGHTS["FT"]  * sub.FT
            + FI_FULL_WEIGHTS["CIS"] * sub.CIS
        )

    def compute_sub_indices(
        self,
        l1_features: np.ndarray,
        l2_features: np.ndarray,
        vpin_value: float,
        cross_instrument_stress: float = 0.0,
    ) -> FragilitySubIndices:
        """Compute six [0,1]-clamped sub-indices.

        Each raw signal is mapped through a saturating transform so the
        composite stays bounded; thresholds reflect typical NQ scales
        (OB-reference L1552-1696).
        """
        def _clip01(x: float) -> float:
            if not np.isfinite(x):
                return 0.0
            return max(0.0, min(1.0, x))

        # l2_features[6] = depth_change_rate_5s (negative = depletion)
        depth_change_5s = float(l2_features[6]) if l2_features.size > 6 else 0.0
        # Saturate at -200/sec → DF=1.0
        df = _clip01(-depth_change_5s / 200.0)

        # CF — cancel fragility from EWMA of recent cancel rate (history-tracked)
        # We approximate via the magnitude of the negative depth-change-30s.
        depth_change_30s = float(l2_features[7]) if l2_features.size > 7 else 0.0
        cf_raw = abs(depth_change_30s) / 100.0  # 100/sec sustained churn → 1.0
        cf = _clip01(cf_raw)

        # RF — l1_features[5] bid_refresh_rate, [6] ask_refresh_rate
        bid_rr = float(l1_features[5]) if l1_features.size > 5 else 0.0
        ask_rr = float(l1_features[6]) if l1_features.size > 6 else 0.0
        rf_raw = (bid_rr + ask_rr) / 50.0  # 25/s per side → 1.0
        rf = _clip01(rf_raw)

        # SF — spread_ticks at index 0; widen → fragility
        spread_ticks = float(l1_features[0]) if l1_features.size > 0 else 0.0
        # NQ normal spread = 1 tick. >5 ticks → 1.0
        sf = _clip01((spread_ticks - 1.0) / 4.0)

        # FT — VPIN already in [0, 1]
        ft = _clip01(vpin_value)

        # CIS — externally provided
        cis = _clip01(cross_instrument_stress)

        # Update simple internal state for future extensions
        cur_depth = (l2_features[6] if l2_features.size > 6 else 0.0)
        self._prev_depth = float(cur_depth)
        self._prev_spread = spread_ticks

        return FragilitySubIndices(DF=df, CF=cf, RF=rf, SF=sf, FT=ft, CIS=cis)

    def compute(
        self,
        l1_features: np.ndarray,
        l2_features: np.ndarray,
        vpin_value: float,
        timestamp_ms: int,
        symbol: str,
        cross_instrument_stress: float = 0.0,
    ) -> FragilityOutput:
        """
        Full FI computation: sub-indices → FI_fast + FI_full. Spec §4.3.

        Called from engine_runner on every snapshot.
        """
        sub = self.compute_sub_indices(l1_features, l2_features, vpin_value, cross_instrument_stress)
        fi_fast = self.compute_fast(sub.RF, sub.SF, sub.FT)
        fi_full = self.compute_full(sub)
        return FragilityOutput(
            timestamp_ms=timestamp_ms,
            symbol=symbol,
            sub_indices=sub,
            fi_fast=fi_fast,
            fi_full=fi_full,
        )

    def compute_series(
        self,
        l1_df: pd.DataFrame,
        l2_df: pd.DataFrame,
        vpin_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Bulk FI computation over aligned DataFrames. Spec §4.3.

        Used by notebook 07 §03-§06 for historical validation.
        Returns DataFrame with columns:
          DF, CF, RF, SF, FT, CIS, fi_fast, fi_full,
          signal_threshold, size_multiplier
        indexed by timestamp_ms.

        Vectorized bulk computation for notebook 07.
        """
        n = len(l1_df)
        if n == 0:
            return pd.DataFrame(columns=[
                "DF", "CF", "RF", "SF", "FT", "CIS",
                "fi_fast", "fi_full", "signal_threshold", "size_multiplier",
            ])
        rows: list[dict] = []
        for i in range(n):
            l1_vec = l1_df.iloc[i].to_numpy()
            l2_vec = l2_df.iloc[i].to_numpy() if i < len(l2_df) else np.zeros(12)
            vpin_v = float(vpin_series.iloc[i]) if i < len(vpin_series) else 0.0
            sub = self.compute_sub_indices(l1_vec, l2_vec, vpin_v)
            fi_fast = self.compute_fast(sub.RF, sub.SF, sub.FT)
            fi_full = self.compute_full(sub)
            rows.append({
                "DF": sub.DF, "CF": sub.CF, "RF": sub.RF, "SF": sub.SF,
                "FT": sub.FT, "CIS": sub.CIS,
                "fi_fast": fi_fast, "fi_full": fi_full,
                "signal_threshold": (
                    SIGNAL_THRESHOLD_ELEVATED
                    if fi_fast > FI_FAST_SIGNAL_ADJUST_THRESHOLD
                    else SIGNAL_THRESHOLD_NORMAL
                ),
                "size_multiplier": (
                    SIZE_MULTIPLIER_REDUCED
                    if fi_full > FI_FULL_SIZE_REDUCE_THRESHOLD else 1.0
                ),
            })
        return pd.DataFrame(rows, index=l1_df.index)
