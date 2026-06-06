"""
p6lab.ingestion.instrument_normalizer — Cross-instrument normalization.

Spec: p6-notebook-lab-spec.md §3.3
Ref:  OB-reference.md L453-468

Normalization scheme:
  - Depth as ratio of 20-day median
  - Spread in basis points
  - ATR-normalized price moves
  - Self-normalizing book shape vector
  - VIX-tagged regime buckets (low <15, normal 15-25, elevated 25-35, high >35)

Per-instrument backfill tables cached in:
  artifacts/p6lab/normalization/{symbol}_median_depth.parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VIX regime constants (spec §3.3, OB-reference L464-466)
# ---------------------------------------------------------------------------

class VIXRegime(Enum):
    """
    VIX-tagged regime buckets for template selection.

    Spec §3.3, §7.3 — OB-reference.md L464-466.
    Used by:
      - InstrumentNormalizer.classify_regime()
      - RegimeConditioner (spec §7.3) for per-bucket template selection
      - library.yaml pattern entries (regime_specific: true)
    """
    LOW = "low"           # VIX < 15
    NORMAL = "normal"     # 15 <= VIX < 25
    ELEVATED = "elevated" # 25 <= VIX < 35
    HIGH = "high"         # VIX >= 35

    @classmethod
    def from_vix(cls, vix_value: float) -> "VIXRegime":
        """Classify a VIX reading into a regime bucket. Spec §3.3."""
        if vix_value < 15.0:
            return cls.LOW
        elif vix_value < 25.0:
            return cls.NORMAL
        elif vix_value < 35.0:
            return cls.ELEVATED
        else:
            return cls.HIGH


# VIX threshold constants (spec §3.3)
VIX_LOW_THRESHOLD: float = 15.0
VIX_NORMAL_THRESHOLD: float = 25.0
VIX_ELEVATED_THRESHOLD: float = 35.0

# Normalization lookback (spec §3.3)
MEDIAN_DEPTH_LOOKBACK_DAYS: int = 20


# ---------------------------------------------------------------------------
# Normalizer config
# ---------------------------------------------------------------------------

@dataclass
class NormalizerConfig:
    """
    Per-instrument normalization config.

    Spec §3.3, OB-reference L453-468.
    Populated from artifacts/p6lab/normalization/{symbol}_median_depth.parquet
    after a calibration run (typically first run of notebook 03).
    """
    symbol: str
    tick_size: float                  # price increment
    atr_20d: float = 1.0             # 20-day ATR for price-move normalization
    median_depth_20d: float = 1.0    # 20-day median total depth, for ratio
    cache_dir: Path = Path("artifacts/p6lab/normalization")


# ---------------------------------------------------------------------------
# Main normalizer class
# ---------------------------------------------------------------------------

class InstrumentNormalizer:
    """
    Normalizes raw order-book quantities for cross-instrument ML.

    Spec §3.3 — OB-reference.md L453-468.

    Used by:
      - TripleViewEmitter (spec §3.1) before writing parquet
      - EventShapeExtractor in miner.py (spec §5.2)
      - CorrelationEngine at match time (spec §7.1)

    All normalization is stateless after initialization — safe for
    concurrent calls from the engine runner.
    """

    def __init__(self, config: NormalizerConfig) -> None:
        self.config = config
        self._loaded = False

    @classmethod
    def from_cache(cls, symbol: str, cache_dir: Path) -> "InstrumentNormalizer":
        """Load per-instrument calibration parquet, or return default config if missing."""
        path = cache_dir / f"{symbol}_median_depth.parquet"
        if not path.exists():
            logger.warning("Calibration cache missing for %s (%s) — using defaults",
                           symbol, path)
            return cls(NormalizerConfig(symbol=symbol, tick_size=0.25, cache_dir=cache_dir))
        df = pd.read_parquet(path)
        row = df.iloc[-1]
        cfg = NormalizerConfig(
            symbol=symbol,
            tick_size=float(row.get("tick_size", 0.25)),
            atr_20d=float(row.get("atr_20d", 1.0)),
            median_depth_20d=float(row.get("median_depth_20d", 1.0)),
            cache_dir=cache_dir,
        )
        inst = cls(cfg)
        inst._loaded = True
        return inst

    def normalize_depth(self, raw_depth: float) -> float:
        """Depth as ratio of 20-day median (spec §3.3)."""
        if self.config.median_depth_20d <= 0:
            return 0.0
        return float(raw_depth) / self.config.median_depth_20d

    def spread_to_bps(self, best_bid: float, best_ask: float) -> float:
        """Spread in basis points."""
        if best_bid <= 0 or best_ask <= 0:
            return 0.0
        mid = 0.5 * (best_ask + best_bid)
        if mid <= 0:
            return 0.0
        return (best_ask - best_bid) / mid * 10_000.0

    def normalize_price_move(self, raw_move: float) -> float:
        """
        ATR-normalized price move. Spec §3.3.

        Formula: raw_move / self.config.atr_20d
        """
        if self.config.atr_20d <= 0:
            return 0.0
        return raw_move / self.config.atr_20d

    def normalize_book_shape_vector(self, raw_vector: np.ndarray) -> np.ndarray:
        """Self-normalizing 40-dim book shape: each side sums to 1.0."""
        v = np.asarray(raw_vector, dtype=np.float64).copy()
        if v.size == 0:
            return v
        # Treat first half as bid side, second half as ask side
        half = v.size // 2
        if half > 0:
            bid = v[:half]
            ask = v[half:]
            bid_sum = bid.sum()
            ask_sum = ask.sum()
            if bid_sum > 0:
                v[:half] = bid / bid_sum
            if ask_sum > 0:
                v[half:] = ask / ask_sum
        return v

    def classify_regime(self, vix_value: float) -> VIXRegime:
        """
        Classify VIX into regime bucket. Spec §3.3, §7.3.

        Used for per-instrument, per-VIX-bucket template selection.
        Returns VIXRegime enum value.
        """
        return VIXRegime.from_vix(vix_value)

    def update_calibration(self, depth_history: pd.Series, atr_history: pd.Series) -> None:
        """Recompute 20-day median depth and ATR; persist to parquet cache."""
        if len(depth_history) == 0 or len(atr_history) == 0:
            logger.warning("update_calibration skipped: empty input series")
            return
        # Tail 20 observations (caller is responsible for daily aggregation)
        median_depth = float(depth_history.tail(MEDIAN_DEPTH_LOOKBACK_DAYS).median())
        atr = float(atr_history.tail(MEDIAN_DEPTH_LOOKBACK_DAYS).mean())
        if median_depth > 0:
            self.config.median_depth_20d = median_depth
        if atr > 0:
            self.config.atr_20d = atr
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.cache_dir / f"{self.config.symbol}_median_depth.parquet"
        pd.DataFrame([{
            "symbol": self.config.symbol,
            "tick_size": self.config.tick_size,
            "median_depth_20d": self.config.median_depth_20d,
            "atr_20d": self.config.atr_20d,
            "updated_ts_ms": int(pd.Timestamp.now().timestamp() * 1000),
        }]).to_parquet(path, index=False)
        logger.info("Wrote calibration cache: %s (depth=%.2f atr=%.4f)",
                    path, self.config.median_depth_20d, self.config.atr_20d)
