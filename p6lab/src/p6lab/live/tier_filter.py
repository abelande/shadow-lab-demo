"""Wave 8.5-K — rolling-percentile tier filter + renderer wrapper.

The PercentileTierFilter classifies matches by recent-history rank rather
than absolute probability. The TaggingRenderer wraps any broker-subscriber
renderer so that only matches passing the filter get forwarded.

Composition, not monkey-patching: the wrapped renderer is unaware of the
filter, and the broker treats the wrapper as just another subscriber.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

import numpy as np
import json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TierFilterConfig:
    history_size: int = 10_000
    warmup_samples: int = 1_000
    tier_percentiles: dict[str, float] = field(default_factory=lambda: {
        "A_strict":  0.995,
        "A_relaxed": 0.99,
        "B":         0.975,
    })
    # Set to True for the first run only — emits dir(match) once so you
    # can confirm which attribute carries the model probability.
    debug_attr_probe: bool = False


class PercentileTierFilter:
    """Classify predictions by rolling-percentile rank."""

    def __init__(self, config: TierFilterConfig | None = None) -> None:
        self.config = config or TierFilterConfig()
        self._history: deque[float] = deque(maxlen=self.config.history_size)
        self._sorted_tiers = sorted(
            self.config.tier_percentiles.items(),
            key=lambda kv: -kv[1],     # strictest first
        )
        self._n_observed = 0

    def observe(self, proba: float) -> Optional[str]:
        self._history.append(float(proba))
        self._n_observed += 1
        if len(self._history) < self.config.warmup_samples:
            return None
        history_arr = np.fromiter(self._history, dtype=np.float64)
        for tier_name, percentile in self._sorted_tiers:
            threshold = float(np.quantile(history_arr, percentile))
            if proba >= threshold:
                return tier_name
        return None

    @property
    def is_warm(self) -> bool:
        return len(self._history) >= self.config.warmup_samples

    def current_thresholds(self) -> dict[str, float]:
        if not self.is_warm:
            return {}
        history_arr = np.fromiter(self._history, dtype=np.float64)
        return {
            name: float(np.quantile(history_arr, pct))
            for name, pct in self._sorted_tiers
        }
# Tried-in-order list of attribute names. Wave 9 should refactor matches
# to expose a single canonical `score` property; for Wave 8.5 we discover.
_PROBA_ATTRS = ("ensemble_score", "score", "confidence", "tier_a_proba", "proba")


def _extract_proba(match: Any) -> float:
    """Best-effort probability extraction. Logs once on miss."""
    for attr in _PROBA_ATTRS:
        val = getattr(match, attr, None)
        if val is not None:
            return float(val)
    return 0.0


# In tier_filter.py — replace FilteredRenderer with this:
class TaggingRenderer:
    """Composes a renderer with a PercentileTierFilter — always forwards,
    tags matches passing the filter with their tier in metadata['tier_pct'].
    Also writes a side-car JSONL with (pattern_id, entry_ts_ms, tier_pct)
    so downstream analysis can join with outcome records.
    """
    def __init__(
        self,
        inner: Any,
        tier_filter: PercentileTierFilter,
        tier_log_path: Path | None = None,
    ) -> None:
        self._inner = inner
        self._filter = tier_filter
        self._tier_log_path = tier_log_path
        self._probed = False
        if tier_log_path is not None:
            tier_log_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, match: Any) -> None:
        self._handle(match)

    def on_match(self, match: Any) -> None:
        self._handle(match)

    def _handle(self, match: Any) -> None:
        if (not self._probed) and self._filter.config.debug_attr_probe:
            attrs = sorted(a for a in dir(match) if not a.startswith("_"))
            logger.info("TaggingRenderer: match dir() = %s", attrs)
            self._probed = True

        proba = _extract_proba(match)
        tier = self._filter.observe(proba)

        # Write side-car tier log — one row per match, tier may be None
        if self._tier_log_path is not None:
            import json
            row = {
                "pattern_id": str(getattr(match, "pattern_id", "")),
                "symbol": str(getattr(match, "instrument", "")),
                "entry_ts_ms": int(getattr(match, "match_window_end_ms", 0) or 0),
                "proba": float(proba),
                "tier_pct": tier,
            }
            with open(self._tier_log_path, "a") as f:
                f.write(json.dumps(row) + "\n")

        # Tag metadata for any consumer that reads it (best-effort)
        meta = getattr(match, "metadata", None)
        if isinstance(meta, dict):
            meta["tier_pct"] = tier

        # Always forward
        if callable(self._inner):
            self._inner(match)
        elif hasattr(self._inner, "on_match"):
            self._inner.on_match(match)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
