"""Online activity detector — Wave 9 §H.1.a / A5.

Stateful, per-snapshot CUSUM detector that mirrors the offline
``activity_mask`` in ``p6lab.validation.labelers`` but operates online
(streaming snapshots, no future lookahead).

The model in optimized Strategy A is trained on rows where the activity
mask is True. At inference the engine asks this detector "is the current
snapshot in an active period?" and routes accordingly:

  * ``is_active=True``  → engine emits the model's actual ``predict_proba``
  * ``is_active=False`` → engine emits a base-rate prior (uninformative)

The soft-prior path (rather than hard-skip) is the MVP per build doc
§H.1.a — the same model can be A/B-compared against a hard gate later
without retraining.

Asymmetry vs offline mask
-------------------------
The offline ``activity_mask`` (default ``lookforward_ms=0``) marks rows
*before* CUSUM events — "leading up to detected activity." The online
detector can only mark *after* events because it has no future lookahead.
Both windows are configurable; for end-to-end consistency, set the offline
mask's ``lookforward_ms`` ≥ the online ``lookback_ms``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActivityDetectorConfig:
    """Tunable parameters for ``OnlineActivityDetector``.

    Attributes
    ----------
    cusum_threshold : float
        Two-sided CUSUM firing threshold in price units (NQ tick = 0.25;
        threshold 0.5 ≈ 2 ticks of cumulative drift before an event).
    lookback_ms : int
        How long after a CUSUM event a snapshot is still considered
        "active." Default 60s mirrors the canonical training mask.
    """
    cusum_threshold: float = 0.5
    lookback_ms: int = 60_000


class OnlineActivityDetector:
    """Stateful per-snapshot activity gate.

    Internally maintains a two-sided CUSUM accumulator on price
    first-differences. When either the upper accumulator crosses
    ``+threshold`` or the lower crosses ``-threshold``, an event fires
    and both accumulators reset to zero. After an event, snapshots
    within ``lookback_ms`` of the event timestamp are marked active.

    Usage
    -----
    >>> det = OnlineActivityDetector(ActivityDetectorConfig())
    >>> for snap in stream:
    ...     active = det.update(snap.mid, snap.timestamp_ms)
    ...     if active:
    ...         # engine should run the model on this snapshot
    """

    def __init__(self, config: ActivityDetectorConfig) -> None:
        self.config = config
        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._last_price: float | None = None
        self._last_event_ts: int | None = None
        self._event_count: int = 0

    def update(self, price: float, ts_ms: int) -> bool:
        """Ingest one snapshot's price and return whether currently active.

        First call (no prior price) returns False — CUSUM needs at least
        one prior observation. Subsequent calls accumulate price diffs;
        an event fires whenever an accumulator crosses ±threshold.
        Returns True iff the most recent event was within ``lookback_ms``
        of the supplied timestamp.

        Parameters
        ----------
        price : float
            Current mid price (or any drift-bearing price series).
        ts_ms : int
            Snapshot timestamp in milliseconds.

        Returns
        -------
        bool
            True if the snapshot is in an active window.
        """
        if self._last_price is None:
            self._last_price = float(price)
            return False
        diff = float(price) - self._last_price
        self._last_price = float(price)
        self._s_pos = max(0.0, self._s_pos + diff)
        self._s_neg = min(0.0, self._s_neg + diff)
        if (self._s_pos >= self.config.cusum_threshold
                or self._s_neg <= -self.config.cusum_threshold):
            self._last_event_ts = int(ts_ms)
            self._event_count += 1
            self._s_pos = 0.0
            self._s_neg = 0.0
        return self._is_active_at(int(ts_ms))

    def is_active_at(self, ts_ms: int) -> bool:
        """Public read-only check without ingesting a price.

        Useful when the engine wants to re-query activity state at
        match time without advancing the CUSUM (e.g., the runner
        already advanced it in the per-snapshot loop).
        """
        return self._is_active_at(int(ts_ms))

    def _is_active_at(self, ts_ms: int) -> bool:
        if self._last_event_ts is None:
            return False
        return (ts_ms - self._last_event_ts) <= self.config.lookback_ms

    @property
    def event_count(self) -> int:
        """Number of CUSUM events that have fired since construction."""
        return self._event_count

    def reset(self) -> None:
        """Clear all state — useful for tests and session boundaries."""
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._last_price = None
        self._last_event_ts = None
        self._event_count = 0
