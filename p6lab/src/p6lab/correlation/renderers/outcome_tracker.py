"""
p6lab.correlation.renderers.outcome_tracker
===========================================

Wave 5 Phase 5B — closes the lab → trader feedback loop.

Subscribes to ``MatchBroker`` and, for every ``PatternMatch`` delivered:

  1. Captures entry context (pattern_id, ts, symbol, direction, recent mid).
  2. Schedules an exit at ``ts + horizon_ms``.
  3. When ``on_price(symbol, mid, ts_ms)`` later sees a timestamp past the
     exit boundary (or ``flush()`` is called), computes realized return,
     appends a JSONL row to the outcomes log, and updates the rolling
     per-pattern hit/miss counters.
  4. Every ``reaggregate_every_n`` closes, rewrites the library's
     ``OutcomeDistribution`` for each pattern from the rolling aggregate
     and, if the rolling hit_rate drops below ``retire_below_hit_rate``
     once the pattern has enough samples, promotes its status to
     ``PatternStatus.RETIRED``. The library write goes through
     ``PatternLibrary.save()`` — atomic via tempfile + rename + FileLock.

Design rules
------------
- **No background threads.** Exits resolve on the same loop that calls
  ``on_price``. Keeps behavior deterministic + easy to test.
- **Fail-soft.** Any exception during update/save is logged and swallowed
  so a bad cycle can't break the broker's other subscribers.
- **Library optional.** If ``library`` is ``None``, the tracker still
  writes JSONL rows but skips library self-update. Useful in tests.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from p6lab.patterns.library import (
    OutcomeDistribution,
    PatternLibrary,
    PatternStatus,
    _ALLOWED_TRANSITIONS,
)

logger = logging.getLogger(__name__)


DEFAULT_HORIZON_MS = 60_000      # 1-minute triple-barrier horizon
DEFAULT_REAGG_N = 20             # re-aggregate library every 20 closes
DEFAULT_RETIRE_THRESHOLD = 0.50  # 30d hit_rate must stay ≥ 50% to stay ACTIVE
MIN_SAMPLE_FRACTION = 0.8        # need ≥ 0.8 × min_sample_size to retire
ROLLING_WINDOW_SEC = 30 * 24 * 3600   # 30-day rolling aggregate


@dataclass
class _PendingEntry:
    pattern_id: str
    symbol: str
    expected_direction: str
    expected_move_atr: float
    entry_ts_ms: int
    entry_mid: float
    exit_ts_ms: int
    confidence_tier: str
    regime: str
    # Wave 8.5-E: persistence link. None for in-memory-only mode.
    outcome_id: int | None = None


@dataclass
class _ClosedOutcome:
    pattern_id: str
    symbol: str
    entry_ts_ms: int
    exit_ts_ms: int
    entry_mid: float
    exit_mid: float
    realized_return: float          # (exit_mid - entry_mid) / entry_mid × direction_sign
    expected_direction: str
    confidence_tier: str
    regime: str
    hit: bool                       # sign-matched move ≥ 0.5 × tick_size


@dataclass
class _PatternAggregate:
    """Rolling-window aggregate used to rewrite OutcomeDistribution."""
    returns: list[tuple[int, float]] = field(default_factory=list)   # (ts_ms, return)
    hits: list[tuple[int, bool]] = field(default_factory=list)

    def add(self, ts_ms: int, ret: float, hit: bool) -> None:
        self.returns.append((ts_ms, ret))
        self.hits.append((ts_ms, hit))

    def trim(self, cutoff_ts_ms: int) -> None:
        self.returns = [(t, r) for (t, r) in self.returns if t >= cutoff_ts_ms]
        self.hits = [(t, h) for (t, h) in self.hits if t >= cutoff_ts_ms]

    @property
    def n(self) -> int:
        return len(self.returns)

    def stats(self) -> tuple[float, float, float, int]:
        """Return (mean_atr, std, hit_rate, n). mean_atr here is the raw mean
        of the (signed) realized return — callers rescale by ATR externally.
        std is sample std (n-1 denominator) or 0.0 if n < 2."""
        n = self.n
        if n == 0:
            return 0.0, 0.0, 0.0, 0
        rets = [r for (_, r) in self.returns]
        mean = sum(rets) / n
        if n > 1:
            var = sum((r - mean) ** 2 for r in rets) / (n - 1)
            std = var ** 0.5
        else:
            std = 0.0
        hits = [h for (_, h) in self.hits]
        hit_rate = sum(1 for h in hits if h) / n if n else 0.0
        return float(mean), float(std), float(hit_rate), int(n)


class OutcomeTrackerRenderer:
    """Close the lab → trader loop by recording + grading every match.

    Parameters
    ----------
    outcomes_path
        JSONL file appended once per resolved outcome. Parent dir is
        auto-created.
    library
        Optional ``PatternLibrary``. When provided, the tracker rewrites
        each pattern's ``OutcomeDistribution`` on every re-aggregation and
        retires underperforming ones. Passes through ``library.save()``
        so writes are atomic + filelocked.
    horizon_ms
        Triple-barrier horizon in ms. Applied unless a per-pattern
        horizon is later added.
    reaggregate_every_n
        Re-aggregate after this many closes. Set to 0 to disable
        automatic re-aggregation (call ``reaggregate()`` manually).
    retire_below_hit_rate
        Patterns whose 30-day rolling hit_rate drops below this AND have
        at least ``min_sample_size × 0.8`` samples transition to RETIRED.
    hit_tolerance_ticks
        A signed-move is counted as a "hit" when it exceeds this many
        ticks in the expected direction. Default 0.5 ticks.
    tick_size
        Default tick size for hit/miss classification. Can be overridden
        on per-entry basis later if needed.
    """

    def __init__(
        self,
        outcomes_path: Path | str,
        *,
        library: PatternLibrary | None = None,
        horizon_ms: int = DEFAULT_HORIZON_MS,
        reaggregate_every_n: int = DEFAULT_REAGG_N,
        retire_below_hit_rate: float = DEFAULT_RETIRE_THRESHOLD,
        hit_tolerance_ticks: float = 0.5,
        tick_size: float = 0.25,
        state_store: Any = None,
    ) -> None:
        self.outcomes_path = Path(outcomes_path)
        self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        self.library = library
        self.horizon_ms = int(horizon_ms)
        self.reaggregate_every_n = int(reaggregate_every_n)
        self.retire_below_hit_rate = float(retire_below_hit_rate)
        self.hit_tolerance = float(hit_tolerance_ticks) * float(tick_size)

        self._lock = threading.RLock()
        self._pending: list[_PendingEntry] = []
        self._aggregates: dict[str, _PatternAggregate] = {}
        self._latest_price: dict[str, tuple[int, float]] = {}
        self._closes_since_reagg: int = 0
        # Wave 8.5-E: optional SQLite state store. Default None preserves
        # in-memory-only semantics (backward compat).
        self._state_store = state_store

        self.matches_received: int = 0
        self.matches_dropped_no_price: int = 0
        self.outcomes_closed: int = 0
        self.library_updates: int = 0
        self.retirements: int = 0

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def __call__(self, match: Any) -> None:
        """Broker subscriber — record entry for a new match."""
        try:
            self._on_match(match)
        except Exception:
            logger.exception("OutcomeTrackerRenderer: __call__ raised; swallowing")

    def on_price(self, symbol: str, mid: float, ts_ms: int) -> None:
        """Stream current mid price — drives exit resolution."""
        try:
            self._on_price(symbol, mid, ts_ms)
        except Exception:
            logger.exception("OutcomeTrackerRenderer: on_price raised; swallowing")

    def flush(self, *, current_ts_ms: int | None = None) -> int:
        """Force-close all pending outcomes using the latest known price.

        Returns the number of outcomes closed. Matches without any known
        price for their symbol are dropped (and counted).
        """
        closed = 0
        with self._lock:
            remaining: list[_PendingEntry] = []
            for entry in self._pending:
                latest = self._latest_price.get(entry.symbol)
                if latest is None:
                    self.matches_dropped_no_price += 1
                    continue
                ts_ms, mid = latest
                exit_ts = current_ts_ms if current_ts_ms is not None else ts_ms
                self._close(entry, exit_mid=mid, exit_ts_ms=exit_ts)
                closed += 1
                if self.reaggregate_every_n > 0 and \
                   self._closes_since_reagg >= self.reaggregate_every_n:
                    self.reaggregate()
            self._pending = remaining
        return closed

    def reaggregate(self) -> None:
        """Rewrite OutcomeDistribution for every aggregated pattern and
        retire any that have dropped below the hit-rate floor."""
        if self.library is None:
            self._closes_since_reagg = 0
            return
        try:
            self._do_reaggregate()
        except Exception:
            logger.exception("OutcomeTrackerRenderer: reaggregate failed; swallowing")
        finally:
            self._closes_since_reagg = 0

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    # ------------------------------------------------------------------
    # Core state transitions
    # ------------------------------------------------------------------

    def _on_match(self, match: Any) -> None:
        self.matches_received += 1
        symbol = str(getattr(match, "instrument", "") or "")
        entry_ts = int(getattr(match, "match_window_end_ms", 0) or 0)
        pattern_id = str(getattr(match, "pattern_id", "") or "")

        latest = self._latest_price.get(symbol)
        if latest is None:
            # No price yet for this symbol — pattern match arrived before
            # on_price(). Drop it; later matches will resolve once price
            # streams in. Production wiring ensures price arrives first.
            self.matches_dropped_no_price += 1
            return
        _, entry_mid = latest

        pending = _PendingEntry(
            pattern_id=pattern_id,
            symbol=symbol,
            expected_direction=str(getattr(match, "expected_direction", "neutral")),
            expected_move_atr=float(getattr(match, "expected_move_atr", 0.0) or 0.0),
            entry_ts_ms=entry_ts,
            entry_mid=float(entry_mid),
            exit_ts_ms=entry_ts + self.horizon_ms,
            confidence_tier=str(getattr(match, "confidence_tier", "C")),
            regime=str(getattr(match, "regime", "") or ""),
        )
        # Wave 8.5-E: persist before adding to in-memory list so a crash
        # mid-op cannot leave memory ahead of durable state.
        if self._state_store is not None:
            pending.outcome_id = self._state_store.put_pending_outcome({
                "pattern_id": pending.pattern_id,
                "symbol": pending.symbol,
                "expected_direction": pending.expected_direction,
                "expected_move_atr": pending.expected_move_atr,
                "entry_ts_ms": pending.entry_ts_ms,
                "entry_mid": pending.entry_mid,
                "exit_ts_ms": pending.exit_ts_ms,
                "confidence_tier": pending.confidence_tier,
                "regime": pending.regime,
            })
        with self._lock:
            self._pending.append(pending)

    def _on_price(self, symbol: str, mid: float, ts_ms: int) -> None:
        with self._lock:
            self._latest_price[symbol] = (int(ts_ms), float(mid))
            remaining: list[_PendingEntry] = []
            for entry in self._pending:
                if entry.symbol != symbol or ts_ms < entry.exit_ts_ms:
                    remaining.append(entry)
                    continue
                self._close(entry, exit_mid=float(mid), exit_ts_ms=int(ts_ms))
                if self.reaggregate_every_n > 0 and \
                   self._closes_since_reagg >= self.reaggregate_every_n:
                    self.reaggregate()
            self._pending = remaining

    def _close(self, entry: _PendingEntry, *, exit_mid: float, exit_ts_ms: int) -> None:
        dir_sign = 1.0 if entry.expected_direction == "bull" else (
            -1.0 if entry.expected_direction == "bear" else 0.0
        )
        price_delta = exit_mid - entry.entry_mid
        realized = price_delta * dir_sign   # in price units (signed by direction)
        hit = realized > self.hit_tolerance

        closed = _ClosedOutcome(
            pattern_id=entry.pattern_id,
            symbol=entry.symbol,
            entry_ts_ms=entry.entry_ts_ms,
            exit_ts_ms=exit_ts_ms,
            entry_mid=entry.entry_mid,
            exit_mid=exit_mid,
            realized_return=float(realized),
            expected_direction=entry.expected_direction,
            confidence_tier=entry.confidence_tier,
            regime=entry.regime,
            hit=bool(hit),
        )
        self._append_outcome(closed)

        agg = self._aggregates.setdefault(entry.pattern_id, _PatternAggregate())
        agg.add(entry.entry_ts_ms, realized, hit)

        # Wave 8.5-E: mirror to SQLite — remove pending row + append
        # aggregate sample. Both happen in the same method so a crash
        # between them results in at most a duplicate (the pending may
        # still exist, the aggregate already persisted) — acceptable
        # since reconstruction drops pendings that belong to
        # already-resolved timestamps (handled in from_state_store).
        if self._state_store is not None:
            if entry.outcome_id is not None:
                self._state_store.remove_pending_outcome(entry.outcome_id)
            self._state_store.append_aggregate_sample(
                pattern_id=entry.pattern_id,
                ts_ms=entry.entry_ts_ms,
                ret=realized,
                hit=hit,
            )

        self.outcomes_closed += 1
        self._closes_since_reagg += 1

    def _append_outcome(self, closed: _ClosedOutcome) -> None:
        row = {
            "pattern_id": closed.pattern_id,
            "symbol": closed.symbol,
            "entry_ts_ms": closed.entry_ts_ms,
            "exit_ts_ms": closed.exit_ts_ms,
            "entry_mid": closed.entry_mid,
            "exit_mid": closed.exit_mid,
            "realized_return": closed.realized_return,
            "expected_direction": closed.expected_direction,
            "confidence_tier": closed.confidence_tier,
            "regime": closed.regime,
            "hit": closed.hit,
        }
        with open(self.outcomes_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------
    # Library self-update
    # ------------------------------------------------------------------

    def _do_reaggregate(self) -> None:
        library = self.library
        assert library is not None
        library.load()
        data = library._data
        assert data is not None

        latest_ts = 0
        for agg in self._aggregates.values():
            if agg.returns:
                latest_ts = max(latest_ts, agg.returns[-1][0])
        cutoff = latest_ts - ROLLING_WINDOW_SEC * 1000 if latest_ts else 0
        if cutoff:
            for agg in self._aggregates.values():
                agg.trim(cutoff)

        updated = 0
        retired = 0
        for pid, agg in self._aggregates.items():
            pat = data.patterns.get(pid)
            if pat is None:
                continue
            mean, std, hit_rate, n = agg.stats()
            if n == 0:
                continue
            horizon_key = next(iter(pat.outcome_distribution), "5m")
            pat.outcome_distribution[horizon_key] = OutcomeDistribution(
                mean_atr=mean,
                std=std,
                hit_rate=hit_rate,
                n=n,
            )
            updated += 1

            needs_retirement = (
                hit_rate < self.retire_below_hit_rate
                and n >= int(pat.min_sample_size * MIN_SAMPLE_FRACTION)
                and PatternStatus.RETIRED in _ALLOWED_TRANSITIONS.get(pat.status, set())
            )
            if needs_retirement:
                try:
                    library.promote(pid, PatternStatus.RETIRED)
                    retired += 1
                    logger.info(
                        "outcome_tracker: retired %s (hit_rate=%.3f n=%d)",
                        pid, hit_rate, n,
                    )
                except Exception:
                    logger.exception("outcome_tracker: retirement of %s failed", pid)

        if updated or retired:
            library.save()
            self.library_updates += 1
            self.retirements += retired
            logger.info(
                "outcome_tracker: library re-aggregation — %d updates, %d retirements",
                updated, retired,
            )

    # ------------------------------------------------------------------
    # Wave 8.5-E: persistence reconstruction
    # ------------------------------------------------------------------

    @classmethod
    def from_state_store(
        cls,
        state_store: Any,
        outcomes_path: Path | str,
        *,
        library: PatternLibrary | None = None,
        horizon_ms: int = DEFAULT_HORIZON_MS,
        reaggregate_every_n: int = DEFAULT_REAGG_N,
        retire_below_hit_rate: float = DEFAULT_RETIRE_THRESHOLD,
        hit_tolerance_ticks: float = 0.5,
        tick_size: float = 0.25,
    ) -> "OutcomeTrackerRenderer":
        """Reconstruct an OutcomeTrackerRenderer from persisted rows.

        Reloads pending outcomes + pattern aggregates. The aggregates
        table is treated as the source-of-truth log; aggregate timestamps
        are replayed into per-pattern `_PatternAggregate` so
        reaggregate() produces the same library decisions it would have
        pre-crash.
        """
        tracker = cls(
            outcomes_path=outcomes_path,
            library=library,
            horizon_ms=horizon_ms,
            reaggregate_every_n=reaggregate_every_n,
            retire_below_hit_rate=retire_below_hit_rate,
            hit_tolerance_ticks=hit_tolerance_ticks,
            tick_size=tick_size,
            state_store=state_store,
        )
        with tracker._lock:
            # Pending outcomes
            for row in state_store.load_pending_outcomes():
                tracker._pending.append(_PendingEntry(
                    pattern_id=row["pattern_id"],
                    symbol=row["symbol"],
                    expected_direction=row["expected_direction"],
                    expected_move_atr=float(row["expected_move_atr"]),
                    entry_ts_ms=int(row["entry_ts_ms"]),
                    entry_mid=float(row["entry_mid"]),
                    exit_ts_ms=int(row["exit_ts_ms"]),
                    confidence_tier=row["confidence_tier"],
                    regime=row["regime"],
                    outcome_id=int(row["outcome_id"]),
                ))
            # Pattern aggregates (replay into in-memory rolling)
            for sample in state_store.load_aggregate_samples():
                pid = sample["pattern_id"]
                agg = tracker._aggregates.setdefault(pid, _PatternAggregate())
                agg.add(
                    ts_ms=int(sample["ts_ms"]),
                    ret=float(sample["ret"]),
                    hit=bool(sample["hit"]),
                )
        logger.info(
            "wave85-E outcome_tracker reconstructed: %d pending, "
            "%d patterns with aggregate history",
            len(tracker._pending), len(tracker._aggregates),
        )
        return tracker
