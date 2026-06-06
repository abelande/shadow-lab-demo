"""
p6lab.execution.fill_simulator — Passive order fill model.

Spec: p6-notebook-lab-spec.md §6.2

Two modes:
  simulate_bulk(orders) — no trajectory, vectorized, for notebook 05
    (target: 10,000+ orders in <5 min).
  simulate_interactive(order) — full trajectory for UI animation (§10.2).

Both consume the same ``QueueTracker`` internally. The tracker state
is shared across orders within a bulk call so that MBO events are
processed exactly once.

Fill detection logic:
  For each event in the stream after order placement:
    1. QueueTracker.on_event(event) updates all level queues.
    2. If event is a FILL/TRADE at our order's price: the tracker
       consumes from the front. If the front was our virtual order,
       its ``size`` drops — we detect this as a fill.
    3. After each event, we re-query our position. If the virtual
       order has been removed (fully filled) or its size reduced
       (partial fill), we record the fill.
    4. On non-fill events we also check adverse-exit conditions
       (mid price moved > ``adverse_exit_ticks`` against our price)
       and timeout.

Exit conditions (ordered by precedence):
  "full"         — fully filled
  "partial"      — filled_size > 0 but < order.size at timeout/exit
  "adverse_exit" — price moved ``adverse_exit_ticks`` against us
  "timeout"      — ``max_horizon_ms`` elapsed
  "cancelled"    — manual cancellation (simulate_interactive only)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Literal

from .queue_tracker import (
    OrderHandle,
    QueueSnapshot,
    QueueTracker,
    Side,
)


FillReason = Literal[
    "full", "partial", "adverse_exit", "timeout", "cancelled"
]


@dataclass(frozen=True)
class FillOutcome:
    """Result of simulating one passive order."""
    filled: bool
    filled_size: float
    fill_timestamp_ms: int | None
    queue_position_at_entry: float
    queue_position_at_fill: float | None
    adverse_ticks_at_fill: int
    realized_pnl: float
    fill_reason: FillReason
    trajectory: list[QueueSnapshot] = field(default_factory=list)


@dataclass
class OrderSpec:
    """Specification for a simulated order."""
    timestamp_ms: int
    side: Side
    price: float
    size: float
    order_type: Literal["limit", "market", "step_ahead"] = "limit"
    max_horizon_ms: int = 60_000
    adverse_exit_ticks: int = 4


# ---------------------------------------------------------------------------
# FillSimulator
# ---------------------------------------------------------------------------

class FillSimulator:
    """Walks the MBO event stream forward from order placement.

    The simulator is driven by an **event iterator** — any iterable
    yielding objects with the attributes QueueTracker.on_event()
    consumes. For convenience the caller can also pass a list of
    pre-collected events.

    For market data that ships snapshot-by-snapshot (e.g., p6-v2's
    ``OrderBookSnapshot``), the caller should flatten ``recent_events``
    from each snapshot into a single event iterator before calling.

    Mid-price for adverse-exit tracking is computed from tracker state:
    best bid = max price in ``_bid_levels``, best ask = min in
    ``_ask_levels``. When the tracker has no state for one side, we
    fall back to the order's own price as the reference.
    """

    def __init__(
        self,
        tick_size: float = 0.25,
        tick_value: float = 12.50,
    ) -> None:
        self.tick_size = tick_size
        self.tick_value = tick_value

    # ──────────────────────────────────────────────────────────────
    # Public: bulk path
    # ──────────────────────────────────────────────────────────────

    # Bulk path uses a coarser sweep for timeout/adverse checks
    # (per-event sweeps across all active orders are O(N_orders×N_events)
    # and dominate runtime on long replays).
    BULK_SWEEP_INTERVAL_MS: int = 1000
    BULK_PRICE_KEY_PRECISION: int = 8   # round price for dict key stability

    def simulate_bulk(
        self,
        orders: list[OrderSpec],
        event_stream: Iterable[Any],
    ) -> list[FillOutcome]:
        """Simulate many orders in a single pass over the event stream.

        All orders share one ``QueueTracker`` so MBO events are
        processed exactly once regardless of order count. Orders are
        sorted by placement time; each becomes active when its
        ``timestamp_ms`` is reached.

        Performance: the hot loop is O(E + A × S) where E is the event
        count, A is the order count, and S is the number of
        timeout/adverse sweeps over the replay (one per
        ``BULK_SWEEP_INTERVAL_MS``). Each event only triggers an
        order-specific check for orders AT THE SAME PRICE as the event.
        Orders at other prices are only examined during the periodic
        sweep — they can't have fill / advance events caused by an
        event at a different price.

        No trajectory recording — ``FillOutcome.trajectory`` is empty.
        Target: 10k orders × millions of events in <5 min.
        """
        tracker = QueueTracker(tick_size=self.tick_size)
        # Sort orders by placement time; keep original index for output order.
        indexed = sorted(enumerate(orders), key=lambda p: p[1].timestamp_ms)

        results: list[FillOutcome | None] = [None] * len(orders)
        active: dict[int, _ActiveState] = {}
        # Price-keyed index: (Side, rounded_price) → set of active idx
        by_price: dict[tuple[Side, float], set[int]] = {}
        pending = list(indexed)

        last_event_ts: int | None = None
        last_sweep_ts: int = 0

        for event in event_stream:
            ev_ts = int(getattr(event, "timestamp_ms", 0) or 0)
            last_event_ts = ev_ts

            # 1. Activate any orders whose placement time has arrived.
            while pending and pending[0][1].timestamp_ms <= ev_ts:
                idx, spec = pending.pop(0)
                state = self._place_and_capture(tracker, spec, ev_ts)
                active[idx] = state
                self._add_to_price_index(by_price, idx, spec)

            # 2. Determine which price bucket this event touches (if any).
            ev_side = QueueTracker._side_from_event(event)
            ev_key: tuple[Side, float] | None = None
            if ev_side is not None:
                ev_key = (
                    ev_side,
                    round(float(event.price), self.BULK_PRICE_KEY_PRECISION),
                )

            # 3. Apply event to the tracker.
            tracker.on_event(event)

            # 4. Check ONLY orders at the affected price (if any).
            finished: list[int] = []
            if ev_key is not None and ev_key in by_price:
                for idx in list(by_price[ev_key]):
                    state = active[idx]
                    outcome = self._check_after_event(
                        tracker, state, ev_ts, record_trajectory=False,
                    )
                    if outcome is not None:
                        results[idx] = outcome
                        finished.append(idx)

            # 5. Periodic coarse sweep for timeout + adverse-exit over
            #    orders NOT touched in step 4.
            if ev_ts - last_sweep_ts >= self.BULK_SWEEP_INTERVAL_MS:
                last_sweep_ts = ev_ts
                for idx, state in active.items():
                    if results[idx] is not None:
                        continue
                    # Cheap timeout check first
                    if ev_ts - state.spec.timestamp_ms >= state.spec.max_horizon_ms:
                        results[idx] = self._timeout_outcome(state, ev_ts)
                        if idx not in finished:
                            finished.append(idx)
                        continue
                    # Adverse-exit check (estimate mid once per sweep)
                    mid = self._estimate_mid(tracker, state.spec.price)
                    state.last_mid = mid
                    adverse = self._adverse_ticks(state.spec, mid)
                    if adverse >= state.spec.adverse_exit_ticks:
                        results[idx] = self._exit_outcome(
                            state, ev_ts, reason="adverse_exit",
                            adverse_ticks=adverse, mid=mid,
                        )
                        if idx not in finished:
                            finished.append(idx)

            # 6. Evict finished orders from both indices.
            for idx in finished:
                state = active.pop(idx, None)
                if state is not None:
                    self._remove_from_price_index(by_price, idx, state.spec)

        # 7. Any remaining pending orders never got activated → timeout.
        for idx, spec in pending:
            pseudo_ts = last_event_ts if last_event_ts is not None else spec.timestamp_ms
            state = self._place_and_capture(tracker, spec, pseudo_ts)
            results[idx] = self._timeout_outcome(state, pseudo_ts)

        # 8. Any still-active orders at end of stream → timeout.
        for idx, state in active.items():
            if results[idx] is not None:
                continue
            end_ts = last_event_ts if last_event_ts is not None else state.spec.timestamp_ms
            results[idx] = self._timeout_outcome(state, end_ts)

        return [r if r is not None else self._empty_outcome() for r in results]

    @staticmethod
    def _add_to_price_index(
        by_price: dict[tuple[Side, float], set[int]],
        idx: int,
        spec: OrderSpec,
    ) -> None:
        key = (spec.side, round(spec.price, 8))
        by_price.setdefault(key, set()).add(idx)

    @staticmethod
    def _remove_from_price_index(
        by_price: dict[tuple[Side, float], set[int]],
        idx: int,
        spec: OrderSpec,
    ) -> None:
        key = (spec.side, round(spec.price, 8))
        bucket = by_price.get(key)
        if bucket is None:
            return
        bucket.discard(idx)
        if not bucket:
            del by_price[key]

    # ──────────────────────────────────────────────────────────────
    # Public: interactive path (single order + trajectory)
    # ──────────────────────────────────────────────────────────────

    def simulate_interactive(
        self,
        order: OrderSpec,
        event_stream: Iterable[Any],
    ) -> FillOutcome:
        """Simulate one order with full per-event trajectory capture.

        Returns ``FillOutcome`` with ``trajectory`` populated — one
        ``QueueSnapshot`` per relevant post-placement event.
        """
        tracker = QueueTracker(tick_size=self.tick_size)

        state: _ActiveState | None = None
        last_event_ts: int | None = None

        for event in event_stream:
            ev_ts = int(getattr(event, "timestamp_ms", 0) or 0)
            last_event_ts = ev_ts

            if state is None and ev_ts >= order.timestamp_ms:
                state = self._place_and_capture(tracker, order, ev_ts)
                # Record the initial "placed" snapshot
                state.trajectory.append(self._snap(
                    state, ev_ts, event_type="placed",
                    mid_price=self._estimate_mid(tracker, order.price),
                ))

            tracker.on_event(event)

            if state is None:
                continue

            outcome = self._check_after_event(
                tracker, state, ev_ts, record_trajectory=True,
            )
            if outcome is not None:
                return outcome

        # Stream exhausted without fill/exit → timeout
        end_ts = last_event_ts if last_event_ts is not None else order.timestamp_ms
        if state is None:
            state = self._place_and_capture(tracker, order, end_ts)
            state.trajectory.append(self._snap(
                state, end_ts, event_type="placed",
                mid_price=self._estimate_mid(tracker, order.price),
            ))
        return self._timeout_outcome(state, end_ts)

    # ──────────────────────────────────────────────────────────────
    # Public: marketable path (crosses the spread, fills immediately)
    # ──────────────────────────────────────────────────────────────

    def simulate_marketable(
        self,
        order: OrderSpec,
        event_stream: Iterable[Any],
    ) -> FillOutcome:
        """Simulate one aggressive order that crosses the spread immediately.

        Walks the event stream just far enough to rebuild the book at / after
        ``order.timestamp_ms``, then fills the requested size by sweeping
        opposite-side levels from the inside out. Slippage = number of levels
        crossed × ``tick_size``. The order never rests; if the opposite side
        is empty the outcome is ``NO_FILL_TIMEOUT`` (treated as a cancel).

        No trajectory capture. Intended for modeling market orders and
        marketable limits in the bulk-backtest path.
        """
        tracker = QueueTracker(tick_size=self.tick_size)
        arrival_ts: int | None = None

        for event in event_stream:
            ev_ts = int(getattr(event, "timestamp_ms", 0) or 0)
            tracker.on_event(event)
            if ev_ts >= order.timestamp_ms:
                arrival_ts = ev_ts
                break

        if arrival_ts is None:
            # Stream never reached the order's placement time — empty book
            return self._empty_outcome()

        # Sweep opposite side from the inside out
        opposite_side = Side.SELL if order.side == Side.BUY else Side.BUY
        sizes_map = tracker.level_sizes(opposite_side)
        if not sizes_map:
            return self._empty_outcome()

        # Best ask = lowest price; best bid = highest price.
        prices = sorted(sizes_map.keys(),
                        reverse=(opposite_side == Side.BUY))

        remaining = order.size
        filled_size = 0.0
        weighted_px = 0.0
        levels_crossed = 0

        for px in prices:
            if remaining <= 0:
                break
            avail = sizes_map.get(px, 0.0)
            take = min(avail, remaining)
            if take <= 0:
                levels_crossed += 1
                continue
            weighted_px += px * take
            filled_size += take
            remaining -= take
            levels_crossed += 1

        if filled_size <= 0:
            return self._empty_outcome()

        vwap = weighted_px / filled_size
        # Slippage in ticks: number of levels consumed beyond the best (best = 1 level)
        slippage_ticks = max(0, levels_crossed - 1)
        mid_at_fill = self._estimate_mid(tracker, vwap)

        # PnL: immediate mark vs. fill price
        if self.tick_size > 0:
            sign = 1.0 if order.side == Side.BUY else -1.0
            ticks_pnl = (mid_at_fill - vwap) / self.tick_size * sign
            pnl = ticks_pnl * self.tick_value * filled_size
        else:
            pnl = 0.0

        reason: FillReason = "full" if remaining <= 0 else "partial"
        return FillOutcome(
            filled=True,
            filled_size=filled_size,
            fill_timestamp_ms=arrival_ts,
            queue_position_at_entry=0.0,   # marketables skip the queue
            queue_position_at_fill=0.0,
            adverse_ticks_at_fill=slippage_ticks,
            realized_pnl=pnl,
            fill_reason=reason,
            trajectory=[],
        )

    # ──────────────────────────────────────────────────────────────
    # Core step: inspect state after one event
    # ──────────────────────────────────────────────────────────────

    def _check_after_event(
        self,
        tracker: QueueTracker,
        state: "_ActiveState",
        ev_ts: int,
        record_trajectory: bool,
    ) -> FillOutcome | None:
        """Return ``FillOutcome`` if the order has resolved, else ``None``."""
        # 1. Check for fill: look up current queue state for our handle.
        #    If the handle has been removed or its size has shrunk, the
        #    entry was consumed by a FIFO fill ahead of / at us.
        current_size = self._tracked_size(tracker, state.handle)

        if current_size is None:
            # Order is gone from all level queues → fully filled
            state.filled_size = state.spec.size
            filled_ts = ev_ts
            adverse = 0  # price at fill; see notes below
            mid = self._estimate_mid(tracker, state.spec.price)
            fill_price = state.spec.price
            pnl = self._realized_pnl(state, fill_price, mid)

            if record_trajectory:
                state.trajectory.append(self._snap(
                    state, ev_ts, event_type="filled",
                    mid_price=mid, total=0.0, position=0.0,
                ))

            return FillOutcome(
                filled=True,
                filled_size=state.spec.size,
                fill_timestamp_ms=filled_ts,
                queue_position_at_entry=state.entry_position,
                queue_position_at_fill=0.0,
                adverse_ticks_at_fill=adverse,
                realized_pnl=pnl,
                fill_reason="full",
                trajectory=state.trajectory,
            )

        # Detect partial fill: size reduced without removal
        if current_size < state.last_observed_size:
            delta = state.last_observed_size - current_size
            state.filled_size += delta
            state.last_observed_size = current_size
            # partial fill — we continue (the spec's "partial" reason is
            # only emitted at timeout/exit with any fill_size > 0)

        # 2. Update current queue position
        position = tracker.get_position(state.handle).position_from_front
        total_at_level = tracker.get_position(state.handle).total_at_level
        mid = self._estimate_mid(tracker, state.spec.price)

        if record_trajectory:
            event_type = self._classify_event(state, position, total_at_level)
            state.trajectory.append(self._snap(
                state, ev_ts, event_type=event_type,
                mid_price=mid, total=total_at_level, position=position,
            ))

        state.last_position = position
        state.last_total = total_at_level
        state.last_mid = mid

        # 3. Adverse exit check
        adverse_ticks = self._adverse_ticks(state.spec, mid)
        if adverse_ticks >= state.spec.adverse_exit_ticks:
            return self._exit_outcome(
                state, ev_ts, reason="adverse_exit",
                adverse_ticks=adverse_ticks, mid=mid,
            )

        # 4. Timeout check
        if ev_ts - state.spec.timestamp_ms >= state.spec.max_horizon_ms:
            return self._timeout_outcome(state, ev_ts)

        return None

    # ──────────────────────────────────────────────────────────────
    # Adverse selection
    # ──────────────────────────────────────────────────────────────

    def _compute_adverse_ticks(
        self,
        fill_price: float,
        side: Side,
        post_fill_events: Iterable[Any],
        horizons_ms: list[int] | None = None,
    ) -> dict[int, int]:
        """Measure worst adverse excursion across horizons after fill.

        For each horizon, walks events forward and tracks the worst
        adverse price. Returns ``{horizon_ms: adverse_ticks}`` where
        ``adverse_ticks = max ticks price moved against us within the
        horizon window``.

        "Against us" means:
          BUY order filled at ``fill_price`` → adverse = price falls
          SELL order filled at ``fill_price`` → adverse = price rises
        """
        if horizons_ms is None:
            horizons_ms = [1000, 5000, 30000]

        # Collect mid-price samples per-event using a throwaway tracker
        mini = QueueTracker(tick_size=self.tick_size)
        samples: list[tuple[int, float]] = []  # (ts_offset_ms, mid)
        fill_ts = None
        for ev in post_fill_events:
            ev_ts = int(getattr(ev, "timestamp_ms", 0) or 0)
            if fill_ts is None:
                fill_ts = ev_ts
            mini.on_event(ev)
            mid = self._estimate_mid(mini, fill_price)
            samples.append((ev_ts - fill_ts, mid))

        result: dict[int, int] = {}
        for h in horizons_ms:
            worst = 0
            for ts_off, mid in samples:
                if ts_off > h:
                    break
                if side == Side.BUY:
                    # Adverse = price falls below fill price
                    ticks = int(round((fill_price - mid) / self.tick_size))
                else:
                    # Adverse = price rises above fill price
                    ticks = int(round((mid - fill_price) / self.tick_size))
                if ticks > worst:
                    worst = ticks
            result[h] = worst
        return result

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _place_and_capture(
        self, tracker: QueueTracker, spec: OrderSpec, ev_ts: int,
    ) -> "_ActiveState":
        """Insert the virtual order and snapshot its entry conditions."""
        handle = tracker.place_limit_order(
            timestamp_ms=ev_ts, side=spec.side,
            price=spec.price, size=spec.size,
        )
        pos = tracker.get_position(handle)
        return _ActiveState(
            spec=spec, handle=handle,
            entry_position=pos.position_from_front,
            last_position=pos.position_from_front,
            last_total=pos.total_at_level,
            last_observed_size=spec.size,
            filled_size=0.0,
            trajectory=[],
            last_mid=self._estimate_mid(tracker, spec.price),
        )

    def _tracked_size(
        self, tracker: QueueTracker, handle: OrderHandle,
    ) -> float | None:
        """Return current size of the virtual order, or None if it's gone."""
        level = tracker._level_for(handle.side, handle.price)
        if level is None:
            return None
        entry = level.get(handle._internal_id)
        return entry.size if entry is not None else None

    def _estimate_mid(
        self, tracker: QueueTracker, fallback_price: float,
    ) -> float:
        """Approximate mid from tracker state."""
        bids = tracker.level_sizes(Side.BUY)
        asks = tracker.level_sizes(Side.SELL)
        if bids and asks:
            return 0.5 * (max(bids) + min(asks))
        if bids:
            return max(bids)
        if asks:
            return min(asks)
        return fallback_price

    def _adverse_ticks(self, spec: OrderSpec, mid: float) -> int:
        """Ticks the mid has moved against the order's side."""
        if self.tick_size <= 0:
            return 0
        if spec.side == Side.BUY:
            # Adverse if mid < spec.price
            return max(0, int(round((spec.price - mid) / self.tick_size)))
        else:
            # Adverse if mid > spec.price
            return max(0, int(round((mid - spec.price) / self.tick_size)))

    def _realized_pnl(
        self, state: "_ActiveState", fill_price: float, mid_at_fill: float,
    ) -> float:
        """PnL at fill = (mid - fill_price) × size × tick_value / tick_size
        for BUY; opposite sign for SELL.

        Approximate: uses current mid as proxy for exit.
        """
        if self.tick_size <= 0:
            return 0.0
        sign = 1.0 if state.spec.side == Side.BUY else -1.0
        ticks = (mid_at_fill - fill_price) / self.tick_size * sign
        return ticks * self.tick_value * state.filled_size

    def _classify_event(
        self, state: "_ActiveState",
        new_position: float, new_total: float,
    ) -> str:
        """Return a descriptive tag for the trajectory snapshot."""
        if new_position < state.last_position:
            return "advanced"       # Fill or cancel ahead of us
        if new_total > state.last_total:
            return "add_behind"     # New depth piled on
        if new_total < state.last_total:
            return "cancel_elsewhere"
        return "unchanged"

    def _snap(
        self,
        state: "_ActiveState",
        ts: int,
        event_type: str,
        mid_price: float,
        total: float | None = None,
        position: float | None = None,
    ) -> QueueSnapshot:
        return QueueSnapshot(
            timestamp_ms=ts,
            position_from_front=position if position is not None else state.last_position,
            total_at_level=total if total is not None else state.last_total,
            event_type=event_type,
            price=state.spec.price,
            mid_price=mid_price,
        )

    def _exit_outcome(
        self,
        state: "_ActiveState", ts: int, reason: FillReason,
        adverse_ticks: int, mid: float,
    ) -> FillOutcome:
        """Exit with or without partial fill."""
        if state.filled_size > 0:
            # Partial fill then exit
            pnl = self._realized_pnl(state, state.spec.price, mid)
            final_reason: FillReason = (
                "partial" if reason == "timeout" else reason
            )
        else:
            pnl = 0.0
            final_reason = reason
        return FillOutcome(
            filled=state.filled_size >= state.spec.size,
            filled_size=state.filled_size,
            fill_timestamp_ms=ts if state.filled_size > 0 else None,
            queue_position_at_entry=state.entry_position,
            queue_position_at_fill=state.last_position if state.filled_size > 0 else None,
            adverse_ticks_at_fill=adverse_ticks,
            realized_pnl=pnl,
            fill_reason=final_reason,
            trajectory=state.trajectory,
        )

    def _timeout_outcome(
        self, state: "_ActiveState", ts: int,
    ) -> FillOutcome:
        """Timeout exit — partial fill if any, else unfilled."""
        mid = state.last_mid
        adverse = self._adverse_ticks(state.spec, mid)
        reason: FillReason = "partial" if state.filled_size > 0 else "timeout"
        if state.filled_size >= state.spec.size:
            reason = "full"
        return self._exit_outcome(state, ts, reason, adverse, mid)

    def _empty_outcome(self) -> FillOutcome:
        """Fallback for orders that somehow never got a result recorded."""
        return FillOutcome(
            filled=False,
            filled_size=0.0,
            fill_timestamp_ms=None,
            queue_position_at_entry=0.0,
            queue_position_at_fill=None,
            adverse_ticks_at_fill=0,
            realized_pnl=0.0,
            fill_reason="timeout",
            trajectory=[],
        )


# ---------------------------------------------------------------------------
# Internal state per active virtual order
# ---------------------------------------------------------------------------

@dataclass
class _ActiveState:
    """Mutable state held for each virtual order while it's alive."""
    spec: OrderSpec
    handle: OrderHandle
    entry_position: float
    last_position: float
    last_total: float
    last_observed_size: float   # used to detect partial fills
    filled_size: float
    trajectory: list[QueueSnapshot]
    last_mid: float
