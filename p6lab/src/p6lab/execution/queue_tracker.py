"""
p6lab.execution.queue_tracker — Per-Order Absolute Queue Position.

Spec: p6-notebook-lab-spec.md §6.1 | staircase:L74-80 (missing infra)
Ref:  OB-reference.md §2 strategy #2 (Queue Priority Alpha)

Maintains per-price-level FIFO queues of orders, updating in response to
raw MBO events. Supports "virtual" orders injected via
``place_limit_order()`` — these are tracked alongside real orders so
their queue position can be queried at any time, enabling fill
simulation and queue-priority analysis.

Design:
  - Levels keyed by price in a ``SortedDict`` (from ``sortedcontainers``)
    → O(log L) insert/lookup across L levels, O(N) per-level queue ops
  - Each level holds an ``OrderedDict[order_id, _Entry]`` preserving
    insertion order (FIFO). Per-level dict gives O(1) lookup by id and
    deterministic iteration for position calculation.
  - MODIFY price changes → treated as CANCEL + ADD (standard CME model)
  - MODIFY size-down → in-place reduction preserving queue position
  - MODIFY size-up → CME rules: if size grows, order loses priority
    (treated as CANCEL + ADD at end of queue). Configurable via
    ``modify_loses_priority`` flag.
  - FILL events consume from the FRONT of the level's queue (FIFO
    matching) for FIFO mode, or proportionally across the level for
    pro-rata mode.

Event dispatch (``on_event``):
  ADD        — insert at back of level's queue
  CANCEL     — remove from its level; if removed entry was ahead of
               any virtual order at the same level, that virtual order
               advances by the cancelled size
  MODIFY     — size decrease in place; price change = CANCEL + ADD
  FILL/TRADE — consume from front of level; same advancement rule

Virtual orders are a thin wrapper around real ``_Entry`` objects with
an ``is_virtual=True`` flag. They sit in the same FIFO queues so the
position math is identical — just flagged so ``place_limit_order()``
can return a stable ``OrderHandle`` and skip event propagation.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from sortedcontainers import SortedDict
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sortedcontainers is required for QueueTracker. "
        "Install with: pip install sortedcontainers"
    ) from e


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class MatchingAlgorithm(str, Enum):
    FIFO = "fifo"          # NQ, ES, most futures
    PRO_RATA = "pro_rata"  # some options / non-linear futures


@dataclass
class OrderHandle:
    """Handle returned when placing a virtual order.

    Holds enough context to recover the order from the tracker's state
    (price, side, internal id). The caller keeps this to later call
    ``get_position()`` or ``cancel_order()``.
    """
    handle_id: int
    side: Side
    price: float
    size: float
    timestamp_ms: int
    _internal_id: str = ""   # internal id string used in level queues


@dataclass(frozen=True)
class QueuePosition:
    """Current queue position for a tracked order."""
    handle_id: int
    position_from_front: float   # contracts ahead of this order
    total_at_level: float        # total contracts at this price
    fill_probability_estimate: float
    timestamp_ms: int


@dataclass(frozen=True)
class QueueSnapshot:
    """Per-event snapshot of queue state (for trajectory recording)."""
    timestamp_ms: int
    position_from_front: float
    total_at_level: float
    event_type: str   # "fill_ahead" | "cancel_ahead" | "add_behind" | "placed"
    price: float
    mid_price: float


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    """A single order (real or virtual) sitting in a level's FIFO queue."""
    order_id: str
    size: float
    timestamp_ms: int
    is_virtual: bool = False
    handle_id: int | None = None   # set iff is_virtual


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class QueueTracker:
    """Per-order absolute queue position tracker.

    Args:
        matching_algorithm: FIFO (NQ/ES) or PRO_RATA (some options).
        tick_size: Price increment. Used only for float-equality
            comparisons on prices to avoid FP roundoff issues.
        modify_loses_priority: If True, MODIFY that increases size is
            treated as CANCEL + ADD at end of queue (CME rule). If
            False, in-place (allows priority gaming — useful for
            research). Default True per CME convention.
    """

    def __init__(
        self,
        matching_algorithm: MatchingAlgorithm = MatchingAlgorithm.FIFO,
        tick_size: float = 0.25,
        modify_loses_priority: bool = True,
    ) -> None:
        self.matching_algorithm = matching_algorithm
        self.tick_size = tick_size
        self.modify_loses_priority = modify_loses_priority

        # Per-side level queues. Key = price (float), Value = OrderedDict
        # keyed by order_id. Preserves FIFO insertion order for O(1)
        # position-calc via cumulative-size walk.
        self._bid_levels: SortedDict = SortedDict()
        self._ask_levels: SortedDict = SortedDict()

        # Virtual order bookkeeping
        self._next_handle_id = 0
        self._handles: dict[int, OrderHandle] = {}
        # Virtual ids get a distinct prefix so they can't collide with
        # real MBO order ids.
        self._next_virtual_id = 0

    # ──────────────────────────────────────────────────────────────
    # Public: virtual-order placement / query
    # ──────────────────────────────────────────────────────────────

    def place_limit_order(
        self,
        timestamp_ms: int,
        side: Side | str,
        price: float,
        size: float,
    ) -> OrderHandle:
        """Place a virtual limit order and begin tracking its queue position.

        The virtual order is inserted at the BACK of the existing queue
        at ``price`` — same treatment as a real ADD. Returns an
        ``OrderHandle`` for subsequent ``get_position`` / ``cancel_order``
        calls.
        """
        side_enum = Side(side) if isinstance(side, str) else side
        self._next_handle_id += 1
        self._next_virtual_id += 1
        internal_id = f"virt-{self._next_virtual_id}"

        handle = OrderHandle(
            handle_id=self._next_handle_id,
            side=side_enum,
            price=price,
            size=size,
            timestamp_ms=timestamp_ms,
            _internal_id=internal_id,
        )
        self._handles[handle.handle_id] = handle

        entry = _Entry(
            order_id=internal_id,
            size=size,
            timestamp_ms=timestamp_ms,
            is_virtual=True,
            handle_id=handle.handle_id,
        )
        self._insert_entry_back(side_enum, price, entry)
        return handle

    def get_position(self, handle: OrderHandle) -> QueuePosition:
        """Current absolute queue position for a tracked virtual order.

        ``position_from_front`` is the sum of sizes of real+virtual orders
        strictly ahead of this one at the same price. A newly-placed
        order sees this as "the resting depth it has to wait through."
        """
        level = self._level_for(handle.side, handle.price)
        total_at_level = 0.0
        position = 0.0
        seen = False
        for oid, entry in (level.items() if level is not None else []):
            total_at_level += entry.size
            if oid == handle._internal_id:
                seen = True
            elif not seen:
                position += entry.size

        fill_prob = self._fill_probability(position, total_at_level)
        return QueuePosition(
            handle_id=handle.handle_id,
            position_from_front=position,
            total_at_level=total_at_level,
            fill_probability_estimate=fill_prob,
            timestamp_ms=handle.timestamp_ms,
        )

    def cancel_order(self, handle: OrderHandle) -> None:
        """Remove a tracked virtual order from its level."""
        level = self._level_for(handle.side, handle.price)
        if level is None:
            return
        level.pop(handle._internal_id, None)
        self._cleanup_empty_level(handle.side, handle.price)
        self._handles.pop(handle.handle_id, None)

    def get_all_positions(self) -> list[QueuePosition]:
        """Return queue position for every active virtual order."""
        return [self.get_position(h) for h in self._handles.values()]

    # ──────────────────────────────────────────────────────────────
    # Public: real-event ingestion
    # ──────────────────────────────────────────────────────────────

    def on_event(self, event: Any) -> None:
        """Process one MBO event and update all tracked orders.

        Accepted event interfaces (duck-typed):
          ``event.order_id``           : str
          ``event.side``               : Side enum with .name in {'BID','ASK'}
          ``event.price``              : float
          ``event.size``               : float
          ``event.action``             : Action enum with .name in
                                         {'ADD','CANCEL','MODIFY','FILL','TRADE'}
          ``event.timestamp_ms``       : int

        Unknown or malformed events are silently ignored — the tracker
        is tolerant of feed glitches.
        """
        action_name = self._action_name(event)
        side_enum = self._side_from_event(event)
        if side_enum is None:
            return

        order_id = str(getattr(event, "order_id", ""))
        if not order_id:
            return

        price = float(event.price)
        size = float(getattr(event, "size", 0.0) or 0.0)
        ts = int(getattr(event, "timestamp_ms", 0))

        if action_name == "ADD":
            self._on_add(side_enum, price, order_id, size, ts)
        elif action_name == "CANCEL":
            self._on_cancel(side_enum, price, order_id)
        elif action_name == "MODIFY":
            self._on_modify(side_enum, price, order_id, size, ts)
        elif action_name in ("FILL", "TRADE"):
            self._on_fill(side_enum, price, size)
        # Unknown actions are ignored

    # ──────────────────────────────────────────────────────────────
    # Event handlers
    # ──────────────────────────────────────────────────────────────

    def _on_add(
        self, side: Side, price: float, order_id: str,
        size: float, ts: int,
    ) -> None:
        """Insert a real order at back of level queue (standard CME ADD)."""
        if size <= 0:
            return
        entry = _Entry(order_id=order_id, size=size, timestamp_ms=ts)
        self._insert_entry_back(side, price, entry)

    def _on_cancel(self, side: Side, price: float, order_id: str) -> None:
        """Remove an order from its level.

        No explicit "advance virtual order" step is needed — virtual
        orders' positions are computed on demand by ``get_position()``,
        which walks the current level. Removing the cancelled entry
        from the OrderedDict naturally advances anyone behind it.
        """
        level = self._level_for(side, price)
        if level is None:
            return
        level.pop(order_id, None)
        self._cleanup_empty_level(side, price)

    def _on_modify(
        self, side: Side, price: float, order_id: str,
        new_size: float, ts: int,
    ) -> None:
        """MODIFY: size-down in place; size-up or price-change = CANCEL + ADD.

        Standard CME rule: only size-down preserves queue priority.
        """
        if new_size <= 0:
            # Effectively a cancel
            self._on_cancel(side, price, order_id)
            return

        # Find the existing entry across both sides' levels (modify may
        # come with a different price than the original)
        existing_level, existing_price = self._locate_entry(side, order_id)
        if existing_level is None:
            # Unknown order — treat as fresh ADD
            self._on_add(side, price, order_id, new_size, ts)
            return

        existing_entry = existing_level[order_id]

        price_changed = abs(existing_price - price) > self.tick_size * 0.5
        size_up = new_size > existing_entry.size

        if price_changed or (size_up and self.modify_loses_priority):
            # Cancel + re-add at new price and end of queue
            self._on_cancel(side, existing_price, order_id)
            self._on_add(side, price, order_id, new_size, ts)
        else:
            # Size decrease (or size-up with priority preservation config)
            existing_entry.size = new_size

    def _on_fill(self, side: Side, price: float, filled_size: float) -> None:
        """Consume ``filled_size`` from the front of the level (FIFO).

        For pro-rata mode, distribute the consumption proportionally
        across all entries at the level. Virtual orders are treated
        identically to real ones — if a virtual order is at the front
        and gets hit, its entry size is reduced (caller may inspect
        via ``get_position()`` to detect the fill).
        """
        if filled_size <= 0:
            return
        level = self._level_for(side, price)
        if level is None:
            return

        if self.matching_algorithm == MatchingAlgorithm.FIFO:
            self._consume_fifo(level, filled_size)
        else:
            self._consume_pro_rata(level, filled_size)

        self._cleanup_empty_level(side, price)

    # ──────────────────────────────────────────────────────────────
    # Level helpers
    # ──────────────────────────────────────────────────────────────

    def _levels(self, side: Side) -> SortedDict:
        return self._bid_levels if side == Side.BUY else self._ask_levels

    def _level_for(self, side: Side, price: float) -> OrderedDict | None:
        """Return the level OrderedDict at ``price`` or ``None`` if empty."""
        # Quantize to tick to avoid FP lookup misses
        key = self._quantize(price)
        levels = self._levels(side)
        return levels.get(key)

    def _insert_entry_back(
        self, side: Side, price: float, entry: _Entry,
    ) -> None:
        key = self._quantize(price)
        levels = self._levels(side)
        level = levels.get(key)
        if level is None:
            level = OrderedDict()
            levels[key] = level
        # If the id already exists (shouldn't for well-formed feeds)
        # replace it at the same position rather than duplicating.
        level.pop(entry.order_id, None)
        level[entry.order_id] = entry

    def _cleanup_empty_level(self, side: Side, price: float) -> None:
        key = self._quantize(price)
        levels = self._levels(side)
        level = levels.get(key)
        if level is not None and len(level) == 0:
            del levels[key]

    def _locate_entry(
        self, side: Side, order_id: str,
    ) -> tuple[OrderedDict | None, float]:
        """Find the level containing ``order_id``.

        Scans all levels on the given side — O(L) worst case, but for
        active orders the MBO feed usually includes the original price
        in the MODIFY record, so this path is only reached for edge
        cases (feed glitches).
        """
        levels = self._levels(side)
        for price, level in levels.items():
            if order_id in level:
                return level, price
        return None, 0.0

    def _quantize(self, price: float) -> float:
        """Snap price to nearest tick to produce consistent dict keys."""
        if self.tick_size <= 0:
            return price
        return round(price / self.tick_size) * self.tick_size

    # ──────────────────────────────────────────────────────────────
    # FIFO / pro-rata consumption
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _consume_fifo(level: OrderedDict, filled_size: float) -> None:
        """Consume size from front of FIFO queue.

        Removes fully-consumed entries; reduces size of partially
        consumed front entry.
        """
        remaining = filled_size
        to_remove: list[str] = []
        for oid, entry in level.items():
            if remaining <= 0:
                break
            if entry.size <= remaining:
                remaining -= entry.size
                to_remove.append(oid)
            else:
                entry.size -= remaining
                remaining = 0.0
        for oid in to_remove:
            del level[oid]

    @staticmethod
    def _consume_pro_rata(level: OrderedDict, filled_size: float) -> None:
        """Distribute consumption proportionally across all entries.

        Each entry's share = filled_size × (entry.size / total_at_level).
        Rounds down to avoid floating-point residues exceeding 0.
        """
        total = sum(e.size for e in level.values())
        if total <= 0:
            return
        remaining = min(filled_size, total)
        to_remove: list[str] = []
        for oid, entry in level.items():
            share = remaining * (entry.size / total)
            if share >= entry.size:
                to_remove.append(oid)
            else:
                entry.size -= share
        for oid in to_remove:
            del level[oid]

    # ──────────────────────────────────────────────────────────────
    # Event parsing helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _action_name(event: Any) -> str:
        action = getattr(event, "action", None)
        if action is None:
            return ""
        name = getattr(action, "name", None)
        return str(name).upper() if name is not None else str(action).upper()

    @staticmethod
    def _side_from_event(event: Any) -> Side | None:
        side = getattr(event, "side", None)
        if side is None:
            return None
        name = getattr(side, "name", None)
        side_str = str(name).upper() if name is not None else str(side).upper()
        if side_str in ("BID", "BUY"):
            return Side.BUY
        if side_str in ("ASK", "SELL"):
            return Side.SELL
        return None

    # ──────────────────────────────────────────────────────────────
    # Fill probability estimator
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fill_probability(
        position_from_front: float, total_at_level: float,
    ) -> float:
        """Rough P(fill) estimate from queue position.

        Uses a linear decay: an order at the front (position=0) gets
        1.0; an order at the back (position=total-size) gets ~0.
        This is a first-order approximation; the full fill_simulator
        uses empirical replay-based estimates.
        """
        if total_at_level <= 0:
            return 0.0
        position = max(0.0, min(position_from_front, total_at_level))
        return 1.0 - (position / total_at_level)

    # ──────────────────────────────────────────────────────────────
    # Inspection helpers (useful for tests + sanity checks)
    # ──────────────────────────────────────────────────────────────

    def level_sizes(self, side: Side) -> dict[float, float]:
        """Map of price → total resting size at that level."""
        levels = self._levels(side)
        return {p: sum(e.size for e in lvl.values()) for p, lvl in levels.items()}

    def level_order_count(self, side: Side) -> dict[float, int]:
        """Map of price → number of distinct orders at that level."""
        levels = self._levels(side)
        return {p: len(lvl) for p, lvl in levels.items()}

    def reset(self) -> None:
        """Clear all state — useful between replay files."""
        self._bid_levels = SortedDict()
        self._ask_levels = SortedDict()
        self._handles.clear()
        self._next_handle_id = 0
        self._next_virtual_id = 0
