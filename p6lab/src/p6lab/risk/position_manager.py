"""
p6lab.risk.position_manager — Wave 5 Phase 5D

Intraday position + risk book. The router calls the manager on every
``submit_from_match``; the outcome tracker (or a broker fill subscriber)
calls ``on_fill`` / ``on_exit`` as state changes. The manager owns four
pre-trade gates:

  1. **one-entry-per-pattern throttle** — a pattern may have at most
     ``max_open_orders_per_pattern`` live entries at once.
  2. **per-instrument exposure cap** — the sum of ``|open_quantity|``
     across all positions on a symbol ≤ ``max_contracts_per_instrument``.
  3. **concurrent-positions cap** — total live positions across all
     symbols ≤ ``max_concurrent_positions``.
  4. **daily-loss circuit breaker** — once realized P&L for the trading
     day drops below ``-daily_loss_circuit_breaker``, the manager halts
     and rejects every subsequent ``can_open`` call until
     ``reset_day()`` is called manually (or the clock rolls into a
     new trading day).

Design rules
------------
- **Pure Python, no external deps.** Keeps import overhead and test
  surface small.
- **Deterministic clock.** The manager takes an optional ``clock``
  callable returning the current epoch-ms so tests can advance time
  without sleeping.
- **Fail-safe.** ``can_open`` never raises — it always returns
  ``(bool, str)``. Routers and tests treat the string as a rejection
  reason for logging.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)


OrderSide = Literal["BUY", "SELL"]


@dataclass
class PositionLimits:
    """Risk parameters for the intraday book."""
    max_open_orders_per_pattern: int = 1
    max_contracts_per_instrument: int = 5
    max_concurrent_positions: int = 10
    daily_loss_circuit_breaker: float = 500.0
    trading_day_length_ms: int = 24 * 60 * 60 * 1000


@dataclass
class _OpenPosition:
    pattern_id: str
    symbol: str
    side: OrderSide
    quantity: int
    entry_price: float | None = None
    # Wave 8.5-E: persistence link — when a StateStore is attached,
    # pos_id carries the SQLite primary key so updates + removals stay
    # in sync. None for pure in-memory mode (backward compat).
    pos_id: int | None = None


class PositionManager:
    """Thread-safe intraday position + risk bookkeeper."""

    def __init__(
        self,
        limits: PositionLimits | None = None,
        *,
        clock: Callable[[], int] | None = None,
        state_store: Any = None,
    ) -> None:
        self.limits = limits or PositionLimits()
        self._clock = clock or (lambda: int(_monotonic_ms()))
        self._lock = threading.RLock()
        self._open_by_pattern: dict[str, list[_OpenPosition]] = {}
        self._net_by_symbol: dict[str, int] = {}
        self._daily_realized_pnl: float = 0.0
        self._day_start_ms: int = self._clock()
        self._halted: bool = False
        # Wave 8.5-E: optional SQLite state store. Default None preserves
        # legacy in-memory behavior — zero-regression migration.
        self._state_store = state_store

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    @property
    def halted(self) -> bool:
        with self._lock:
            self._maybe_roll_day()
            return self._halted

    @property
    def daily_realized_pnl(self) -> float:
        with self._lock:
            self._maybe_roll_day()
            return float(self._daily_realized_pnl)

    def open_positions(self, pattern_id: str | None = None) -> int:
        """Number of live positions, optionally filtered to one pattern_id."""
        with self._lock:
            if pattern_id is None:
                return sum(len(v) for v in self._open_by_pattern.values())
            return len(self._open_by_pattern.get(pattern_id, []))

    def exposure(self, symbol: str) -> int:
        """Signed net exposure on ``symbol``. BUYs +qty, SELLs -qty."""
        with self._lock:
            return int(self._net_by_symbol.get(symbol, 0))

    # ------------------------------------------------------------------
    # Pre-trade gate
    # ------------------------------------------------------------------

    def can_open(
        self,
        *,
        pattern_id: str,
        symbol: str,
        quantity: int,
    ) -> tuple[bool, str]:
        """Decide whether the router may submit the order.

        Returns
        -------
        (approved, reason)
            ``approved`` True means the router should submit.
            ``reason`` is empty on approval; otherwise a short
            human-readable explanation for audit + UI.
        """
        with self._lock:
            self._maybe_roll_day()
            if self._halted:
                return False, "halted: daily loss circuit breaker tripped"
            if quantity <= 0:
                return False, "non-positive quantity"
            per_pat = len(self._open_by_pattern.get(pattern_id, []))
            if per_pat >= self.limits.max_open_orders_per_pattern:
                return (
                    False,
                    f"pattern {pattern_id} already has {per_pat} open orders "
                    f"(cap {self.limits.max_open_orders_per_pattern})",
                )
            concurrent = sum(len(v) for v in self._open_by_pattern.values())
            if concurrent >= self.limits.max_concurrent_positions:
                return (
                    False,
                    f"{concurrent} open positions (cap "
                    f"{self.limits.max_concurrent_positions})",
                )
            net = abs(self._net_by_symbol.get(symbol, 0)) + quantity
            if net > self.limits.max_contracts_per_instrument:
                return (
                    False,
                    f"{symbol} exposure would reach {net} "
                    f"(cap {self.limits.max_contracts_per_instrument})",
                )
        return True, ""

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def on_submit(
        self,
        *,
        pattern_id: str,
        symbol: str,
        side: OrderSide,
        quantity: int,
    ) -> None:
        """Record an order once the broker ACCEPTs it."""
        with self._lock:
            pos = _OpenPosition(
                pattern_id=pattern_id,
                symbol=symbol,
                side=side,
                quantity=int(quantity),
            )
            # Wave 8.5-E: persist before appending so a mid-op crash
            # can't leave in-memory state ahead of durable state.
            if self._state_store is not None:
                pos.pos_id = self._state_store.put_open_position({
                    "pattern_id": pattern_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": int(quantity),
                    "entry_price": None,
                    "opened_at_ms": int(self._clock()),
                })
            self._open_by_pattern.setdefault(pattern_id, []).append(pos)
            delta = quantity if side == "BUY" else -quantity
            self._net_by_symbol[symbol] = self._net_by_symbol.get(symbol, 0) + delta

    def on_fill(
        self,
        *,
        pattern_id: str,
        symbol: str,
        fill_price: float,
    ) -> None:
        """Attach an entry price to the most-recent open position for the
        given pattern_id. Silently no-ops if no matching open entry
        exists (late fill / duplicate fill)."""
        with self._lock:
            positions = self._open_by_pattern.get(pattern_id, [])
            for pos in reversed(positions):
                if pos.symbol == symbol and pos.entry_price is None:
                    pos.entry_price = float(fill_price)
                    # Wave 8.5-E: mirror to SQLite if persistence attached.
                    if self._state_store is not None and pos.pos_id is not None:
                        self._state_store.update_position_entry_price(
                            pos_id=pos.pos_id, entry_price=float(fill_price),
                        )
                    return

    def on_exit(
        self,
        *,
        pattern_id: str,
        symbol: str,
        exit_price: float,
        contract_multiplier: float = 1.0,
    ) -> float:
        """Close the oldest open position for ``pattern_id`` on ``symbol``
        and record realized P&L. Returns the realized P&L (may be 0 if
        no matching open position)."""
        with self._lock:
            self._maybe_roll_day()
            positions = self._open_by_pattern.get(pattern_id, [])
            for i, pos in enumerate(positions):
                if pos.symbol != symbol:
                    continue
                entry = pos.entry_price
                realized = 0.0
                if entry is not None:
                    sign = 1.0 if pos.side == "BUY" else -1.0
                    realized = (float(exit_price) - entry) * sign * pos.quantity * contract_multiplier
                    self._daily_realized_pnl += realized
                    if self._daily_realized_pnl <= -self.limits.daily_loss_circuit_breaker:
                        self._halted = True
                        logger.warning(
                            "position_manager: CIRCUIT BREAKER tripped "
                            "(daily_pnl=%.2f ≤ -%.2f)",
                            self._daily_realized_pnl,
                            self.limits.daily_loss_circuit_breaker,
                        )
                # Wave 8.5-E: remove from SQLite before dropping from memory.
                if self._state_store is not None and pos.pos_id is not None:
                    self._state_store.remove_open_position(pos.pos_id)
                # Drop the position whether or not an entry was recorded
                positions.pop(i)
                if not positions:
                    self._open_by_pattern.pop(pattern_id, None)
                delta = -pos.quantity if pos.side == "BUY" else pos.quantity
                self._net_by_symbol[symbol] = self._net_by_symbol.get(symbol, 0) + delta
                if self._net_by_symbol[symbol] == 0:
                    self._net_by_symbol.pop(symbol)
                # Wave 8.5-E: persist daily-P&L snapshot so day-roll state
                # survives restart.
                if self._state_store is not None:
                    self._state_store.put_counter_snapshot(
                        "position_manager.pnl",
                        {
                            "daily_realized_pnl": self._daily_realized_pnl,
                            "day_start_ms": self._day_start_ms,
                            "halted": self._halted,
                        },
                    )
                return float(realized)
        return 0.0

    # ------------------------------------------------------------------
    # Wave 8.5-E: persistence reconstruction
    # ------------------------------------------------------------------

    @classmethod
    def from_state_store(
        cls,
        state_store: Any,
        limits: PositionLimits | None = None,
        *,
        clock: Callable[[], int] | None = None,
    ) -> "PositionManager":
        """Reconstruct a PositionManager from persisted rows.

        Reloads open positions, net-by-symbol exposure, daily P&L, halt
        flag, and day-start timestamp. New instance is bound to the
        same state_store for continued persistence.
        """
        pm = cls(limits=limits, clock=clock, state_store=state_store)
        with pm._lock:
            # Open positions
            for row in state_store.load_open_positions():
                pos = _OpenPosition(
                    pattern_id=row["pattern_id"],
                    symbol=row["symbol"],
                    side=row["side"],
                    quantity=int(row["quantity"]),
                    entry_price=row["entry_price"],
                    pos_id=int(row["pos_id"]),
                )
                pm._open_by_pattern.setdefault(pos.pattern_id, []).append(pos)
                delta = pos.quantity if pos.side == "BUY" else -pos.quantity
                pm._net_by_symbol[pos.symbol] = pm._net_by_symbol.get(pos.symbol, 0) + delta
            # P&L snapshot
            pnl_snap = state_store.get_counter_snapshot("position_manager.pnl")
            if pnl_snap is not None:
                pm._daily_realized_pnl = float(pnl_snap.get("daily_realized_pnl", 0.0))
                pm._day_start_ms = int(pnl_snap.get("day_start_ms", pm._day_start_ms))
                pm._halted = bool(pnl_snap.get("halted", False))
        logger.info(
            "wave85-E position_manager reconstructed: %d open positions, "
            "pnl=%.2f, halted=%s",
            sum(len(v) for v in pm._open_by_pattern.values()),
            pm._daily_realized_pnl, pm._halted,
        )
        return pm

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def reset_day(self) -> None:
        """Reset daily P&L + uncoil the circuit breaker. Called on
        trading-day roll, by operators after a halt, or by tests."""
        with self._lock:
            self._daily_realized_pnl = 0.0
            self._halted = False
            self._day_start_ms = self._clock()

    def _maybe_roll_day(self) -> None:
        now = self._clock()
        if now - self._day_start_ms >= self.limits.trading_day_length_ms:
            self._daily_realized_pnl = 0.0
            self._halted = False
            self._day_start_ms = now


def _monotonic_ms() -> int:
    import time
    return int(time.monotonic() * 1000)
