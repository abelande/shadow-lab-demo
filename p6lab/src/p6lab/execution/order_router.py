"""
p6lab.execution.order_router — Wave 5 Phase 5D

Abstract ``BrokerClient`` interface + ``NoopBroker`` scaffold so Wave 5 can
close the loop from PatternMatch → OrderRequest → OrderAck without
touching a real broker yet. Real broker integrations (Tradovate, IB,
Rithmic, CME iLink) plug in later by implementing the same Protocol.

Design rules
------------
- **No side effects.** ``NoopBroker`` logs and returns ACCEPTED acks; never
  touches a network, filesystem (unless ``log_path`` is given), or process
  state. Tests can drive the whole pipeline deterministically.
- **Traceable.** Every order carries a stable ``client_order_id`` derived
  from the pattern_id + entry timestamp so downstream outcome tracking can
  join orders back to matches.
- **Pre-trade gate.** ``OrderRouter.submit_from_match`` consults an
  optional ``PositionManager`` before every submission; rejections
  produce a REJECTED ack with the reason embedded, never a thrown
  exception.
"""
from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["MKT", "LMT"]
TimeInForce = Literal["DAY", "IOC", "GTC"]
OrderStatus = Literal["ACCEPTED", "REJECTED", "PENDING", "CANCELLED"]


@dataclass(frozen=True)
class OrderRequest:
    """One shot to a broker. Immutable — once created, callers keep the
    ``client_order_id`` around for subsequent cancel/fill joins."""
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = "MKT"
    limit_price: float | None = None
    time_in_force: TimeInForce = "DAY"
    pattern_id: str | None = None
    ensemble_score: float = 0.0
    submitted_at_ms: int = 0


@dataclass(frozen=True)
class OrderAck:
    """Broker's response. ``broker_order_id`` is an opaque venue string —
    we copy it verbatim, never try to parse it."""
    client_order_id: str
    broker_order_id: str
    status: OrderStatus
    reason: str = ""
    submitted_at_ms: int = 0


@runtime_checkable
class BrokerClient(Protocol):
    """Minimum broker surface.

    Real brokers will accrue more methods (streaming fills, position
    queries, modify, etc.) — the router only needs these two.
    """
    def submit_order(self, order: OrderRequest) -> OrderAck: ...
    def cancel_order(self, client_order_id: str) -> bool: ...


# ---------------------------------------------------------------------------
# NoopBroker
# ---------------------------------------------------------------------------


class NoopBroker:
    """Logs-only broker — safe to run in any environment.

    Parameters
    ----------
    log_path
        When supplied, each order + cancel is appended as a JSONL row so
        an audit trail exists even without stdout logging. Missing dirs
        are created.
    simulated_delay_ms
        If > 0, the broker sleeps this many ms in ``submit_order`` to let
        test harnesses observe the temporal gap between signal and ack.
    """

    def __init__(
        self,
        *,
        log_path: Path | str | None = None,
        simulated_delay_ms: int = 0,
    ) -> None:
        self.log_path = Path(log_path) if log_path else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.simulated_delay_ms = int(simulated_delay_ms)
        self._id_counter = itertools.count(1)
        self._lock = threading.Lock()
        self.orders_submitted: int = 0
        self.orders_cancelled: int = 0
        self._live_ids: set[str] = set()

    def submit_order(self, order: OrderRequest) -> OrderAck:
        if self.simulated_delay_ms > 0:
            time.sleep(self.simulated_delay_ms / 1000.0)
        with self._lock:
            broker_id = f"noop-{next(self._id_counter):06d}"
            self._live_ids.add(order.client_order_id)
            self.orders_submitted += 1
        ack = OrderAck(
            client_order_id=order.client_order_id,
            broker_order_id=broker_id,
            status="ACCEPTED",
            submitted_at_ms=order.submitted_at_ms or int(time.time() * 1000),
        )
        self._write_log({"event": "submit", "order": _dataclass_to_dict(order),
                         "ack": _dataclass_to_dict(ack)})
        logger.info(
            "NoopBroker: %s %s %d @ %s (pattern=%s, score=%.3f) -> %s",
            order.side, order.symbol, order.quantity,
            order.order_type if order.order_type == "MKT" else f"{order.limit_price}",
            order.pattern_id, order.ensemble_score, broker_id,
        )
        return ack

    def cancel_order(self, client_order_id: str) -> bool:
        with self._lock:
            if client_order_id not in self._live_ids:
                return False
            self._live_ids.remove(client_order_id)
            self.orders_cancelled += 1
        self._write_log({"event": "cancel", "client_order_id": client_order_id})
        return True

    def _write_log(self, row: dict) -> None:
        if self.log_path is None:
            return
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")


# ---------------------------------------------------------------------------
# OrderRouter
# ---------------------------------------------------------------------------


class OrderRouter:
    """Convert a ``PatternMatch`` into an ``OrderRequest`` and submit.

    The router is deliberately thin: it doesn't know about feeds, the
    outcome tracker, or the matcher — only pattern matches in and acks
    out. All the domain decisions (can we open? how large? is the loss
    cap tripped?) live on the optional ``PositionManager``.
    """

    def __init__(
        self,
        broker: BrokerClient,
        *,
        position_manager: Any = None,
        default_quantity: int = 1,
        tick_size: float = 0.25,
    ) -> None:
        self.broker = broker
        self.position_manager = position_manager
        self.default_quantity = int(default_quantity)
        self.tick_size = float(tick_size)
        self._id_counter = itertools.count(1)
        self._lock = threading.Lock()
        self.submissions_accepted: int = 0
        self.submissions_rejected: int = 0

    def submit_from_match(
        self,
        match: Any,
        *,
        quantity: int | None = None,
        reference_price: float | None = None,
        limit_offset_ticks: int | None = None,
    ) -> OrderAck:
        """Build + submit an order from a ``PatternMatch`` instance.

        ``reference_price`` is the live mid used when ``limit_offset_ticks``
        is given (builds an LMT order offset from mid). When both are
        omitted, the router submits a market order.
        """
        pattern_id = str(getattr(match, "pattern_id", "?"))
        symbol = str(getattr(match, "instrument", "?"))
        direction = str(getattr(match, "expected_direction", "neutral"))
        if direction == "neutral":
            return self._reject(
                pattern_id, symbol, direction,
                reason="neutral direction has no trade side",
            )

        side: OrderSide = "BUY" if direction == "bull" else "SELL"
        qty = int(quantity if quantity is not None else self.default_quantity)
        if qty <= 0:
            return self._reject(pattern_id, symbol, direction, reason="non-positive quantity")

        if self.position_manager is not None:
            approved, reason = self.position_manager.can_open(
                pattern_id=pattern_id, symbol=symbol, quantity=qty,
            )
            if not approved:
                return self._reject(pattern_id, symbol, direction, reason=reason)

        order_type: OrderType = "MKT"
        limit_price: float | None = None
        if reference_price is not None and limit_offset_ticks is not None:
            order_type = "LMT"
            offset = float(limit_offset_ticks) * self.tick_size
            # For BUY we want fills below mid; SELL above. Sign handled here.
            limit_price = reference_price + (offset if side == "SELL" else -offset)

        with self._lock:
            seq = next(self._id_counter)
        client_id = f"{pattern_id}-{seq:06d}"
        submitted_at = int(getattr(match, "match_window_end_ms", 0) or int(time.time() * 1000))

        req = OrderRequest(
            client_order_id=client_id,
            symbol=symbol,
            side=side,
            quantity=qty,
            order_type=order_type,
            limit_price=limit_price,
            pattern_id=pattern_id,
            ensemble_score=float(getattr(match, "ensemble_score", 0.0) or 0.0),
            submitted_at_ms=submitted_at,
        )
        try:
            ack = self.broker.submit_order(req)
        except Exception as exc:
            logger.exception("broker.submit_order raised for %s", client_id)
            return self._reject(pattern_id, symbol, direction, reason=f"broker exception: {exc}")

        if ack.status == "ACCEPTED":
            self.submissions_accepted += 1
            if self.position_manager is not None:
                self.position_manager.on_submit(
                    pattern_id=pattern_id, symbol=symbol,
                    side=side, quantity=qty,
                )
        else:
            self.submissions_rejected += 1
        return ack

    def _reject(
        self, pattern_id: str, symbol: str, direction: str, *, reason: str,
    ) -> OrderAck:
        self.submissions_rejected += 1
        logger.info(
            "OrderRouter REJECT pattern=%s symbol=%s direction=%s reason=%s",
            pattern_id, symbol, direction, reason,
        )
        return OrderAck(
            client_order_id=f"{pattern_id}-reject",
            broker_order_id="",
            status="REJECTED",
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return dict(obj)
