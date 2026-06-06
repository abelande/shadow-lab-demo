"""
p6lab.correlation.match_broker
==============================

Thin in-process pub/sub between ``CorrelationEngine.match()`` and the N
consumers that care about its output (WebSocket broadcaster, audit log
writer, live signal dock, future chart-level overlay).

Design goals
------------
- **Zero dependencies.** Pure stdlib — works anywhere the engine runs.
- **Thread-safe.** Subscribers can live on the asyncio server loop while
  the engine runs on a worker thread.
- **Fail-soft.** An exception in one subscriber never stops the others;
  broker only logs + swallows.
- **Decoupled.** The engine doesn't know who's listening. New consumers
  plug in via ``subscribe(callback)`` without engine changes.

Typical wiring
--------------
    broker = MatchBroker()

    # Consumers subscribe
    broker.subscribe(websocket_broadcaster.broadcast_match)
    broker.subscribe(audit_log.write_match)
    broker.subscribe(signal_dock_feed.push)

    # Engine emits
    for m in engine.match(l2_window, l1_window, ctx):
        broker.emit(m)
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
Subscriber = Callable[[T], None]


class MatchBroker:
    """In-process pub/sub for engine match events.

    The broker is intentionally generic over the event type — this keeps it
    reusable (correlation matches today, fragility updates tomorrow) and
    avoids a circular import with ``engine.py``.
    """

    __slots__ = ("_subscribers", "_lock")

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, callback: Subscriber) -> None:
        """Register *callback* to receive every subsequent ``emit`` event.

        Idempotent: subscribing the same callback twice is a no-op (prevents
        double-delivery when a caller accidentally re-registers).
        """
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Subscriber) -> bool:
        """Remove *callback*. Returns ``True`` if it was registered."""
        with self._lock:
            try:
                self._subscribers.remove(callback)
                return True
            except ValueError:
                return False

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def emit(self, event: T) -> int:
        """Deliver *event* to every current subscriber.

        Returns the number of subscribers that handled the event without
        raising. Exceptions are caught + logged so one bad consumer can't
        break the bus.
        """
        # Snapshot under lock, dispatch outside lock — prevents a long-running
        # subscriber from blocking subscribe/unsubscribe on other threads.
        with self._lock:
            subs = tuple(self._subscribers)

        delivered = 0
        for cb in subs:
            try:
                cb(event)
                delivered += 1
            except Exception:
                logger.exception(
                    "MatchBroker: subscriber %r raised; continuing",
                    getattr(cb, "__qualname__", repr(cb)),
                )
        return delivered

    def clear(self) -> None:
        """Drop all subscribers. Intended for test teardown."""
        with self._lock:
            self._subscribers.clear()
