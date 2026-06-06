"""
Unit tests for ``p6lab.correlation.match_broker.MatchBroker``.

Covers:
- multi-subscriber delivery
- idempotent subscribe (no double-delivery on re-registration)
- unsubscribe returns bool + removes from future dispatch
- exception in one subscriber doesn't poison the others
- subscriber_count reflects current state
- clear() empties the bus
- thread-safe concurrent emit (smoke: no crash, no lost events from the
  perspective of a single-subscriber counter)
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest
from p6lab.correlation.match_broker import MatchBroker


def test_multi_subscriber_delivery():
    bus: MatchBroker = MatchBroker()
    a, b = [], []
    bus.subscribe(a.append)
    bus.subscribe(b.append)
    assert bus.subscriber_count == 2

    delivered = bus.emit({"id": 1})
    assert delivered == 2
    assert a == [{"id": 1}]
    assert b == [{"id": 1}]


def test_idempotent_subscribe():
    bus = MatchBroker()
    seen: list = []
    bus.subscribe(seen.append)
    bus.subscribe(seen.append)     # duplicate — should be a no-op
    assert bus.subscriber_count == 1
    bus.emit("x")
    assert seen == ["x"]           # not ["x", "x"]


def test_unsubscribe():
    bus = MatchBroker()
    captured: list = []
    cb = captured.append
    bus.subscribe(cb)
    assert bus.unsubscribe(cb) is True
    assert bus.subscriber_count == 0
    assert bus.unsubscribe(cb) is False   # already gone

    bus.emit("y")
    assert captured == []          # nothing delivered after unsubscribe


def test_exception_in_subscriber_does_not_break_bus(caplog):
    bus = MatchBroker()
    good: list = []

    def bad(_event):
        raise RuntimeError("subscriber blew up")

    bus.subscribe(bad)
    bus.subscribe(good.append)

    delivered = bus.emit("evt")
    # `bad` raised, `good` still delivered — delivered counter is 1
    assert delivered == 1
    assert good == ["evt"]


def test_clear():
    bus = MatchBroker()
    bus.subscribe(lambda _: None)
    bus.subscribe(lambda _: None)
    assert bus.subscriber_count == 2
    bus.clear()
    assert bus.subscriber_count == 0
    assert bus.emit("ignored") == 0


def test_thread_safe_concurrent_emit():
    """Fire N events from M threads into one subscriber; confirm all land."""
    bus = MatchBroker()
    N_EVENTS, N_THREADS = 200, 8
    received: list = []
    lock = threading.Lock()

    def subscriber(ev):
        with lock:
            received.append(ev)

    bus.subscribe(subscriber)

    def producer(tid: int):
        for i in range(N_EVENTS):
            bus.emit((tid, i))

    threads = [threading.Thread(target=producer, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(received) == N_EVENTS * N_THREADS


def test_unsubscribe_during_emit_snapshots_subscribers():
    """Snapshot-before-dispatch semantics: a subscriber that removes itself
    during emit() still receives the in-flight event; new subscribers added
    during dispatch only see subsequent emits."""
    bus = MatchBroker()
    events_a, events_b = [], []

    def a(ev):
        events_a.append(ev)
        bus.unsubscribe(a)

    bus.subscribe(a)
    bus.subscribe(events_b.append)

    bus.emit("first")
    bus.emit("second")

    assert events_a == ["first"]             # a removed after first
    assert events_b == ["first", "second"]   # b stuck around
