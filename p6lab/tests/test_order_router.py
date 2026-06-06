"""Tests for OrderRouter + NoopBroker (Wave 5 Phase 5D)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from p6lab.execution.order_router import (
    BrokerClient,
    NoopBroker,
    OrderAck,
    OrderRequest,
    OrderRouter,
)


@dataclass
class _FakeMatch:
    pattern_id: str = "p1"
    expected_direction: str = "bull"
    expected_move_atr: float = 0.8
    ensemble_score: float = 0.82
    instrument: str = "NQ"
    match_window_end_ms: int = 1_700_000_000_100


class _StubBroker:
    """A broker that records every submit/cancel call without logging."""
    def __init__(self, *, reject_reason: str | None = None) -> None:
        self.submitted: list[OrderRequest] = []
        self.cancelled: list[str] = []
        self.reject_reason = reject_reason

    def submit_order(self, order: OrderRequest) -> OrderAck:
        self.submitted.append(order)
        if self.reject_reason is not None:
            return OrderAck(
                client_order_id=order.client_order_id,
                broker_order_id="",
                status="REJECTED",
                reason=self.reject_reason,
            )
        return OrderAck(
            client_order_id=order.client_order_id,
            broker_order_id=f"bkr-{len(self.submitted)}",
            status="ACCEPTED",
        )

    def cancel_order(self, client_order_id: str) -> bool:
        self.cancelled.append(client_order_id)
        return True


# ---------------------------------------------------------------------------
# NoopBroker
# ---------------------------------------------------------------------------


def test_noop_broker_accepts_orders() -> None:
    broker = NoopBroker()
    assert isinstance(broker, BrokerClient)
    req = OrderRequest(
        client_order_id="test-1", symbol="NQ", side="BUY", quantity=1,
    )
    ack = broker.submit_order(req)
    assert ack.status == "ACCEPTED"
    assert ack.broker_order_id.startswith("noop-")
    assert broker.orders_submitted == 1


def test_noop_broker_cancel_known() -> None:
    broker = NoopBroker()
    ack = broker.submit_order(OrderRequest(
        client_order_id="c1", symbol="NQ", side="BUY", quantity=1,
    ))
    assert broker.cancel_order("c1") is True
    assert broker.orders_cancelled == 1


def test_noop_broker_cancel_unknown_returns_false() -> None:
    broker = NoopBroker()
    assert broker.cancel_order("does-not-exist") is False


def test_noop_broker_log_file_written(tmp_path: Path) -> None:
    path = tmp_path / "orders.jsonl"
    broker = NoopBroker(log_path=path)
    broker.submit_order(OrderRequest(
        client_order_id="c1", symbol="NQ", side="BUY", quantity=1,
    ))
    broker.cancel_order("c1")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert {r["event"] for r in rows} == {"submit", "cancel"}


# ---------------------------------------------------------------------------
# OrderRouter
# ---------------------------------------------------------------------------


def test_router_submits_bull_match_as_buy() -> None:
    broker = _StubBroker()
    router = OrderRouter(broker)
    ack = router.submit_from_match(_FakeMatch(expected_direction="bull"))
    assert ack.status == "ACCEPTED"
    assert len(broker.submitted) == 1
    assert broker.submitted[0].side == "BUY"
    assert broker.submitted[0].quantity == 1
    assert broker.submitted[0].pattern_id == "p1"
    assert router.submissions_accepted == 1


def test_router_submits_bear_match_as_sell() -> None:
    broker = _StubBroker()
    router = OrderRouter(broker)
    ack = router.submit_from_match(_FakeMatch(expected_direction="bear"))
    assert ack.status == "ACCEPTED"
    assert broker.submitted[0].side == "SELL"


def test_router_rejects_neutral_direction() -> None:
    broker = _StubBroker()
    router = OrderRouter(broker)
    ack = router.submit_from_match(_FakeMatch(expected_direction="neutral"))
    assert ack.status == "REJECTED"
    assert "neutral" in ack.reason
    assert not broker.submitted


def test_router_rejects_non_positive_quantity() -> None:
    broker = _StubBroker()
    router = OrderRouter(broker)
    ack = router.submit_from_match(_FakeMatch(), quantity=0)
    assert ack.status == "REJECTED"
    assert not broker.submitted


def test_router_limit_order_from_reference_price() -> None:
    broker = _StubBroker()
    router = OrderRouter(broker, tick_size=0.25)
    ack = router.submit_from_match(
        _FakeMatch(expected_direction="bull"),
        reference_price=20_000.0,
        limit_offset_ticks=2,
    )
    assert ack.status == "ACCEPTED"
    req = broker.submitted[0]
    assert req.order_type == "LMT"
    # BUY LMT is offset below mid by the offset in ticks (2 ticks × 0.25)
    assert req.limit_price == pytest.approx(19_999.5)


def test_router_limit_order_sell_is_above_mid() -> None:
    broker = _StubBroker()
    router = OrderRouter(broker, tick_size=0.25)
    router.submit_from_match(
        _FakeMatch(expected_direction="bear"),
        reference_price=20_000.0,
        limit_offset_ticks=2,
    )
    req = broker.submitted[0]
    assert req.limit_price == pytest.approx(20_000.5)


def test_router_broker_rejection_passes_through() -> None:
    broker = _StubBroker(reject_reason="venue closed")
    router = OrderRouter(broker)
    ack = router.submit_from_match(_FakeMatch())
    assert ack.status == "REJECTED"
    assert router.submissions_rejected == 1


def test_router_broker_exception_produces_rejection() -> None:
    class _ExplodingBroker:
        def submit_order(self, order):
            raise RuntimeError("simulated disconnect")
        def cancel_order(self, _):
            return False
    router = OrderRouter(_ExplodingBroker())
    ack = router.submit_from_match(_FakeMatch())
    assert ack.status == "REJECTED"
    assert "simulated disconnect" in ack.reason


def test_router_consults_position_manager() -> None:
    class _PM:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int]] = []
        def can_open(self, *, pattern_id, symbol, quantity):
            self.calls.append((pattern_id, symbol, quantity))
            return False, "limit exceeded"
        def on_submit(self, **_kw):
            pytest.fail("on_submit should not run on rejection")
    pm = _PM()
    broker = _StubBroker()
    router = OrderRouter(broker, position_manager=pm)
    ack = router.submit_from_match(_FakeMatch())
    assert ack.status == "REJECTED"
    assert ack.reason == "limit exceeded"
    assert pm.calls == [("p1", "NQ", 1)]
    assert not broker.submitted


def test_router_notifies_position_manager_on_accept() -> None:
    class _PM:
        def __init__(self) -> None:
            self.submits: list[dict] = []
        def can_open(self, **_kw):
            return True, ""
        def on_submit(self, *, pattern_id, symbol, side, quantity):
            self.submits.append({
                "pattern_id": pattern_id, "symbol": symbol,
                "side": side, "quantity": quantity,
            })
    pm = _PM()
    router = OrderRouter(_StubBroker(), position_manager=pm)
    router.submit_from_match(_FakeMatch(expected_direction="bull"))
    assert len(pm.submits) == 1
    assert pm.submits[0]["side"] == "BUY"
