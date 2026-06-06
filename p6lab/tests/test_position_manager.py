"""Tests for PositionManager (Wave 5 Phase 5D)."""
from __future__ import annotations

import pytest

from p6lab.risk.position_manager import PositionLimits, PositionManager


class _ManualClock:
    def __init__(self, start_ms: int = 0) -> None:
        self.now_ms = int(start_ms)

    def __call__(self) -> int:
        return self.now_ms

    def advance(self, ms: int) -> None:
        self.now_ms += int(ms)


# ---------------------------------------------------------------------------
# Approvals + rejections
# ---------------------------------------------------------------------------


def test_can_open_accepts_fresh_pattern() -> None:
    pm = PositionManager()
    ok, reason = pm.can_open(pattern_id="p", symbol="NQ", quantity=1)
    assert ok is True
    assert reason == ""


def test_can_open_rejects_non_positive_quantity() -> None:
    pm = PositionManager()
    ok, _ = pm.can_open(pattern_id="p", symbol="NQ", quantity=0)
    assert ok is False
    ok2, _ = pm.can_open(pattern_id="p", symbol="NQ", quantity=-3)
    assert ok2 is False


def test_can_open_enforces_per_pattern_throttle() -> None:
    pm = PositionManager(PositionLimits(max_open_orders_per_pattern=1))
    pm.on_submit(pattern_id="p", symbol="NQ", side="BUY", quantity=1)
    ok, reason = pm.can_open(pattern_id="p", symbol="NQ", quantity=1)
    assert ok is False
    assert "pattern" in reason


def test_can_open_allows_other_patterns_while_first_open() -> None:
    pm = PositionManager(PositionLimits(max_open_orders_per_pattern=1))
    pm.on_submit(pattern_id="p1", symbol="NQ", side="BUY", quantity=1)
    ok, _ = pm.can_open(pattern_id="p2", symbol="NQ", quantity=1)
    assert ok is True


def test_can_open_enforces_instrument_exposure_cap() -> None:
    pm = PositionManager(PositionLimits(
        max_open_orders_per_pattern=10,
        max_contracts_per_instrument=5,
    ))
    pm.on_submit(pattern_id="p1", symbol="NQ", side="BUY", quantity=4)
    ok, reason = pm.can_open(pattern_id="p2", symbol="NQ", quantity=2)
    assert ok is False
    assert "exposure" in reason


def test_exposure_is_signed() -> None:
    pm = PositionManager()
    pm.on_submit(pattern_id="p", symbol="NQ", side="BUY", quantity=3)
    assert pm.exposure("NQ") == 3
    pm.on_submit(pattern_id="p2", symbol="NQ", side="SELL", quantity=1)
    assert pm.exposure("NQ") == 2
    assert pm.exposure("ES") == 0


def test_can_open_enforces_concurrent_position_cap() -> None:
    pm = PositionManager(PositionLimits(
        max_open_orders_per_pattern=10,
        max_concurrent_positions=2,
        max_contracts_per_instrument=99,
    ))
    pm.on_submit(pattern_id="a", symbol="NQ", side="BUY", quantity=1)
    pm.on_submit(pattern_id="b", symbol="ES", side="BUY", quantity=1)
    ok, reason = pm.can_open(pattern_id="c", symbol="YM", quantity=1)
    assert ok is False
    assert "open positions" in reason


# ---------------------------------------------------------------------------
# P&L + circuit breaker
# ---------------------------------------------------------------------------


def test_on_exit_realizes_pnl_for_buy() -> None:
    pm = PositionManager()
    pm.on_submit(pattern_id="p", symbol="NQ", side="BUY", quantity=2)
    pm.on_fill(pattern_id="p", symbol="NQ", fill_price=20_000.0)
    pnl = pm.on_exit(pattern_id="p", symbol="NQ", exit_price=20_010.0)
    assert pnl == pytest.approx(20.0)   # (20010 - 20000) × +1 × 2
    assert pm.exposure("NQ") == 0
    assert pm.open_positions() == 0


def test_on_exit_realizes_pnl_for_sell() -> None:
    pm = PositionManager()
    pm.on_submit(pattern_id="p", symbol="NQ", side="SELL", quantity=1)
    pm.on_fill(pattern_id="p", symbol="NQ", fill_price=20_000.0)
    pnl = pm.on_exit(pattern_id="p", symbol="NQ", exit_price=19_990.0)
    assert pnl == pytest.approx(10.0)


def test_on_exit_without_entry_price_returns_zero() -> None:
    pm = PositionManager()
    pm.on_submit(pattern_id="p", symbol="NQ", side="BUY", quantity=1)
    pnl = pm.on_exit(pattern_id="p", symbol="NQ", exit_price=20_000.0)
    assert pnl == 0.0
    assert pm.open_positions() == 0


def test_on_exit_handles_unknown_pattern_gracefully() -> None:
    pm = PositionManager()
    pnl = pm.on_exit(pattern_id="ghost", symbol="NQ", exit_price=1.0)
    assert pnl == 0.0


def test_circuit_breaker_trips_on_daily_loss() -> None:
    pm = PositionManager(PositionLimits(
        max_contracts_per_instrument=99,
        daily_loss_circuit_breaker=100.0,
    ))
    pm.on_submit(pattern_id="p", symbol="NQ", side="BUY", quantity=10)
    pm.on_fill(pattern_id="p", symbol="NQ", fill_price=20_000.0)
    pm.on_exit(pattern_id="p", symbol="NQ", exit_price=19_985.0)  # −150 pnl
    assert pm.halted is True
    ok, reason = pm.can_open(pattern_id="new", symbol="NQ", quantity=1)
    assert ok is False
    assert "halted" in reason


def test_circuit_breaker_does_not_trip_under_threshold() -> None:
    pm = PositionManager(PositionLimits(daily_loss_circuit_breaker=500.0))
    pm.on_submit(pattern_id="p", symbol="NQ", side="BUY", quantity=1)
    pm.on_fill(pattern_id="p", symbol="NQ", fill_price=20_000.0)
    pm.on_exit(pattern_id="p", symbol="NQ", exit_price=19_990.0)  # −10
    assert pm.halted is False
    assert pm.daily_realized_pnl == pytest.approx(-10.0)


def test_reset_day_clears_halt() -> None:
    pm = PositionManager(PositionLimits(daily_loss_circuit_breaker=10.0))
    pm.on_submit(pattern_id="p", symbol="NQ", side="BUY", quantity=1)
    pm.on_fill(pattern_id="p", symbol="NQ", fill_price=20_000.0)
    pm.on_exit(pattern_id="p", symbol="NQ", exit_price=19_950.0)  # −50
    assert pm.halted is True
    pm.reset_day()
    assert pm.halted is False
    assert pm.daily_realized_pnl == 0.0


def test_day_auto_rolls_via_clock() -> None:
    clock = _ManualClock()
    pm = PositionManager(
        PositionLimits(
            daily_loss_circuit_breaker=10.0,
            trading_day_length_ms=1_000,
        ),
        clock=clock,
    )
    pm.on_submit(pattern_id="p", symbol="NQ", side="BUY", quantity=1)
    pm.on_fill(pattern_id="p", symbol="NQ", fill_price=20_000.0)
    pm.on_exit(pattern_id="p", symbol="NQ", exit_price=19_950.0)  # −50
    assert pm.halted is True
    clock.advance(2_000)  # next day
    # halted should auto-clear once the rolling window matures
    assert pm.halted is False
    assert pm.daily_realized_pnl == 0.0


def test_exposure_zeros_out_after_round_trip() -> None:
    pm = PositionManager()
    pm.on_submit(pattern_id="p", symbol="NQ", side="BUY", quantity=1)
    pm.on_fill(pattern_id="p", symbol="NQ", fill_price=100.0)
    pm.on_exit(pattern_id="p", symbol="NQ", exit_price=101.0)
    assert pm.exposure("NQ") == 0
