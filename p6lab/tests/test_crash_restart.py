"""Wave 8.5-E — crash-restart regression for SQLite persistence layer.

Simulates an unexpected process exit mid-flight by:
1. Constructing a fresh `StateStore` on disk.
2. Driving a `PositionManager` + `OutcomeTrackerRenderer` through live-
   trading-like state transitions (submits, fills, matches, exits).
3. Dropping all in-memory refs (imitating a process crash).
4. Reconstructing both via `from_state_store(...)` against the same
   SQLite file.
5. Asserting every counter / field matches the pre-drop state.

This is the gate that makes Wave 8 (real broker) viable — live trading
must survive an OS-level crash without losing open positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from p6lab.correlation.renderers.outcome_tracker import OutcomeTrackerRenderer
from p6lab.persistence import StateStore
from p6lab.risk.position_manager import PositionLimits, PositionManager


@dataclass
class _FakeMatch:
    pattern_id: str
    expected_direction: str
    expected_move_atr: float = 1.0
    match_window_end_ms: int = 0
    instrument: str = "NQ"
    confidence_tier: str = "B"
    regime: str = "normal"
    ensemble_score: float = 0.75


# ---------------------------------------------------------------------------
# PositionManager crash-restart
# ---------------------------------------------------------------------------


def test_wave_85_e_position_manager_crash_restart(tmp_path: Path) -> None:
    """Drive submits + fills, drop refs, reconstruct, verify state."""
    db = tmp_path / "pm.db"
    store = StateStore(db)
    pm = PositionManager(
        PositionLimits(max_contracts_per_instrument=10),
        state_store=store,
    )
    pm.on_submit(pattern_id="a", symbol="NQ", side="BUY", quantity=2)
    pm.on_submit(pattern_id="b", symbol="ES", side="SELL", quantity=1)
    pm.on_fill(pattern_id="a", symbol="NQ", fill_price=20_000.0)
    # Pre-drop snapshot
    pre_open = pm.open_positions()
    pre_nq = pm.exposure("NQ")
    pre_es = pm.exposure("ES")
    assert pre_open == 2
    assert pre_nq == 2
    assert pre_es == -1

    # Simulate crash: drop in-memory refs; store persists
    del pm
    store.close()
    store2 = StateStore(db)

    # Reconstruct
    pm2 = PositionManager.from_state_store(
        store2,
        PositionLimits(max_contracts_per_instrument=10),
    )
    assert pm2.open_positions() == pre_open
    assert pm2.exposure("NQ") == pre_nq
    assert pm2.exposure("ES") == pre_es
    # The NQ position must still carry its filled entry price
    nq_positions = pm2._open_by_pattern["a"]
    assert nq_positions[0].entry_price == 20_000.0
    # The ES position did not get a fill; entry_price should be None
    es_positions = pm2._open_by_pattern["b"]
    assert es_positions[0].entry_price is None
    store2.close()


def test_wave_85_e_position_manager_pnl_restart(tmp_path: Path) -> None:
    """Realized P&L and halt flag must survive restart."""
    db = tmp_path / "pm_pnl.db"
    store = StateStore(db)
    pm = PositionManager(
        PositionLimits(
            max_contracts_per_instrument=10,
            daily_loss_circuit_breaker=50.0,
        ),
        state_store=store,
    )
    pm.on_submit(pattern_id="a", symbol="NQ", side="BUY", quantity=1)
    pm.on_fill(pattern_id="a", symbol="NQ", fill_price=20_000.0)
    pm.on_exit(pattern_id="a", symbol="NQ", exit_price=19_900.0)  # −100
    assert pm.halted is True
    pre_pnl = pm.daily_realized_pnl

    # Crash + restart
    del pm
    store.close()
    store2 = StateStore(db)
    pm2 = PositionManager.from_state_store(
        store2,
        PositionLimits(
            max_contracts_per_instrument=10,
            daily_loss_circuit_breaker=50.0,
        ),
    )
    assert pm2.daily_realized_pnl == pytest.approx(pre_pnl)
    assert pm2.halted is True
    store2.close()


# ---------------------------------------------------------------------------
# OutcomeTracker crash-restart
# ---------------------------------------------------------------------------


def test_wave_85_e_outcome_tracker_crash_restart(tmp_path: Path) -> None:
    """Pending outcomes + aggregate history must persist across restart."""
    db = tmp_path / "ot.db"
    outcomes_jsonl = tmp_path / "outcomes.jsonl"
    store = StateStore(db)

    tracker = OutcomeTrackerRenderer(
        outcomes_jsonl,
        horizon_ms=1_000,
        state_store=store,
    )
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    # Three matches; first two will close on next price update, third is pending
    tracker(_FakeMatch("p_a", "bull", match_window_end_ms=0))
    tracker(_FakeMatch("p_a", "bull", match_window_end_ms=100))
    tracker(_FakeMatch("p_b", "bull", match_window_end_ms=5_000))  # exit later
    tracker.on_price("NQ", mid=20_010.0, ts_ms=2_000)   # closes first two

    pre_pending = tracker.pending_count
    pre_aggregates_p_a = len(tracker._aggregates["p_a"].returns)
    assert pre_pending == 1
    assert pre_aggregates_p_a == 2

    # Crash + restart
    del tracker
    store.close()
    store2 = StateStore(db)

    tracker2 = OutcomeTrackerRenderer.from_state_store(
        store2,
        outcomes_path=outcomes_jsonl,
        horizon_ms=1_000,
    )
    assert tracker2.pending_count == pre_pending
    assert "p_a" in tracker2._aggregates
    assert len(tracker2._aggregates["p_a"].returns) == pre_aggregates_p_a
    store2.close()


def test_wave_85_e_outcome_tracker_in_memory_mode_unchanged(tmp_path: Path) -> None:
    """Backward-compat: state_store=None preserves pre-8.5-E semantics."""
    outcomes_jsonl = tmp_path / "outcomes.jsonl"
    tracker = OutcomeTrackerRenderer(outcomes_jsonl, horizon_ms=1_000)
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker(_FakeMatch("p", "bull", match_window_end_ms=0))
    tracker.on_price("NQ", mid=20_010.0, ts_ms=2_000)
    assert tracker.outcomes_closed == 1


# ---------------------------------------------------------------------------
# Cross-component crash-restart
# ---------------------------------------------------------------------------


def test_wave_85_e_full_stack_crash_restart(tmp_path: Path) -> None:
    """PositionManager + OutcomeTracker share a StateStore; both must
    reconstruct independently with correct state."""
    db = tmp_path / "stack.db"
    outcomes_jsonl = tmp_path / "outcomes.jsonl"
    store = StateStore(db)
    pm = PositionManager(state_store=store)
    tracker = OutcomeTrackerRenderer(outcomes_jsonl, state_store=store)

    # Drive some state
    pm.on_submit(pattern_id="a", symbol="NQ", side="BUY", quantity=1)
    pm.on_fill(pattern_id="a", symbol="NQ", fill_price=20_000.0)
    tracker.on_price("NQ", mid=20_000.0, ts_ms=0)
    tracker(_FakeMatch("a", "bull", match_window_end_ms=0))

    # Snapshot expectations
    pre_positions = pm.open_positions()
    pre_pending = tracker.pending_count

    # Crash + restart
    del pm
    del tracker
    store.close()
    store2 = StateStore(db)

    pm2 = PositionManager.from_state_store(store2)
    tracker2 = OutcomeTrackerRenderer.from_state_store(
        store2, outcomes_path=outcomes_jsonl,
    )
    assert pm2.open_positions() == pre_positions
    assert tracker2.pending_count == pre_pending
    # Both continue accepting new events on the restored store
    pm2.on_submit(pattern_id="b", symbol="ES", side="BUY", quantity=1)
    assert pm2.open_positions() == pre_positions + 1
    store2.close()
