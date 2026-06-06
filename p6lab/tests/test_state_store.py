"""Tests for p6lab.persistence.state_store (Wave 8.5-E)."""
from __future__ import annotations

from pathlib import Path

import pytest

from p6lab.persistence import StateStore


@pytest.fixture
def store(tmp_path: Path):
    s = StateStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def mem_store():
    s = StateStore(":memory:")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Schema + lifecycle
# ---------------------------------------------------------------------------


def test_wave_85_e_init_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "foo.db"
    assert not path.exists()
    s = StateStore(path)
    assert path.exists()
    s.close()


def test_wave_85_e_init_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "state.db"
    s = StateStore(path)
    assert path.parent.exists()
    s.close()


def test_wave_85_e_in_memory_store(mem_store: StateStore) -> None:
    """In-memory store is useful in tests; no file created."""
    mem_store.put_open_position({
        "pattern_id": "p", "symbol": "NQ", "side": "BUY",
        "quantity": 1, "entry_price": 20000.0, "opened_at_ms": 0,
    })
    assert len(mem_store.load_open_positions()) == 1


def test_wave_85_e_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "ctx.db"
    with StateStore(path) as s:
        s.put_open_position({
            "pattern_id": "p", "symbol": "NQ", "side": "BUY",
            "quantity": 1, "entry_price": 20000.0, "opened_at_ms": 0,
        })
    # Reopen and verify persistence
    with StateStore(path) as s2:
        assert len(s2.load_open_positions()) == 1


# ---------------------------------------------------------------------------
# Open positions
# ---------------------------------------------------------------------------


def test_wave_85_e_open_position_roundtrip(store: StateStore) -> None:
    pid = store.put_open_position({
        "pattern_id": "bull_flag", "symbol": "NQ", "side": "BUY",
        "quantity": 2, "entry_price": 20_000.0, "opened_at_ms": 1_000,
    })
    rows = store.load_open_positions()
    assert len(rows) == 1
    r = rows[0]
    assert r["pos_id"] == pid
    assert r["pattern_id"] == "bull_flag"
    assert r["side"] == "BUY"
    assert r["quantity"] == 2
    assert r["entry_price"] == 20_000.0


def test_wave_85_e_open_position_null_entry_price(store: StateStore) -> None:
    """A position can be opened before the fill arrives; entry_price null."""
    pid = store.put_open_position({
        "pattern_id": "p", "symbol": "NQ", "side": "BUY",
        "quantity": 1, "entry_price": None, "opened_at_ms": 0,
    })
    rows = store.load_open_positions()
    assert rows[0]["entry_price"] is None
    store.update_position_entry_price(pos_id=pid, entry_price=20_000.0)
    assert store.load_open_positions()[0]["entry_price"] == 20_000.0


def test_wave_85_e_remove_position(store: StateStore) -> None:
    pid = store.put_open_position({
        "pattern_id": "p", "symbol": "NQ", "side": "BUY",
        "quantity": 1, "entry_price": 20_000.0, "opened_at_ms": 0,
    })
    assert len(store.load_open_positions()) == 1
    store.remove_open_position(pid)
    assert store.load_open_positions() == []


def test_wave_85_e_load_open_positions_ordered_oldest_first(store: StateStore) -> None:
    for i, ts in enumerate([3_000, 1_000, 2_000]):
        store.put_open_position({
            "pattern_id": f"p{i}", "symbol": "NQ", "side": "BUY",
            "quantity": 1, "entry_price": 20_000.0, "opened_at_ms": ts,
        })
    rows = store.load_open_positions()
    timestamps = [r["opened_at_ms"] for r in rows]
    assert timestamps == [1_000, 2_000, 3_000]


# ---------------------------------------------------------------------------
# Pending outcomes
# ---------------------------------------------------------------------------


def _pending_row(**overrides):
    base = dict(
        pattern_id="bull_flag", symbol="NQ",
        expected_direction="bull", expected_move_atr=0.8,
        entry_ts_ms=1_000, entry_mid=20_000.0,
        exit_ts_ms=2_000, confidence_tier="B", regime="normal",
    )
    base.update(overrides)
    return base


def test_wave_85_e_pending_outcome_roundtrip(store: StateStore) -> None:
    oid = store.put_pending_outcome(_pending_row())
    rows = store.load_pending_outcomes()
    assert len(rows) == 1
    assert rows[0]["outcome_id"] == oid
    assert rows[0]["expected_direction"] == "bull"
    assert rows[0]["entry_mid"] == 20_000.0


def test_wave_85_e_pending_outcome_remove(store: StateStore) -> None:
    oid = store.put_pending_outcome(_pending_row())
    store.remove_pending_outcome(oid)
    assert store.load_pending_outcomes() == []


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def test_wave_85_e_append_and_load_aggregates(store: StateStore) -> None:
    for i in range(5):
        store.append_aggregate_sample(
            pattern_id="p", ts_ms=i * 100, ret=float(i), hit=(i % 2 == 0),
        )
    all_samples = store.load_aggregate_samples()
    assert len(all_samples) == 5
    assert all_samples[0]["hit"] is True
    assert all_samples[1]["hit"] is False


def test_wave_85_e_aggregates_filtered_by_pattern(store: StateStore) -> None:
    store.append_aggregate_sample(pattern_id="a", ts_ms=0, ret=1.0, hit=True)
    store.append_aggregate_sample(pattern_id="b", ts_ms=1, ret=2.0, hit=False)
    a_samples = store.load_aggregate_samples("a")
    assert len(a_samples) == 1
    assert a_samples[0]["pattern_id"] == "a"


def test_wave_85_e_trim_aggregates(store: StateStore) -> None:
    for ts in [100, 200, 300, 400]:
        store.append_aggregate_sample(
            pattern_id="p", ts_ms=ts, ret=1.0, hit=True,
        )
    removed = store.trim_aggregates_older_than(250)
    assert removed == 2
    remaining = [r["ts_ms"] for r in store.load_aggregate_samples()]
    assert remaining == [300, 400]


# ---------------------------------------------------------------------------
# Counter snapshots
# ---------------------------------------------------------------------------


def test_wave_85_e_counter_snapshot_roundtrip(store: StateStore) -> None:
    store.put_counter_snapshot("stats", {"ingest_errors": 5, "per_symbol": {"NQ": 1}})
    val = store.get_counter_snapshot("stats")
    assert val == {"ingest_errors": 5, "per_symbol": {"NQ": 1}}


def test_wave_85_e_counter_snapshot_upsert(store: StateStore) -> None:
    store.put_counter_snapshot("k", 1)
    store.put_counter_snapshot("k", 2)
    assert store.get_counter_snapshot("k") == 2


def test_wave_85_e_counter_snapshot_missing_returns_none(store: StateStore) -> None:
    assert store.get_counter_snapshot("nope") is None


# ---------------------------------------------------------------------------
# Thread safety smoke
# ---------------------------------------------------------------------------


def test_wave_85_e_concurrent_writes_do_not_corrupt(tmp_path: Path) -> None:
    """Two threads writing simultaneously should produce consistent totals."""
    import threading

    store = StateStore(tmp_path / "concurrent.db")
    N = 50

    def worker(pattern_id: str) -> None:
        for i in range(N):
            store.append_aggregate_sample(
                pattern_id=pattern_id, ts_ms=i, ret=1.0, hit=True,
            )

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert len(store.load_aggregate_samples("a")) == N
    assert len(store.load_aggregate_samples("b")) == N
    store.close()


def test_wave_85_e_close_is_idempotent(tmp_path: Path) -> None:
    s = StateStore(tmp_path / "idem.db")
    s.close()
    s.close()   # must not raise
