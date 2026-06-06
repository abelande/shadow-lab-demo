"""
p6lab.persistence.state_store — Wave 8.5-E

SQLite-backed durable state for position_manager + outcome_tracker.
Enables crash-restart semantics: Wave 8's live-trading push relies on
this to reconstruct open positions + pending outcomes after an
unexpected restart.

Design rules (from plan §8.5-E):
- **Stdlib sqlite3 only.** No ORM. No external dep.
- **WAL journal mode** for concurrent readers while a writer is active.
- **Single connection per process** + threading.RLock for thread safety.
- **Schema version column** on every table so Wave 9 migrations are
  a straightforward ALTER or rewrite-to-new-table path.
- **Default state_store=None** in callers preserves legacy in-memory
  behavior — zero-regression migration.

Tables
------
- ``meta`` (key/value): schema_version, created_at_ms
- ``open_positions``: one row per live position
- ``pending_outcomes``: one row per pattern match awaiting horizon exit
- ``pattern_aggregates``: append-only (pattern_id, ts_ms, ret, hit)
- ``counter_snapshots``: key/value JSON for arbitrary stats

Not covered
-----------
- Cross-process coordination beyond WAL's reader/writer contract.
  Operators running multiple writers against the same file should use
  an OS-level file lock; that's explicitly out of scope for Wave 8.5.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


SCHEMA_VERSION: int = 1


_CREATE_STATEMENTS: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        schema_version INTEGER NOT NULL DEFAULT 1
    )""",
    """CREATE TABLE IF NOT EXISTS open_positions (
        pos_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        entry_price REAL,
        opened_at_ms INTEGER NOT NULL,
        schema_version INTEGER NOT NULL DEFAULT 1
    )""",
    """CREATE INDEX IF NOT EXISTS idx_open_pos_pattern ON open_positions(pattern_id)""",
    """CREATE INDEX IF NOT EXISTS idx_open_pos_symbol ON open_positions(symbol)""",
    """CREATE TABLE IF NOT EXISTS pending_outcomes (
        outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        expected_direction TEXT NOT NULL,
        expected_move_atr REAL NOT NULL,
        entry_ts_ms INTEGER NOT NULL,
        entry_mid REAL NOT NULL,
        exit_ts_ms INTEGER NOT NULL,
        confidence_tier TEXT NOT NULL,
        regime TEXT NOT NULL,
        schema_version INTEGER NOT NULL DEFAULT 1
    )""",
    """CREATE INDEX IF NOT EXISTS idx_pending_symbol ON pending_outcomes(symbol)""",
    """CREATE TABLE IF NOT EXISTS pattern_aggregates (
        agg_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_id TEXT NOT NULL,
        ts_ms INTEGER NOT NULL,
        ret REAL NOT NULL,
        hit INTEGER NOT NULL,
        schema_version INTEGER NOT NULL DEFAULT 1
    )""",
    """CREATE INDEX IF NOT EXISTS idx_agg_pattern ON pattern_aggregates(pattern_id)""",
    """CREATE TABLE IF NOT EXISTS counter_snapshots (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at_ms INTEGER NOT NULL,
        schema_version INTEGER NOT NULL DEFAULT 1
    )""",
)


class StateStore:
    """Thread-safe SQLite durable store with a focused API.

    Parameters
    ----------
    path:
        SQLite file path. Parent directory is auto-created. Pass
        ``":memory:"`` for an in-memory store (useful in tests; WAL mode
        is not activated for in-memory stores).
    check_same_thread:
        Passed through to sqlite3.connect. Default False so a single
        connection + RLock pattern works across threads.

    Usage
    -----
    >>> store = StateStore("artifacts/p6lab/state/live.db")
    >>> store.put_open_position({
    ...     "pattern_id": "bull_flag", "symbol": "NQ", "side": "BUY",
    ...     "quantity": 1, "entry_price": 20000.0, "opened_at_ms": 1700000000000,
    ... })
    >>> rows = store.load_open_positions()
    >>> store.close()
    """

    def __init__(
        self,
        path: Path | str,
        *,
        check_same_thread: bool = False,
    ) -> None:
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=check_same_thread
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed: bool = False
        self._init_schema()
        logger.info(
            "wave85-E state_store opened: path=%s schema_version=%d",
            self.path, SCHEMA_VERSION,
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            # WAL for concurrent readers (no-op for :memory:)
            if str(self.path) != ":memory:":
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.DatabaseError:
                    # Some filesystems (e.g. overlayfs in containers) reject
                    # WAL; fall through to default DELETE journal.
                    logger.debug("wave85-E WAL unavailable; default journal in use")
            for stmt in _CREATE_STATEMENTS:
                self._conn.execute(stmt)
            # Record schema version + creation timestamp if missing
            now_ms = int(time.time() * 1000)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value, schema_version) VALUES (?, ?, ?)",
                ("created_at_ms", str(now_ms), SCHEMA_VERSION),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value, schema_version) VALUES (?, ?, ?)",
                ("schema_version", str(SCHEMA_VERSION), SCHEMA_VERSION),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Open positions
    # ------------------------------------------------------------------

    def put_open_position(self, row: dict) -> int:
        """Insert one open position. Returns the auto-generated pos_id."""
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO open_positions
                   (pattern_id, symbol, side, quantity, entry_price, opened_at_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    row["pattern_id"], row["symbol"], row["side"],
                    int(row["quantity"]),
                    float(row["entry_price"]) if row.get("entry_price") is not None else None,
                    int(row["opened_at_ms"]),
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def update_position_entry_price(
        self, *, pos_id: int, entry_price: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE open_positions SET entry_price = ? WHERE pos_id = ?",
                (float(entry_price), int(pos_id)),
            )
            self._conn.commit()

    def remove_open_position(self, pos_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM open_positions WHERE pos_id = ?",
                (int(pos_id),),
            )
            self._conn.commit()

    def load_open_positions(self) -> list[dict]:
        """Return all rows, ordered by opened_at_ms ascending (oldest first)."""
        with self._lock:
            cursor = self._conn.execute(
                """SELECT pos_id, pattern_id, symbol, side, quantity,
                          entry_price, opened_at_ms
                   FROM open_positions ORDER BY opened_at_ms ASC"""
            )
            return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Pending outcomes
    # ------------------------------------------------------------------

    def put_pending_outcome(self, row: dict) -> int:
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO pending_outcomes
                   (pattern_id, symbol, expected_direction, expected_move_atr,
                    entry_ts_ms, entry_mid, exit_ts_ms, confidence_tier, regime)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["pattern_id"], row["symbol"],
                    row["expected_direction"], float(row["expected_move_atr"]),
                    int(row["entry_ts_ms"]), float(row["entry_mid"]),
                    int(row["exit_ts_ms"]),
                    row["confidence_tier"], row["regime"],
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def remove_pending_outcome(self, outcome_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM pending_outcomes WHERE outcome_id = ?",
                (int(outcome_id),),
            )
            self._conn.commit()

    def load_pending_outcomes(self) -> list[dict]:
        with self._lock:
            cursor = self._conn.execute(
                """SELECT outcome_id, pattern_id, symbol, expected_direction,
                          expected_move_atr, entry_ts_ms, entry_mid,
                          exit_ts_ms, confidence_tier, regime
                   FROM pending_outcomes ORDER BY entry_ts_ms ASC"""
            )
            return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Pattern aggregates (append-only rolling log)
    # ------------------------------------------------------------------

    def append_aggregate_sample(
        self, *, pattern_id: str, ts_ms: int, ret: float, hit: bool,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO pattern_aggregates
                   (pattern_id, ts_ms, ret, hit) VALUES (?, ?, ?, ?)""",
                (pattern_id, int(ts_ms), float(ret), 1 if hit else 0),
            )
            self._conn.commit()

    def load_aggregate_samples(
        self, pattern_id: str | None = None,
    ) -> list[dict]:
        with self._lock:
            if pattern_id is None:
                cursor = self._conn.execute(
                    "SELECT pattern_id, ts_ms, ret, hit FROM pattern_aggregates ORDER BY ts_ms ASC"
                )
            else:
                cursor = self._conn.execute(
                    """SELECT pattern_id, ts_ms, ret, hit
                       FROM pattern_aggregates
                       WHERE pattern_id = ? ORDER BY ts_ms ASC""",
                    (pattern_id,),
                )
            return [
                {"pattern_id": r["pattern_id"], "ts_ms": r["ts_ms"],
                 "ret": r["ret"], "hit": bool(r["hit"])}
                for r in cursor.fetchall()
            ]

    def trim_aggregates_older_than(self, cutoff_ts_ms: int) -> int:
        """Delete samples older than cutoff_ts_ms. Returns rows removed."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM pattern_aggregates WHERE ts_ms < ?",
                (int(cutoff_ts_ms),),
            )
            self._conn.commit()
            return int(cursor.rowcount or 0)

    # ------------------------------------------------------------------
    # Counter snapshots (arbitrary JSON key/value)
    # ------------------------------------------------------------------

    def put_counter_snapshot(self, key: str, value: Any) -> None:
        with self._lock:
            now_ms = int(time.time() * 1000)
            self._conn.execute(
                """INSERT INTO counter_snapshots (key, value_json, updated_at_ms)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                       value_json = excluded.value_json,
                       updated_at_ms = excluded.updated_at_ms""",
                (key, json.dumps(value), now_ms),
            )
            self._conn.commit()

    def get_counter_snapshot(self, key: str) -> Any | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT value_json FROM counter_snapshots WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()
            return json.loads(row["value_json"]) if row else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Explicit commit — no-op in practice since every write commits."""
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
