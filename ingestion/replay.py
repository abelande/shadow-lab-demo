"""Replay feed for historical order book data from Parquet/CSV files."""
from __future__ import annotations

import os
from enum import Enum
from typing import List, Optional

from ..models import (
    Order, OrderAction, OrderBookLevel, OrderBookSnapshot, Side,
)
from .base_feed import BaseFeed


class PlayState(Enum):
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class ReplayFeed(BaseFeed):
    """Replays historical order book data from Parquet or CSV files.

    Expected columns: timestamp, side, price, volume, order_count
    Rows are grouped by timestamp to reconstruct full snapshots.
    Supports play, pause, step, seek, and speed control.
    """

    def __init__(
        self,
        file_path: str,
        symbol: str = "REPLAY",
        speed_multiplier: float = 1.0,
    ):
        super().__init__(symbol=symbol, data_level="L2")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        self._file_path = file_path
        self.speed_multiplier = max(0.01, speed_multiplier)

        self._groups: List[dict] = []
        self._cursor = 0
        self._state = PlayState.STOPPED
        self._loaded = False

    # ── data loading ─────────────────────────────────────────────

    def _load_data(self) -> None:
        """Load and group data from Parquet or CSV."""
        ext = os.path.splitext(self._file_path)[1].lower()

        if ext in (".parquet", ".pq"):
            self._load_parquet()
        elif ext == ".csv":
            self._load_csv()
        else:
            try:
                self._load_parquet()
            except Exception:
                self._load_csv()

        self._loaded = True

    def _load_parquet(self) -> None:
        """Load from Parquet file using pyarrow or pandas."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(self._file_path)
            df_dict = table.to_pydict()
            self._parse_dict(df_dict)
        except ImportError:
            import pandas as pd
            df = pd.read_parquet(self._file_path)
            self._parse_dict(df.to_dict(orient="list"))

    def _load_csv(self) -> None:
        """Load from CSV file."""
        import csv
        with open(self._file_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            self._groups = []
            return

        required = {"timestamp", "side", "price", "volume"}
        available = set(rows[0].keys())
        missing = required - available
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        df_dict: dict = {k: [] for k in rows[0].keys()}
        for row in rows:
            for k, v in row.items():
                df_dict[k].append(v)

        self._parse_dict(df_dict)

    def _parse_dict(self, d: dict) -> None:
        """Parse column dict into grouped snapshots."""
        required = {"timestamp", "side", "price", "volume"}
        available = set(d.keys())
        missing = required - available
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        n = len(d["timestamp"])
        has_order_count = "order_count" in d

        rows = []
        for i in range(n):
            rows.append({
                "timestamp": int(float(d["timestamp"][i])),
                "side": str(d["side"][i]).upper(),
                "price": float(d["price"][i]),
                "volume": float(d["volume"][i]),
                "order_count": int(d["order_count"][i]) if has_order_count else 1,
            })

        groups: dict = {}
        for r in rows:
            ts = r["timestamp"]
            if ts not in groups:
                groups[ts] = []
            groups[ts].append(r)

        self._groups = [
            {"timestamp": ts, "rows": grp}
            for ts, grp in sorted(groups.items())
        ]

    # ── playback control ─────────────────────────────────────────

    def play(self) -> None:
        """Start or resume playback."""
        self._state = PlayState.PLAYING

    def pause(self) -> None:
        """Pause playback."""
        self._state = PlayState.PAUSED

    def stop(self) -> None:
        """Stop playback and reset cursor."""
        self._state = PlayState.STOPPED
        self._cursor = 0

    def seek(self, position: int) -> None:
        """Seek to a specific position (0-indexed)."""
        if not self._loaded:
            self._load_data()
        self._cursor = max(0, min(position, len(self._groups) - 1))

    def step(self) -> None:
        """Advance one snapshot while paused."""
        self._state = PlayState.PAUSED

    @property
    def total_snapshots(self) -> int:
        if not self._loaded:
            self._load_data()
        return len(self._groups)

    @property
    def current_position(self) -> int:
        return self._cursor

    @property
    def play_state(self) -> PlayState:
        return self._state

    # ── feed interface ───────────────────────────────────────────

    async def connect(self) -> None:
        """Load data and prepare for playback."""
        if not self._loaded:
            self._load_data()
        self._connected = True
        self._state = PlayState.PLAYING

    async def next(self) -> Optional[OrderBookSnapshot]:
        """Return the next snapshot from the replay data."""
        if not self._connected:
            await self.connect()

        if self._state == PlayState.STOPPED:
            return None

        if self._cursor >= len(self._groups):
            return None

        group = self._groups[self._cursor]
        self._cursor += 1

        ts = group["timestamp"]
        bids: List[OrderBookLevel] = []
        asks: List[OrderBookLevel] = []

        for row in group["rows"]:
            side = Side.BID if row["side"] in ("BID", "B", "BUY") else Side.ASK
            level = OrderBookLevel(
                price=row["price"],
                side=side,
                volume=row["volume"],
                order_count=row["order_count"],
            )
            if side == Side.BID:
                bids.append(level)
            else:
                asks.append(level)

        bids.sort(key=lambda l: l.price, reverse=True)
        asks.sort(key=lambda l: l.price)

        return OrderBookSnapshot(
            timestamp_ms=ts,
            symbol=self._symbol,
            bids=bids,
            asks=asks,
        )

    async def disconnect(self) -> None:
        """Close the replay feed."""
        self._connected = False
        self._state = PlayState.STOPPED
