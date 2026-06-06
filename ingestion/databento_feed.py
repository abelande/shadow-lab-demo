"""Databento MBO (L3) feed — historical replay and live streaming.

Converts Databento's MBO schema into P6 OrderBookSnapshot objects with
full L3 order visibility (individual order IDs, ADD/CANCEL/MODIFY/FILL).

Streaming architecture: replay feed reads .dbn.zst record-by-record without
loading into memory. Supports multi-instrument files (full exchange dumps)
via instrument_id filtering, and time-range slicing via ts_event.

Requires: pip install databento
Env var: DATABENTO_API_KEY
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..models import (
    Order, OrderAction, OrderBookLevel, OrderBookSnapshot, Side,
)
from .base_feed import BaseFeed

logger = logging.getLogger(__name__)


class LiveFeedError(RuntimeError):
    """Clean, user-facing error from the Databento live feed."""
    pass


def _classify_live_error(err_text: str) -> str:
    """Map a Databento error string to a human-readable UI message."""
    t = err_text.lower()
    if "not authorized" in t or "unauthorized" in t:
        return (
            f"Not authorized for live MBO feed. "
            f"Check your Databento subscription includes live L3/MBO access "
            f"for this dataset. (Raw: {err_text})"
        )
    if "market" in t and ("closed" in t or "offline" in t or "maintenance" in t):
        return (
            f"Market is currently closed or in maintenance. "
            f"CME Globex is offline 5:00–6:00 PM ET daily and on weekends. "
            f"(Raw: {err_text})"
        )
    if "symbol" in t or "instrument" in t:
        return f"Symbol not found or not available for live feed. (Raw: {err_text})"
    if "rate" in t or "limit" in t:
        return f"Rate limit reached on live feed. (Raw: {err_text})"
    if "timeout" in t or "timed out" in t:
        return (
            f"Live feed connection timed out — market may be closed or outside "
            f"trading hours. (Raw: {err_text})"
        )
    return f"Live feed error: {err_text}"


# Databento action -> P6 OrderAction
_ACTION_MAP = {
    "A": OrderAction.ADD,
    "C": OrderAction.CANCEL,
    "M": OrderAction.MODIFY,
    "F": OrderAction.FILL,
    "T": OrderAction.FILL,  # Trade = fill
}

# Databento side -> P6 Side
_SIDE_MAP = {
    "B": Side.BID,
    "A": Side.ASK,
}


class OrderBook:
    """In-memory L3 order book reconstructed from MBO events.

    Maintains individual orders keyed by order_id and aggregates
    into OrderBookLevel objects on demand.
    """

    def __init__(self, num_levels: int = 10):
        self.num_levels = num_levels
        self._orders: Dict[str, Order] = {}
        self._price_levels: Dict[Side, Dict[float, set]] = {
            Side.BID: defaultdict(set),
            Side.ASK: defaultdict(set),
        }

    def apply(self, order: Order) -> None:
        """Apply an order event to the book."""
        oid = order.order_id

        if order.action == OrderAction.ADD:
            self._orders[oid] = order
            self._price_levels[order.side][order.price].add(oid)

        elif order.action == OrderAction.CANCEL:
            if oid in self._orders:
                old = self._orders.pop(oid)
                self._price_levels[old.side][old.price].discard(oid)
                if not self._price_levels[old.side][old.price]:
                    del self._price_levels[old.side][old.price]

        elif order.action == OrderAction.MODIFY:
            if oid in self._orders:
                old = self._orders[oid]
                self._price_levels[old.side][old.price].discard(oid)
                if not self._price_levels[old.side][old.price]:
                    del self._price_levels[old.side][old.price]
                self._orders[oid] = order
                self._price_levels[order.side][order.price].add(oid)

        elif order.action == OrderAction.FILL:
            if oid in self._orders:
                old = self._orders[oid]
                remaining = old.size - order.size
                if remaining <= 0:
                    self._orders.pop(oid)
                    self._price_levels[old.side][old.price].discard(oid)
                    if not self._price_levels[old.side][old.price]:
                        del self._price_levels[old.side][old.price]
                else:
                    self._orders[oid] = Order(
                        order_id=oid,
                        side=old.side,
                        price=old.price,
                        size=remaining,
                        timestamp_ms=order.timestamp_ms,
                        action=OrderAction.ADD,
                        is_aggressive=old.is_aggressive,
                    )

    def snapshot(self, symbol: str, timestamp_ms: int,
                 recent_trades: List[Order],
                 recent_events: List[Order]) -> OrderBookSnapshot:
        """Build an OrderBookSnapshot from current book state."""
        bids = self._build_levels(Side.BID, reverse=True)
        asks = self._build_levels(Side.ASK, reverse=False)

        return OrderBookSnapshot(
            timestamp_ms=timestamp_ms,
            symbol=symbol,
            bids=bids[:self.num_levels],
            asks=asks[:self.num_levels],
            recent_trades=recent_trades,
            recent_events=recent_events,
        )

    def _build_levels(self, side: Side, reverse: bool) -> List[OrderBookLevel]:
        """Aggregate individual orders into OrderBookLevel objects."""
        levels = []
        for price in sorted(self._price_levels[side].keys(), reverse=reverse):
            oids = self._price_levels[side][price]
            if not oids:
                continue
            orders = [self._orders[oid] for oid in oids if oid in self._orders]
            if not orders:
                continue
            volume = sum(o.size for o in orders)
            levels.append(OrderBookLevel(
                price=price,
                side=side,
                volume=volume,
                order_count=len(orders),
                orders=orders,
            ))
        return levels

    def clear(self):
        """Reset the book to empty state."""
        self._orders.clear()
        self._price_levels = {
            Side.BID: defaultdict(set),
            Side.ASK: defaultdict(set),
        }

    def total_orders(self) -> int:
        return len(self._orders)


# ═══════════════════════════════════════════════════════════════════
# Streaming Replay Feed
# ═══════════════════════════════════════════════════════════════════

class DatabentoReplayFeed(BaseFeed):
    """Replay historical Databento MBO data through P6.

    Streaming architecture: reads .dbn.zst files record-by-record without
    loading into a DataFrame. Supports multi-instrument files (full exchange
    dumps) via instrument_id filtering, and time-range slicing via ts_event.

    Memory usage: O(book_depth + event_window) regardless of file size.
    A month of full-exchange MBO data uses the same memory as a single day.

    Args:
        file_path: Path to .dbn.zst file (if None, fetches from API)
        symbol: Instrument symbol for API fetches (e.g. "ES.c.0")
        instrument_id: Specific instrument_id to filter in multi-instrument files
        filter_symbol: Human-readable symbol to resolve from metadata (e.g. "ES", "NQH6")
        dataset: Databento dataset (e.g. "GLBX.MDP3")
        start/end: Datetime strings for API fetch (ISO format)
        snapshot_interval_ms: How often to yield snapshots (default 100ms)
        num_levels: Number of book levels to include in snapshots
        event_window: Number of recent events to include per snapshot
        trade_window: Number of recent trades to include per snapshot
        api_key: Databento API key (defaults to DATABENTO_API_KEY env var)
        time_start: Start of time range filter (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
        time_end: End of time range filter
    """

    # Databento fixed-point price divisor (9 decimal places)
    PRICE_DIVISOR = 1_000_000_000
    # Sentinel value for null/missing prices
    INT64_MAX = 9223372036854775807

    def __init__(
        self,
        file_path: Optional[str] = None,
        symbol: str = "ES.c.0",
        instrument_id: Optional[int] = None,
        filter_symbol: Optional[str] = None,
        dataset: str = "GLBX.MDP3",
        start: Optional[str] = None,
        end: Optional[str] = None,
        snapshot_interval_ms: int = 100,
        num_levels: int = 10,
        event_window: int = 200,
        trade_window: int = 100,
        api_key: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ):
        super().__init__(symbol=symbol, data_level="L3")
        self.file_path = file_path
        self.instrument_id = instrument_id
        self.filter_symbol = filter_symbol
        self.dataset = dataset
        self.start = start
        self.end = end
        self.time_start = time_start
        self.time_end = time_end
        self.snapshot_interval_ms = snapshot_interval_ms
        self.num_levels = num_levels
        self.event_window = event_window
        self.trade_window = trade_window
        self.api_key = api_key or os.environ.get("DATABENTO_API_KEY", "")

        self._book = OrderBook(num_levels=num_levels)
        # Streaming state
        self._store = None
        self._iterator = None
        self._is_multi_instrument = False
        self._filter_instrument_id: Optional[int] = None
        self._time_start_ns: Optional[int] = None
        self._time_end_ns: Optional[int] = None
        self._recent_events: List[Order] = []
        self._recent_trades: List[Order] = []
        self._last_snapshot_ts: int = 0
        self._records_scanned: int = 0
        self._records_matched: int = 0
        self._exhausted = False

    def _resolve_instrument_id(self, store) -> Optional[int]:
        """Resolve filter_symbol to an instrument_id from file metadata.

        Handles:
          - Specific contract: "ESH6" → exact match in mappings
          - Root symbol: "ES" → finds front-month outright (nearest expiry)
          - Already an int via instrument_id param: pass through
        """
        if self.instrument_id is not None:
            return self.instrument_id

        if self.filter_symbol is None:
            return None

        sym = self.filter_symbol.upper()
        mappings = getattr(store.metadata, "mappings", {})
        if not mappings:
            return None

        # Try exact match first (e.g. "ESH6")
        if sym in mappings:
            return int(mappings[sym][0]["symbol"])

        # Try continuous contract key (e.g. single-instrument files: "NQ.c.0")
        continuous_key = f"{sym}.c.0"
        if continuous_key in mappings:
            return int(mappings[continuous_key][0]["symbol"])

        # Try any key that starts with the symbol followed by a dot
        for key, map_list in mappings.items():
            if key.startswith(sym + ".") and len(mappings) <= 5:
                # Single-instrument file — take first match
                return int(map_list[0]["symbol"])

        # Root symbol match — find front-month outright (no spreads/calendars)
        # Keys look like "ESH6", "NQM6", "CLJ26" — root + month code + year digits
        # Match by checking if key starts with the target symbol and next char is
        # a valid month code followed by digits
        _MONTH_ORDER = "FGHJKMNQUVXZ"
        candidates = []
        for key, map_list in mappings.items():
            if "-" in key or ":" in key:
                continue  # skip spreads
            # Check if key starts with our symbol and has a valid month code after
            if not key.startswith(sym):
                continue
            suffix = key[len(sym):]
            if not suffix or suffix[0] not in _MONTH_ORDER:
                continue  # not a valid contract month code
            # Extract month code and year for sorting
            month_char = suffix[0]
            year_str = "".join(c for c in suffix[1:] if c.isdigit())
            month_idx = _MONTH_ORDER.find(month_char)
            if month_idx < 0:
                month_idx = 99
            year = int(year_str) if year_str else 99
            iid = int(map_list[0]["symbol"])
            candidates.append((key, iid, year, month_idx))

        if not candidates:
            logger.warning("Could not resolve symbol '%s' in file metadata", sym)
            return None

        # Sort by year then month to get front month.
        # CME year convention: single digit wraps at decade boundary.
        # For files dated 2026, year "6" = 2026, "7" = 2027, "0" = 2030.
        # We need the nearest expiry AFTER the file's date.
        # Heuristic: years 0-5 are in the next decade (2030-2035),
        # years 6-9 are current decade (2026-2029).
        def _normalize_year(y):
            if y <= 5:
                return y + 10  # 0→10, 1→11, ..., 5→15
            return y           # 6→6, 7→7, 8→8, 9→9

        candidates.sort(key=lambda c: (_normalize_year(c[2]), c[3]))
        chosen = candidates[0]
        logger.info("Resolved '%s' → '%s' (instrument_id=%d)", sym, chosen[0], chosen[1])
        return chosen[1]

    def _parse_time_ns(self, time_str: str, end_of_day: bool = False) -> int:
        """Parse a time string to nanoseconds since epoch (UTC)."""
        from datetime import datetime, timezone
        s = time_str
        if len(s) == 10:  # bare date
            s = f"{s}T23:59:59" if end_of_day else f"{s}T00:00:00"
        if "+" not in s and "Z" not in s:
            s = s + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)

    @staticmethod
    def _action_to_str(action_val) -> str:
        """Convert a Databento action to single char. str() on the enum gives the char directly."""
        return str(action_val)

    @staticmethod
    def _side_to_str(side_val) -> str:
        """Convert a Databento side to single char. str() on the enum gives the char directly."""
        return str(side_val)

    async def connect(self) -> None:
        """Open the data source for streaming iteration."""
        import databento as db

        if self.file_path and os.path.isfile(self.file_path):
            logger.info("Opening MBO data from %s (streaming mode)", self.file_path)
            self._store = db.DBNStore.from_file(self.file_path)
        else:
            if not self.api_key:
                raise ValueError("No Databento API key. Set DATABENTO_API_KEY env var.")
            if not self.start or not self.end:
                raise ValueError("start and end required when fetching from API")

            logger.info("Fetching MBO data: %s %s %s->%s",
                        self.dataset, self._symbol, self.start, self.end)
            client = db.Historical(self.api_key)
            self._store = client.timeseries.get_range(
                dataset=self.dataset,
                symbols=[self._symbol],
                schema="mbo",
                start=self.start,
                end=self.end,
                stype_in="continuous",
            )

        # Detect multi-instrument file
        symbols = getattr(self._store.metadata, "symbols", [])
        mappings = getattr(self._store.metadata, "mappings", {})
        self._is_multi_instrument = len(mappings) > 10 or len(symbols) > 1

        # Resolve instrument_id filter
        self._filter_instrument_id = self._resolve_instrument_id(self._store)
        if self._is_multi_instrument and self._filter_instrument_id is None:
            if self.filter_symbol:
                raise ValueError(
                    f"Multi-instrument file but couldn't resolve symbol "
                    f"'{self.filter_symbol}'. Available: {symbols}"
                )
            else:
                logger.warning(
                    "Multi-instrument file with no symbol filter — "
                    "will process ALL instruments (may produce garbage). "
                    "Pass filter_symbol='ES' or similar."
                )

        # Parse time range filters to nanoseconds
        if self.time_start:
            self._time_start_ns = self._parse_time_ns(self.time_start, end_of_day=False)
        if self.time_end:
            self._time_end_ns = self._parse_time_ns(self.time_end, end_of_day=True)

        # Create iterator
        self._iterator = iter(self._store)
        self._connected = True
        self._exhausted = False

        filter_desc = ""
        if self._filter_instrument_id is not None:
            filter_desc += f" instrument_id={self._filter_instrument_id}"
        if self._time_start_ns is not None:
            filter_desc += f" from={self.time_start}"
        if self._time_end_ns is not None:
            filter_desc += f" to={self.time_end}"
        logger.info("Streaming ready.%s", filter_desc or " No filters.")
        if self._is_multi_instrument:
            logger.info("Multi-instrument file detected (%d symbols). Filtering enabled.",
                        len(symbols))

    def _record_to_order(self, record) -> Optional[Order]:
        """Convert a raw Databento MBO record to a P6 Order.

        Raw records use fixed-point prices (÷ 1e9), nanosecond timestamps,
        and string/enum action+side codes. INT64_MAX is a null sentinel.
        """
        action_str = self._action_to_str(record.action)
        side_str = self._side_to_str(record.side)

        if action_str not in _ACTION_MAP:
            return None
        if side_str not in _SIDE_MAP:
            return None

        raw_price = record.price
        if raw_price == self.INT64_MAX or raw_price <= 0:
            return None
        price = raw_price / self.PRICE_DIVISOR

        action = _ACTION_MAP[action_str]
        side = _SIDE_MAP[side_str]
        is_aggressive = action == OrderAction.FILL

        ts_ms = int(record.ts_event / 1_000_000)  # ns → ms

        return Order(
            order_id=str(record.order_id),
            side=side,
            price=price,
            size=float(record.size),
            timestamp_ms=ts_ms,
            action=action,
            is_aggressive=is_aggressive,
        )

    async def next(self) -> Optional[OrderBookSnapshot]:
        """Stream records and return next snapshot at the configured interval.

        Filters by instrument_id and time range on the fly — skipped records
        use negligible memory. O(1) memory per skipped record.
        """
        if not self._connected or self._iterator is None:
            raise RuntimeError("Feed not connected. Call connect() first.")

        if self._exhausted:
            return None

        for record in self._iterator:
            self._records_scanned += 1

            # ── Instrument filter ──
            if self._filter_instrument_id is not None:
                if record.instrument_id != self._filter_instrument_id:
                    continue

            # ── Time range filter (ts_event in nanoseconds) ──
            ts_event = record.ts_event
            # Skip sentinel timestamps (INT64_MAX used for status/reset records)
            if ts_event == self.INT64_MAX or ts_event <= 0:
                continue
            if self._time_start_ns is not None and ts_event < self._time_start_ns:
                continue
            if self._time_end_ns is not None and ts_event > self._time_end_ns:
                # Records are chronologically ordered by ts_event within each
                # instrument. To be safe, skip rather than stop hard — avoids
                # false exhaustion from any out-of-order records.
                continue

            # ── Convert to Order ──
            order = self._record_to_order(record)
            if order is None:
                continue

            self._records_matched += 1

            # Apply to book
            self._book.apply(order)

            # Track recent events
            self._recent_events.append(order)
            if len(self._recent_events) > self.event_window:
                self._recent_events = self._recent_events[-self.event_window:]

            # Track trades
            if order.action == OrderAction.FILL:
                self._recent_trades.append(order)
                if len(self._recent_trades) > self.trade_window:
                    self._recent_trades = self._recent_trades[-self.trade_window:]

            # Check if it's time for a snapshot
            ts_ms = order.timestamp_ms
            if self._last_snapshot_ts == 0:
                self._last_snapshot_ts = ts_ms

            if ts_ms - self._last_snapshot_ts >= self.snapshot_interval_ms:
                snapshot = self._book.snapshot(
                    symbol=self._symbol,
                    timestamp_ms=ts_ms,
                    recent_trades=list(self._recent_trades),
                    recent_events=list(self._recent_events),
                )
                self._last_snapshot_ts = ts_ms
                return snapshot

        # Iterator exhausted
        self._exhausted = True
        return None

    def iter_mbo_events(self):
        """Yield raw MBO events as ``Order`` records, one per market action.

        Unlike :meth:`next` (which aggregates events into 100ms snapshots and
        drops the per-event detail), this generator emits every ADD / CANCEL /
        MODIFY / FILL the file contains. Required by execution-sim consumers
        (QueueTracker / FillSimulator) that need full order-id-level state.

        Synchronous generator (Databento's iterator is sync). Honors the same
        instrument and time-range filters as :meth:`next`. Does *not* mutate
        ``self._book`` or any snapshot-path state, so it is safe to call on a
        fresh feed instance independently of the snapshot stream.

        Yields
        ------
        Order
            One per matched MBO record.
        """
        if not self._connected or self._iterator is None:
            raise RuntimeError("Feed not connected. Call connect() first.")

        for record in self._iterator:
            self._records_scanned += 1

            if self._filter_instrument_id is not None:
                if record.instrument_id != self._filter_instrument_id:
                    continue

            ts_event = record.ts_event
            if ts_event == self.INT64_MAX or ts_event <= 0:
                continue
            if self._time_start_ns is not None and ts_event < self._time_start_ns:
                continue
            if self._time_end_ns is not None and ts_event > self._time_end_ns:
                continue

            order = self._record_to_order(record)
            if order is None:
                continue

            self._records_matched += 1
            yield order

        self._exhausted = True

    @property
    def progress(self) -> float:
        """Streaming progress is unknown without pre-scanning. Returns -1."""
        return -1.0

    @property
    def events_processed(self) -> int:
        return self._records_matched

    @property
    def records_scanned(self) -> int:
        return self._records_scanned

    async def disconnect(self) -> None:
        """Clean up."""
        self._store = None
        self._iterator = None
        self._exhausted = True
        self._book = OrderBook(num_levels=self.num_levels)
        self._recent_events.clear()
        self._recent_trades.clear()
        await super().disconnect()


# ═══════════════════════════════════════════════════════════════════
# Live Feed (unchanged)
# ═══════════════════════════════════════════════════════════════════

class DatabentoLiveFeed(BaseFeed):
    """Live streaming Databento MBO feed.

    Connects to Databento's live WebSocket API and yields
    OrderBookSnapshot objects in real-time.

    Args:
        symbol: Instrument symbol (e.g. "ES.c.0")
        dataset: Databento dataset (e.g. "GLBX.MDP3")
        snapshot_interval_ms: How often to yield snapshots
        num_levels: Number of book levels per snapshot
        api_key: Databento API key
    """

    def __init__(
        self,
        symbol: str = "ES.c.0",
        dataset: str = "GLBX.MDP3",
        snapshot_interval_ms: int = 100,
        num_levels: int = 10,
        event_window: int = 200,
        trade_window: int = 100,
        api_key: Optional[str] = None,
    ):
        super().__init__(symbol=symbol, data_level="L3")
        self.dataset = dataset
        self.snapshot_interval_ms = snapshot_interval_ms
        self.num_levels = num_levels
        self.event_window = event_window
        self.trade_window = trade_window
        self.api_key = api_key or os.environ.get("DATABENTO_API_KEY", "")

        self._book = OrderBook(num_levels=num_levels)
        self._client = None
        self._recent_events: List[Order] = []
        self._recent_trades: List[Order] = []
        self._last_snapshot_ts: int = 0
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=50000)
        self._ingest_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Connect to Databento live feed."""
        import databento as db

        if not self.api_key:
            raise ValueError("No Databento API key. Set DATABENTO_API_KEY env var.")

        self._client = db.Live(self.api_key)
        self._client.subscribe(
            dataset=self.dataset,
            schema="mbo",
            symbols=[self._symbol],
            stype_in="continuous",
        )

        self._connected = True
        self._ingest_task = asyncio.create_task(self._ingest_loop())
        logger.info("Connected to Databento live feed: %s %s", self.dataset, self._symbol)

    async def _ingest_loop(self) -> None:
        """Background loop consuming live MBO events."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._blocking_ingest)
        except Exception as e:
            if self._connected:
                logger.error("Live feed error: %s", e)

    def _blocking_ingest(self) -> None:
        """Synchronous blocking iteration over Databento live records."""
        try:
            for record in self._client:
                if not self._connected:
                    break

                if type(record).__name__ == "ErrorMsg":
                    err_text = getattr(record, "err", str(record))
                    raise LiveFeedError(_classify_live_error(err_text))

                order = Order(
                    order_id=str(record.order_id),
                    side=_SIDE_MAP.get(record.side, Side.BID),
                    price=record.price / 1e9,
                    size=float(record.size),
                    timestamp_ms=int(record.ts_event / 1_000_000),
                    action=_ACTION_MAP.get(record.action, OrderAction.ADD),
                    is_aggressive=(record.action in ("F", "T")),
                )

                try:
                    self._event_queue.put_nowait(order)
                except asyncio.QueueFull:
                    try:
                        self._event_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self._event_queue.put_nowait(order)

        except LiveFeedError:
            raise
        except Exception as e:
            if self._connected:
                logger.error("Live feed ingest thread error: %s", e)

    async def next(self) -> Optional[OrderBookSnapshot]:
        """Process queued events and return next snapshot."""
        if not self._connected:
            raise RuntimeError("Feed not connected. Call connect() first.")

        while not self._event_queue.empty():
            try:
                order = self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            self._book.apply(order)

            self._recent_events.append(order)
            if len(self._recent_events) > self.event_window:
                self._recent_events = self._recent_events[-self.event_window:]

            if order.action == OrderAction.FILL:
                self._recent_trades.append(order)
                if len(self._recent_trades) > self.trade_window:
                    self._recent_trades = self._recent_trades[-self.trade_window:]

        ts_ms = int(time.time() * 1000)

        if self._last_snapshot_ts == 0 or ts_ms - self._last_snapshot_ts >= self.snapshot_interval_ms:
            self._last_snapshot_ts = ts_ms
            return self._book.snapshot(
                symbol=self._symbol,
                timestamp_ms=ts_ms,
                recent_trades=list(self._recent_trades),
                recent_events=list(self._recent_events),
            )

        return None

    async def iter_mbo_events(self, *, idle_timeout_ms: int = 1_000):
        """Yield raw MBO events as ``Order`` records from the live queue.

        Async generator form of :meth:`DatabentoReplayFeed.iter_mbo_events` so
        notebooks and auditors that consume ``_common.collect_events`` can
        target either feed interchangeably (Wave 2 parity contract).

        Unlike the replay variant, the live generator never exhausts while
        the feed is connected — it drains the internal queue as events
        arrive. If the queue stays empty for ``idle_timeout_ms`` the
        generator yields control back to the caller so they can decide
        whether to keep waiting (e.g. a time-boxed parity run).

        Parameters
        ----------
        idle_timeout_ms
            Max milliseconds to wait for a new event before returning.
            Callers running time-boxed tests set this low (e.g. 100ms) so
            the generator exits promptly when the queue is quiet.

        Yields
        ------
        Order
            One per live MBO record.

        Notes
        -----
        This method does **not** advance the internal order book — it is a
        pure passthrough. Call sites that want both the event stream *and*
        maintained book state should consume via ``next()`` instead.
        """
        if not self._connected:
            raise RuntimeError("Feed not connected. Call connect() first.")
        timeout_s = idle_timeout_ms / 1_000.0
        while self._connected:
            try:
                order = await asyncio.wait_for(
                    self._event_queue.get(), timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                # Queue idle — return so the caller isn't trapped forever.
                return
            yield order

    async def disconnect(self) -> None:
        """Disconnect from live feed."""
        self._connected = False
        if self._ingest_task and not self._ingest_task.done():
            self._ingest_task.cancel()
            try:
                await self._ingest_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        await super().disconnect()


# ═══════════════════════════════════════════════════════════════════
# L1 Live Feed (MBP-1 schema)
# ═══════════════════════════════════════════════════════════════════

class DatabentoL1LiveFeed(BaseFeed):
    """Live L1 feed using Databento's MBP-1 schema (best bid/ask + trades).

    What works with L1:
      - Price chart (from trade prints + BBO mid)
      - Time & Sales tape (trade prints only — no ADD/CANCEL/MODIFY)
      - Cup Flip tape reader — pressure and streak from fill direction
      - CVD (cumulative volume delta from trade side)

    What is disabled (requires L3 MBO):
      - Spoof detection (no individual order IDs)
      - Fragility scoring (no order-level depth)
      - Iceberg inference (no refill tracking)
      - Regime classifier (no book state)

    Data model: each MBP-1 record carries:
      - The triggering trade (price, size, side, action='T')
      - levels[0]: BidAskPair (bid_px, ask_px, bid_sz, ask_sz, bid_ct, ask_ct)

    We synthesize:
      - One FILL Order per trade print (for tape + cup flip pressure)
      - Two synthetic ADD Orders per BBO update (bid + ask top level)
        so the DOM panel shows best bid/ask even without full depth.
    """

    # Sentinel for null prices
    INT64_MAX = 9223372036854775807
    PRICE_DIVISOR = 1_000_000_000

    def __init__(
        self,
        symbol: str = "ES.c.0",
        dataset: str = "GLBX.MDP3",
        snapshot_interval_ms: int = 100,
        num_levels: int = 10,
        event_window: int = 200,
        trade_window: int = 200,
        api_key: Optional[str] = None,
    ):
        super().__init__(symbol=symbol, data_level="L1")
        self.dataset = dataset
        self.snapshot_interval_ms = snapshot_interval_ms
        self.num_levels = num_levels
        self.event_window = event_window
        self.trade_window = trade_window
        self.api_key = api_key or os.environ.get("DATABENTO_API_KEY", "")

        self._book = OrderBook(num_levels=num_levels)
        self._client = None
        self._recent_events: List[Order] = []
        self._recent_trades: List[Order] = []
        self._last_snapshot_ts: int = 0
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=50000)
        self._ingest_task: Optional[asyncio.Task] = None
        # Rolling synthetic order IDs for BBO levels
        self._bbo_bid_id: str = "bbo-bid-0"
        self._bbo_ask_id: str = "bbo-ask-0"
        self._bbo_seq: int = 0

    async def connect(self) -> None:
        """Connect to Databento live L1 feed (MBP-1 schema)."""
        import databento as db

        if not self.api_key:
            raise ValueError("No Databento API key. Set DATABENTO_API_KEY env var.")

        self._client = db.Live(self.api_key)
        self._client.subscribe(
            dataset=self.dataset,
            schema="mbp-1",          # L1: best bid/ask + trade trigger
            symbols=[self._symbol],
            stype_in="continuous",
        )

        self._connected = True
        self._ingest_task = asyncio.create_task(self._ingest_loop())
        logger.info("Connected to Databento L1 live feed: %s %s (mbp-1)", self.dataset, self._symbol)

    async def _ingest_loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._blocking_ingest)
        except Exception as e:
            if self._connected:
                logger.error("L1 live feed error: %s", e)

    def _blocking_ingest(self) -> None:
        """Convert MBP-1 records to synthetic Orders and queue them."""
        try:
            for record in self._client:
                if not self._connected:
                    break

                if type(record).__name__ == "ErrorMsg":
                    err_text = getattr(record, "err", str(record))
                    raise LiveFeedError(_classify_live_error(err_text))

                # Skip non-market-data records (SystemMsg, InstrumentDefMsg, etc.)
                if not hasattr(record, "price"):
                    continue

                ts_ms = int(record.ts_event / 1_000_000)
                events: List[Order] = []

                # ── 1. Trade print (the triggering event) ──
                trade_price = record.price
                trade_size  = record.size
                trade_side  = str(record.side)
                trade_action = str(record.action)

                if (trade_price not in (self.INT64_MAX, 0) and
                        trade_size > 0 and
                        trade_action in ("T", "F") and
                        trade_side in _SIDE_MAP):
                    fill = Order(
                        order_id=f"trade-{ts_ms}-{trade_size}",
                        side=_SIDE_MAP[trade_side],
                        price=trade_price / self.PRICE_DIVISOR,
                        size=float(trade_size),
                        timestamp_ms=ts_ms,
                        action=OrderAction.FILL,
                        is_aggressive=True,
                    )
                    events.append(fill)

                # ── 2. BBO synthetic level (best bid + best ask) ──
                try:
                    level = record.levels[0]
                    self._bbo_seq += 1
                    seq = self._bbo_seq

                    bid_px = level.bid_px
                    ask_px = level.ask_px
                    bid_sz = level.bid_sz
                    ask_sz = level.ask_sz

                    if bid_px not in (self.INT64_MAX, 0) and bid_sz > 0:
                        # Cancel previous synthetic bid, add new one
                        events.append(Order(
                            order_id=self._bbo_bid_id,
                            side=Side.BID,
                            price=0.0,  # cancel doesn't need price
                            size=0.0,
                            timestamp_ms=ts_ms,
                            action=OrderAction.CANCEL,
                            is_aggressive=False,
                        ))
                        self._bbo_bid_id = f"bbo-bid-{seq}"
                        events.append(Order(
                            order_id=self._bbo_bid_id,
                            side=Side.BID,
                            price=bid_px / self.PRICE_DIVISOR,
                            size=float(bid_sz),
                            timestamp_ms=ts_ms,
                            action=OrderAction.ADD,
                            is_aggressive=False,
                        ))

                    if ask_px not in (self.INT64_MAX, 0) and ask_sz > 0:
                        events.append(Order(
                            order_id=self._bbo_ask_id,
                            side=Side.ASK,
                            price=0.0,
                            size=0.0,
                            timestamp_ms=ts_ms,
                            action=OrderAction.CANCEL,
                            is_aggressive=False,
                        ))
                        self._bbo_ask_id = f"bbo-ask-{seq}"
                        events.append(Order(
                            order_id=self._bbo_ask_id,
                            side=Side.ASK,
                            price=ask_px / self.PRICE_DIVISOR,
                            size=float(ask_sz),
                            timestamp_ms=ts_ms,
                            action=OrderAction.ADD,
                            is_aggressive=False,
                        ))
                except (IndexError, AttributeError):
                    pass  # No level data in this record

                for ev in events:
                    try:
                        self._event_queue.put_nowait(ev)
                    except asyncio.QueueFull:
                        try:
                            self._event_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        self._event_queue.put_nowait(ev)

        except LiveFeedError:
            raise
        except Exception as e:
            if self._connected:
                logger.error("L1 ingest thread error: %s", e)

    def next_event(self) -> Optional[dict]:
        """Return the next individual BBO event as a lightweight dict.

        Returns { mid, bid, ask, ts } or None if no events queued.
        Used by the engine runner to emit price_tick messages for smooth
        chart rendering independent of the snapshot interval.
        """
        try:
            order = self._event_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

        # Apply to book so it stays in sync
        self._book.apply(order)
        self._recent_events.append(order)
        if len(self._recent_events) > self.event_window:
            self._recent_events = self._recent_events[-self.event_window:]
        if order.action == OrderAction.FILL:
            self._recent_trades.append(order)
            if len(self._recent_trades) > self.trade_window:
                self._recent_trades = self._recent_trades[-self.trade_window:]

        # Extract current BBO from book
        snap = self._book.snapshot(
            symbol=self._symbol,
            timestamp_ms=order.timestamp_ms,
            recent_trades=[],
            recent_events=[],
        )
        bid = None
        ask = None
        if snap:
            for lvl in snap.bids:
                if bid is None or lvl.price > bid:
                    bid = lvl.price
            for lvl in snap.asks:
                if ask is None or lvl.price < ask:
                    ask = lvl.price

        mid = None
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
        elif bid is not None:
            mid = bid
        elif ask is not None:
            mid = ask

        if mid is None:
            return None

        return {"mid": mid, "bid": bid, "ask": ask, "ts": order.timestamp_ms}

    async def next(self) -> Optional[OrderBookSnapshot]:
        """Process queued synthetic orders and return next snapshot."""
        if not self._connected:
            raise RuntimeError("Feed not connected. Call connect() first.")

        while not self._event_queue.empty():
            try:
                order = self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            # Skip cancels for unknown order IDs gracefully
            self._book.apply(order)

            self._recent_events.append(order)
            if len(self._recent_events) > self.event_window:
                self._recent_events = self._recent_events[-self.event_window:]

            if order.action == OrderAction.FILL:
                self._recent_trades.append(order)
                if len(self._recent_trades) > self.trade_window:
                    self._recent_trades = self._recent_trades[-self.trade_window:]

        ts_ms = int(time.time() * 1000)

        if self._last_snapshot_ts == 0 or ts_ms - self._last_snapshot_ts >= self.snapshot_interval_ms:
            self._last_snapshot_ts = ts_ms
            return self._book.snapshot(
                symbol=self._symbol,
                timestamp_ms=ts_ms,
                recent_trades=list(self._recent_trades),
                recent_events=list(self._recent_events),
            )

        return None

    async def disconnect(self) -> None:
        self._connected = False
        if self._ingest_task and not self._ingest_task.done():
            self._ingest_task.cancel()
            try:
                await self._ingest_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        await super().disconnect()
