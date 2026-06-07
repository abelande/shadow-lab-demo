"""Background engine runner — processes snapshots and broadcasts frames.

Supports three modes:
- live: Streams from Databento live MBO feed
- replay: Replays from a .dbn.zst file (see backtest_api.py)
- paused: Idle, waiting for mode change
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from .config import config
from .websocket import ws_manager, serialize_frame

logger = logging.getLogger(__name__)


def _import_feed_module():
    """Import databento_feed with fallback for both package and PYTHONPATH modes."""
    try:
        from ..ingestion import databento_feed
        return databento_feed
    except (ImportError, SystemError):
        import importlib
        return importlib.import_module("ingestion.databento_feed")


def _build_correlation_components() -> tuple[object | None, object | None]:
    """Build (engine, accumulator) for the Wave 5 thesis-chain wire.

    Loads the p6lab CorrelationEngine from ``correlation_runs/models/CURRENT.json``
    and returns a ready FeatureAccumulator paired with it. Returns ``(None, None)``
    if any artifact is missing, any import fails, or ``P6LAB_CORRELATION_ENABLED``
    is explicitly ``"0"`` — the pipeline falls back to standard behavior.
    """
    if os.environ.get("P6LAB_CORRELATION_ENABLED", "1") == "0":
        logger.info("correlation engine: disabled via P6LAB_CORRELATION_ENABLED=0")
        return None, None

    try:
        from pathlib import Path
        from p6lab.correlation.engine import CorrelationEngine
        from p6lab.correlation.match_broker import MatchBroker
        from p6lab.correlation.scorer import EnsembleScorer
        from p6lab.live.feature_accumulator import FeatureAccumulator
        from p6lab.patterns.library import PatternLibrary
        from p6lab.patterns.template_matcher import TemplateMatcher
    except Exception as exc:
        logger.info("correlation engine: p6lab imports failed (%s); skipping", exc)
        return None, None

    repo_root = Path(__file__).resolve().parents[1]
    p6lab_root = repo_root / "p6lab"
    # Demo build: prefer the bundled synthetic library; fall back to the (stripped) mined one.
    demo_lib = repo_root / "demo" / "library_demo.yaml"
    lib_path = demo_lib if demo_lib.is_file() else (
        p6lab_root / "artifacts" / "p6lab" / "pattern_library" / "library.yaml"
    )
    registry_path = p6lab_root / "correlation_runs" / "models" / "CURRENT.json"
    if not lib_path.is_file():
        logger.info("correlation engine: library.yaml missing at %s; skipping", lib_path)
        return None, None

    try:
        lib = PatternLibrary(lib_path); lib.load()
        engine = CorrelationEngine(
            library=lib,
            matcher=TemplateMatcher(),
            scorer=EnsembleScorer(),
            broker=MatchBroker(),
        )
        if registry_path.is_file():
            engine.load_current_model(registry_path)
            logger.info("correlation engine: loaded model via %s", registry_path)
        else:
            logger.warning(
                "correlation engine: no model registry at %s; engine runs unloaded",
                registry_path,
            )
        accumulator = FeatureAccumulator(tick_size=0.25, window_seconds=300.0, num_levels=20)
        return engine, accumulator
    except Exception:
        logger.exception("correlation engine: construction failed; skipping")
        return None, None


def _level_capabilities(level: str) -> dict:
    """Return which engine layers are active for a given data level."""
    if level == "L3":
        return {
            "tape": True, "price": True, "dom": True,
            "cup_flip": True, "spoof": True,
            "fragility": True, "iceberg": True, "regime": True,
        }
    # L1: MBP-1 — trades + BBO only
    return {
        "tape": True,        # trade prints
        "price": True,       # mid from BBO
        "dom": True,         # best bid/ask (1 level only)
        "cup_flip": True,    # pressure from fill direction
        "spoof": False,      # needs individual order IDs
        "fragility": False,  # needs order-level depth
        "iceberg": False,    # needs refill tracking
        "regime": False,     # needs book state
    }


class EngineRunner:
    """Async background task that runs the OrderBookMetaPipeline and broadcasts results."""

    def __init__(self) -> None:
        self._pipeline = None
        self._feed = None
        self._task: Optional[asyncio.Task] = None
        self._feed_task: Optional[asyncio.Task] = None
        self._snapshot_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._frame_count: int = 0
        self._last_frame_time: float = 0.0
        self._running: bool = False
        self._live_feed_error: Optional[str] = None
        self._live_level: str = "L1"
        self._rth_only: bool = False
        # Wave 8.5-A: dropped_snapshots counts snapshots the runloop
        # swallowed due to pipeline.run() raising. Prior to this wave, an
        # operator saw the same get_status() output whether the feed was
        # warming up or crashing on every tick. See plan §8.5-A.
        self._dropped_snapshots: int = 0
        self._pipeline_errors: int = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        if self._last_frame_time == 0:
            return 0.0
        elapsed = time.monotonic() - self._last_frame_time
        return 1.0 / elapsed if elapsed > 0 else 0.0

    def init_pipeline(self) -> None:
        """Initialize the pipeline — C++ if available, Python fallback.

        Supports both installed-package (relative import) and PYTHONPATH=.
        (absolute import) launch modes.
        """
        if self._pipeline is None:
            # Wave 5 Phase 5A — attempt to build correlation components up-front
            # so they can be passed into the pure-Python pipeline constructor.
            # Fail-soft: if artifacts are missing, correlation stays off.
            correlation_engine, feature_accumulator = _build_correlation_components()

            # CppAcceleratedPipeline delegates analysis to Python and uses
            # C++ only for order book rendering. C++ analysis layers are
            # deprecated (parity bugs, synthetic book limitations).
            try:
                from ..pipeline_cpp import CppAcceleratedPipeline
                self._pipeline = CppAcceleratedPipeline()
                logger.info("Pipeline: Python analysis + C++ rendering")
                if correlation_engine is not None and hasattr(self._pipeline, "attach_correlation"):
                    self._pipeline.attach_correlation(correlation_engine, feature_accumulator)
                return
            except Exception as e:
                logger.debug("C++ rendering unavailable (%s), using pure Python", e)

            try:
                from ..pipeline import OrderBookMetaPipeline
                self._pipeline = OrderBookMetaPipeline(
                    correlation_engine=correlation_engine,
                    feature_accumulator=feature_accumulator,
                )
                logger.info("Pipeline: pure Python (analysis + rendering)")
            except ImportError:
                try:
                    import importlib
                    mod = importlib.import_module("pipeline")
                    self._pipeline = mod.OrderBookMetaPipeline(
                        correlation_engine=correlation_engine,
                        feature_accumulator=feature_accumulator,
                    )
                    logger.info("Pipeline: pure Python (absolute import)")
                except Exception as e2:
                    logger.error("FATAL: Could not import pipeline: %s", e2)
                    raise RuntimeError(f"Pipeline import failed: {e2}") from e2

    def reset_pipeline(self) -> None:
        """Reset pipeline state for a new session (cumulative counters, etc.)."""
        if self._pipeline and hasattr(self._pipeline, 'reset'):
            self._pipeline.reset()

    async def start_live_feed(
        self,
        symbol: str = "ES.c.0",
        dataset: str = "GLBX.MDP3",
        snapshot_interval_ms: int = 100,
        level: str = "L1",
    ) -> dict:
        """Start live Databento feed.

        level="L3" → MBO schema (requires MBO live subscription)
        level="L1" → MBP-1 schema (best bid/ask + trades, widely available)
        """
        _feed = _import_feed_module()
        DatabentoLiveFeed = _feed.DatabentoLiveFeed
        DatabentoL1LiveFeed = _feed.DatabentoL1LiveFeed

        api_key = os.environ.get("DATABENTO_API_KEY", "")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY not set")

        await self._stop_feed()
        self._live_feed_error = None
        self._live_level = level

        if level == "L3":
            self._feed = DatabentoLiveFeed(
                symbol=symbol,
                dataset=dataset,
                snapshot_interval_ms=snapshot_interval_ms,
                api_key=api_key,
            )
        else:
            # Default: L1 (MBP-1) — available on standard subscriptions
            self._feed = DatabentoL1LiveFeed(
                symbol=symbol,
                dataset=dataset,
                snapshot_interval_ms=snapshot_interval_ms,
                api_key=api_key,
            )

        await self._feed.connect()
        config.mode = "live"
        config.instrument = symbol.split(".")[0].upper()
        self.reset_pipeline()

        self._feed_task = asyncio.create_task(self._live_feed_loop())
        logger.info("Live feed started: %s %s level=%s", dataset, symbol, level)

        return {
            "status": "live",
            "symbol": symbol,
            "dataset": dataset,
            "level": level,
            "capabilities": _level_capabilities(level),
        }

    async def start_replay_feed(
        self,
        file_path: str,
        symbol: str = "ES",
        filter_symbol: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        rth_only: bool = False,
        snapshot_interval_ms: int = 100,
    ) -> dict:
        """Start Databento MBO file replay and begin processing.

        Supports both single-instrument files (es-mbo-*.dbn.zst) and
        full-exchange files (glbx-mdp3-*.dbn.zst) via filter_symbol.
        Time-range slicing via time_start/time_end (ISO format) or rth_only.
        """
        DatabentoReplayFeed = _import_feed_module().DatabentoReplayFeed

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        await self._stop_feed()

        # RTH = 09:30-16:00 ET = 13:30-20:00 UTC
        # We compute RTH bounds from the file date if rth_only is set
        # and no explicit time_start/time_end given
        _time_start = time_start
        _time_end = time_end

        if rth_only and not time_start and not time_end:
            # Extract date from filename to set RTH time bounds.
            # This avoids scanning millions of overnight records.
            import re
            date_match = re.search(r'(\d{4})-?(\d{2})-?(\d{2})', os.path.basename(file_path))
            if date_match:
                y, m, d = date_match.groups()
                _time_start = f"{y}-{m}-{d}T13:30:00"  # RTH start: 13:30 UTC
                _time_end = f"{y}-{m}-{d}T20:00:00"    # RTH end: 20:00 UTC
                logger.info("RTH auto-bounds: %s → %s", _time_start, _time_end)

        self._rth_only = rth_only
        self._feed = DatabentoReplayFeed(
            file_path=file_path,
            symbol=f"{symbol}.c.0" if not symbol.endswith(".c.0") else symbol,
            filter_symbol=filter_symbol or symbol.upper().split(".")[0],
            snapshot_interval_ms=snapshot_interval_ms,
            time_start=_time_start,
            time_end=_time_end,
        )
        await self._feed.connect()
        config.mode = "replay"
        config.instrument = symbol.upper().split(".")[0]
        self.reset_pipeline()

        self._feed_task = asyncio.create_task(self._replay_feed_loop())
        logger.info("Replay feed started: %s symbol=%s start=%s end=%s rth=%s",
                    file_path, symbol, time_start, time_end, rth_only)

        return {
            "status": "replay",
            "file": os.path.basename(file_path),
            "symbol": symbol.upper().split(".")[0],
            "filter_symbol": filter_symbol or symbol.upper().split(".")[0],
            "time_start": time_start,
            "time_end": time_end,
            "rth_only": rth_only,
        }

    async def start_demo_feed(self, symbol: str = "NQ") -> dict:
        """Start the bundled synthetic L3 feed (public demo mode).

        Streams a deterministic order book with a scripted 60s loop of
        microstructure events (momentum runs, institutional walls, spoofs,
        stop runs) so the detector overlays fire continuously. No external
        data or credentials required.
        """
        from ..ingestion.synthetic import SyntheticFeed

        await self._stop_feed()
        self._feed = SyntheticFeed(
            symbol=f"{symbol}.SYN",
            num_levels=20,
            tick_size=0.25,
            base_price=20000.0,
            volatility=0.0008,
        )
        config.mode = "demo"
        config.instrument = symbol.upper()
        self.reset_pipeline()
        self._feed_task = asyncio.create_task(self._demo_feed_loop())
        logger.info("Demo synthetic feed started (symbol=%s)", symbol)
        return {"status": "demo", "symbol": symbol.upper(), "synthetic": True}

    async def _demo_feed_loop(self) -> None:
        """Produce synthetic snapshots at ~30 Hz with a scripted 60s event loop."""
        from ..models import Side

        feed = self._feed
        i = 0
        period = 1800  # 60s @ 30 Hz
        try:
            while self._running and config.mode == "demo" and self._feed is not None:
                phase = i % period
                mid = getattr(feed, "_mid_price", 20000.0)
                # Scripted microstructure timeline (corrected SyntheticFeed API:
                # inject_momentum_run / inject_institutional_wall / inject_spoof /
                # inject_stop_run — there is no inject_layering/cancel_wall).
                if phase == 150:
                    feed.inject_momentum_run(Side.BID, num_levels=5)        # bull streak / cup-flip
                elif phase == 270:
                    feed.inject_institutional_wall(Side.ASK, mid + 1.0, 800)  # phantom wall
                elif phase == 390:
                    feed.inject_spoof(Side.ASK, mid + 1.0, 600)             # spoof / pull-before-touch
                elif phase == 510:
                    feed.inject_stop_run(Side.ASK)                          # stop run
                elif phase == 690:
                    feed.inject_momentum_run(Side.ASK, num_levels=5)        # bear streak
                elif phase in (870, 900, 930):
                    feed.inject_spoof(Side.BID, mid - 1.0, 500)             # layering-like cluster
                elif phase == 1050:
                    feed.inject_institutional_wall(Side.BID, mid - 1.0, 800)
                elif phase == 1230:
                    feed.inject_stop_run(Side.BID)

                snapshot = await feed.next()
                if snapshot is not None:
                    if snapshot.mid_price is not None:
                        await ws_manager.broadcast_tick({
                            "type": "price_tick",
                            "price": snapshot.mid_price,
                            "bid": snapshot.best_bid,
                            "ask": snapshot.best_ask,
                            "ts": snapshot.timestamp_ms,
                        })
                    await self.submit_snapshot(snapshot)
                i += 1
                await asyncio.sleep(1 / 30)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Demo feed loop error")
            config.mode = "paused"

    async def _live_feed_loop(self) -> None:
        """Consume live feed and submit snapshots to processing queue.

        Also emits price_tick messages on every BBO update for smooth
        chart rendering, independent of snapshot interval.
        """
        LiveFeedError = _import_feed_module().LiveFeedError
        try:
            while self._running and config.mode == "live" and self._feed:
                # Drain individual tick events for fast chart updates
                tick_from_event = False
                if hasattr(self._feed, 'next_event'):
                    event = self._feed.next_event()
                    if event is not None:
                        tick_from_event = True
                        tick_msg = {
                            "type": "price_tick",
                            "price": event.get("mid"),
                            "bid": event.get("bid"),
                            "ask": event.get("ask"),
                            "ts": event.get("ts"),
                        }
                        await ws_manager.broadcast_tick(tick_msg)

                snapshot = await self._feed.next()
                if snapshot is not None:
                    # Skip snapshots with no usable data (empty book, no trades) to avoid
                    # flooding the frontend with blank frames on initial connect (e.g. CL/GC).
                    has_bbo = snapshot.best_bid is not None or snapshot.best_ask is not None
                    has_trades = bool(getattr(snapshot, 'recent_trades', None))
                    if not has_bbo and not has_trades:
                        await asyncio.sleep(0.01)
                        continue

                    # Emit a fallback price_tick from snapshot BBO when the event queue
                    # was empty (common on initial connect for less-liquid instruments).
                    if not tick_from_event and snapshot.mid_price is not None:
                        tick_msg = {
                            "type": "price_tick",
                            "price": snapshot.mid_price,
                            "bid": snapshot.best_bid,
                            "ask": snapshot.best_ask,
                            "ts": snapshot.timestamp_ms,
                        }
                        await ws_manager.broadcast_tick(tick_msg)

                    await self.submit_snapshot(snapshot)
                else:
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except LiveFeedError as e:
            logger.error("Live feed error: %s", e)
            self._live_feed_error = str(e)
            config.mode = "paused"
        except Exception:
            logger.exception("Live feed loop error")
            config.mode = "paused"

    async def _replay_feed_loop(self) -> None:
        """Consume replay feed and submit snapshots to processing queue.

        feed.next() is CPU-bound (iterates millions of compressed records
        synchronously). We run it in a thread executor so the event loop
        stays free for WebSocket broadcasts and status polls.
        """
        import concurrent.futures
        rth_only = getattr(self, '_rth_only', False)
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay")

        def _blocking_next():
            """Call feed.next() in a thread — this is the slow part."""
            import asyncio as _aio
            # feed.next() is async in signature but synchronous in practice.
            # Run it in a throwaway event loop inside the thread.
            _loop = _aio.new_event_loop()
            try:
                return _loop.run_until_complete(self._feed.next())
            finally:
                _loop.close()

        try:
            while self._running and config.mode == "replay" and self._feed:
                snapshot = await loop.run_in_executor(executor, _blocking_next)
                if snapshot is None:
                    logger.info("Replay feed exhausted — %d events matched of %d scanned",
                                self._feed.events_processed,
                                getattr(self._feed, 'records_scanned', 0))
                    config.mode = "paused"
                    break

                # RTH filter: 13:30-20:00 UTC (09:30-16:00 ET)
                if rth_only:
                    hour_utc = (snapshot.timestamp_ms // 3_600_000) % 24
                    minute = (snapshot.timestamp_ms // 60_000) % 60
                    if (hour_utc < 13 or (hour_utc == 13 and minute < 30) or hour_utc >= 20):
                        continue

                await self.submit_snapshot(snapshot)

                # Back-pressure: let processing queue drain before feeding more
                while self._snapshot_queue.qsize() > 2:
                    await asyncio.sleep(0.05)

                # Pace replay to the feed's own snapshot interval (1x real-time).
                # Decoupled from frame_rate_limit, which only gates frontend output.
                feed_interval_ms = getattr(self._feed, "snapshot_interval_ms", 100)
                await asyncio.sleep(max(feed_interval_ms / 1000.0, 0.01))
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Replay feed loop error")
        finally:
            executor.shutdown(wait=False)

    async def _stop_feed(self) -> None:
        """Stop any running feed."""
        if self._feed_task and not self._feed_task.done():
            self._feed_task.cancel()
            try:
                await self._feed_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._feed:
            try:
                await self._feed.disconnect()
            except Exception:
                pass
        self._feed = None
        self._feed_task = None

    async def submit_snapshot(self, snapshot: object) -> None:
        """Submit an OrderBookSnapshot to be processed."""
        if self._snapshot_queue.full():
            try:
                self._snapshot_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._snapshot_queue.put(snapshot)

    async def _run_loop(self) -> None:
        """Main processing loop."""
        self.init_pipeline()
        self._running = True
        logger.info("Engine runner started (mode=%s)", config.mode)

        try:
            while self._running:
                if config.mode == "paused":
                    await asyncio.sleep(0.1)
                    continue

                try:
                    snapshot = await asyncio.wait_for(
                        self._snapshot_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                try:
                    # Update instrument from snapshot symbol if available
                    snap_sym = getattr(snapshot, 'symbol', None)
                    if snap_sym and snap_sym != config.instrument:
                        config.instrument = snap_sym.split('.')[0].upper()

                    frame = self._pipeline.run(snapshot)
                    self._frame_count += 1
                    self._last_frame_time = time.monotonic()

                    msg = serialize_frame(frame)
                    await ws_manager.broadcast(msg)

                except Exception:
                    # Wave 8.5-A: count dropped snapshots so get_status()
                    # surfaces a crashing pipeline vs an idle feed.
                    self._dropped_snapshots += 1
                    self._pipeline_errors += 1
                    logger.exception("Error processing snapshot")

                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info("Engine runner cancelled")
        finally:
            self._running = False
            logger.info("Engine runner stopped — %d frames processed", self._frame_count)

    def start(self) -> None:
        """Start the background runner as an asyncio task."""
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.get_event_loop().create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the background runner and any active feed."""
        self._running = False
        await self._stop_feed()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def get_status(self) -> dict:
        """Return current engine status."""
        status = {
            "running": self._running,
            "mode": config.mode,
            "instrument": config.instrument,
            "frame_count": self._frame_count,
            "fps": round(self.fps, 2),
            "ws_clients": ws_manager.client_count,
            "queue_size": self._snapshot_queue.qsize(),
            # Wave 8.5-A: observability counters. dropped_snapshots counts
            # snapshots the runloop lost due to pipeline.run() raising;
            # pipeline_errors is a superset (includes any swallow event).
            # correlation_stats surfaces the pipeline's own counters.
            "dropped_snapshots": self._dropped_snapshots,
            "pipeline_errors": self._pipeline_errors,
            "correlation_stats": (
                self._pipeline.correlation_stats
                if self._pipeline is not None
                and hasattr(self._pipeline, "correlation_stats")
                else {"ingest_errors": 0, "match_errors": 0}
            ),
        }
        if self._feed:
            scanned = getattr(self._feed, 'records_scanned', 0)
            matched = getattr(self._feed, 'events_processed', 0)
            status["records_scanned"] = scanned
            status["records_matched"] = matched
            status["replay_progress"] = -1
            status["data_level"] = getattr(self._feed, 'data_level', 'L1')
        if config.mode == "live":
            status["live_level"] = self._live_level
            status["capabilities"] = _level_capabilities(self._live_level)
        if self._pipeline and hasattr(self._pipeline, 'is_cpp'):
            status["cpp_accelerated"] = self._pipeline.is_cpp
        if self._live_feed_error:
            status["live_feed_error"] = self._live_feed_error
        return status


# Singleton
engine_runner = EngineRunner()
