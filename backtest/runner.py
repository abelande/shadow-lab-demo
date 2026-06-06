"""Backtest runner — drives replay feed through the pipeline and tracks trades."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ..models import (
    DepthIndicatorFrame, OrderBookSnapshot, Side,
)


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Trade:
    """Record of a completed trade."""
    entry_time: int
    exit_time: int
    side: PositionSide
    entry_price: float
    exit_price: float
    pnl: float
    regime: str = "UNKNOWN"

    @property
    def holding_time_ms(self) -> int:
        return self.exit_time - self.entry_time


@dataclass
class BacktestConfig:
    """Configuration for the backtest runner."""
    direction_threshold: float = 0.3
    exit_threshold: float = -0.1
    min_confidence: float = 0.5
    stop_loss_pct: float = 0.02
    position_size: float = 1.0


@dataclass
class BacktestResult:
    """Complete output of a backtest run."""
    trades: List[Trade] = field(default_factory=list)
    frames: List[DepthIndicatorFrame] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)
    config: Optional[BacktestConfig] = None
    start_time: int = 0
    end_time: int = 0


class BacktestRunner:
    """Runs historical order book data through the pipeline and tracks trades.

    Takes a replay feed and an optional pipeline. Processes each snapshot,
    records frames, and tracks hypothetical trades based on signal thresholds.

    Args:
        feed: A BaseFeed instance (typically ReplayFeed)
        pipeline: Optional OrderBookMetaPipeline for signal generation.
                  If None, uses raw snapshot data without signal processing.
        config: Backtest configuration parameters
    """

    def __init__(
        self,
        feed,
        pipeline=None,
        config: Optional[BacktestConfig] = None,
    ):
        self.feed = feed
        self.pipeline = pipeline
        self.config = config or BacktestConfig()

        self._trades: List[Trade] = []
        self._frames: List[DepthIndicatorFrame] = []
        self._position: PositionSide = PositionSide.FLAT
        self._entry_price: float = 0.0
        self._entry_time: int = 0
        self._entry_regime: str = "UNKNOWN"
        self._equity: float = 0.0
        self._equity_curve: List[dict] = []

    async def run(self) -> BacktestResult:
        """Execute the full backtest.

        Returns:
            BacktestResult with all trades, frames, and equity curve.
        """
        await self.feed.connect()
        start_time = 0
        end_time = 0

        while True:
            snapshot = await self.feed.next()
            if snapshot is None:
                break

            if start_time == 0:
                start_time = snapshot.timestamp_ms
            end_time = snapshot.timestamp_ms

            frame = await self._process_snapshot(snapshot)
            self._frames.append(frame)
            self._evaluate_signal(frame, snapshot)

        # Close any open position at end
        if self._position != PositionSide.FLAT and self._frames:
            last = self._frames[-1]
            mid = self._get_mid_price_from_frame(last)
            if mid:
                self._close_position(mid, end_time)

        await self.feed.disconnect()

        return BacktestResult(
            trades=self._trades,
            frames=self._frames,
            equity_curve=self._equity_curve,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
        )

    async def _process_snapshot(self, snapshot: OrderBookSnapshot) -> DepthIndicatorFrame:
        """Process a snapshot through the pipeline or create a basic frame."""
        if self.pipeline is not None:
            try:
                frame = self.pipeline.process(snapshot)
                return frame
            except Exception:
                pass

        return DepthIndicatorFrame(
            timestamp_ms=snapshot.timestamp_ms,
            symbol=snapshot.symbol,
            direction=0.0,
            confidence=0.0,
        )

    def _evaluate_signal(self, frame: DepthIndicatorFrame, snapshot: OrderBookSnapshot) -> None:
        """Evaluate trading signals and manage positions."""
        cfg = self.config
        mid = snapshot.mid_price
        if mid is None:
            return

        regime = "UNKNOWN"
        if frame.regime_weights:
            regime = frame.regime_weights.regime.value if hasattr(frame.regime_weights.regime, 'value') else str(frame.regime_weights.regime)

        # Record equity
        unrealized = 0.0
        if self._position == PositionSide.LONG:
            unrealized = (mid - self._entry_price) * cfg.position_size
        elif self._position == PositionSide.SHORT:
            unrealized = (self._entry_price - mid) * cfg.position_size

        self._equity_curve.append({
            "timestamp": frame.timestamp_ms,
            "equity": self._equity + unrealized,
            "position": self._position.value,
        })

        abstain = False
        if frame.regime_weights:
            abstain = frame.regime_weights.abstain

        if self._position == PositionSide.FLAT:
            if (not abstain
                    and frame.confidence >= cfg.min_confidence
                    and abs(frame.direction) >= cfg.direction_threshold):
                if frame.direction > 0:
                    self._position = PositionSide.LONG
                else:
                    self._position = PositionSide.SHORT
                self._entry_price = mid
                self._entry_time = frame.timestamp_ms
                self._entry_regime = regime
        else:
            should_exit = False

            if self._position == PositionSide.LONG and frame.direction < cfg.exit_threshold:
                should_exit = True
            elif self._position == PositionSide.SHORT and frame.direction > -cfg.exit_threshold:
                should_exit = True

            if self._position == PositionSide.LONG:
                if mid < self._entry_price * (1 - cfg.stop_loss_pct):
                    should_exit = True
            elif self._position == PositionSide.SHORT:
                if mid > self._entry_price * (1 + cfg.stop_loss_pct):
                    should_exit = True

            if should_exit:
                self._close_position(mid, frame.timestamp_ms)

    def _close_position(self, exit_price: float, exit_time: int) -> None:
        """Close the current position and record the trade."""
        if self._position == PositionSide.LONG:
            pnl = (exit_price - self._entry_price) * self.config.position_size
        elif self._position == PositionSide.SHORT:
            pnl = (self._entry_price - exit_price) * self.config.position_size
        else:
            return

        trade = Trade(
            entry_time=self._entry_time,
            exit_time=exit_time,
            side=self._position,
            entry_price=self._entry_price,
            exit_price=exit_price,
            pnl=pnl,
            regime=self._entry_regime,
        )
        self._trades.append(trade)
        self._equity += pnl
        self._position = PositionSide.FLAT

    def _get_mid_price_from_frame(self, frame: DepthIndicatorFrame) -> Optional[float]:
        """Extract mid price from frame's bid/ask bars."""
        if frame.bid_bars and frame.ask_bars:
            return (frame.bid_bars[0].price + frame.ask_bars[0].price) / 2.0
        return None
