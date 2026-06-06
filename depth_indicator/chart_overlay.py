"""Main depth indicator renderer/composer."""
from __future__ import annotations
from typing import List
from ..models import (
    DepthBar,
    DepthIndicatorFrame,
    LevelState,
    OrderBookSnapshot,
    Side,
    StaircaseProfile,
    GameState,
    ForceVector,
    AuthenticityProfile,
    RegimeWeights,
    AggregatedSignal,
)
from .level_aggregator import LevelAggregator
from .bar_renderer import BarRenderer
from .dom_panel import DOMPanel
from .stats_header import StatsHeader
from .tape_feed import TapeFeed


class ChartOverlay:
    def __init__(self):
        self.level_agg = LevelAggregator()
        self.bar_renderer = BarRenderer()
        self.dom = DOMPanel()
        self.stats = StatsHeader()
        self.tape = TapeFeed()

    def render(
        self,
        snapshot: OrderBookSnapshot,
        staircase: StaircaseProfile,
        game_state: GameState,
        force_vector: ForceVector,
        authenticity: AuthenticityProfile,
        regime_weights: RegimeWeights,
        signal: AggregatedSignal,
        level_states: List[LevelState] | None = None,
    ) -> DepthIndicatorFrame:
        if level_states:
            bid_bars, ask_bars = self._render_from_level_states(level_states, authenticity.authenticity_score)
        else:
            # Fallback: build bars from raw snapshot bids/asks
            bids = self.level_agg.aggregate(snapshot.bids)
            asks = self.level_agg.aggregate(snapshot.asks)
            bids = [x for x in bids if x.side.name == 'BID']
            asks = [x for x in asks if x.side.name == 'ASK']
            bid_bars, ask_bars = self.bar_renderer.render(bids, asks, authenticity=authenticity.authenticity_score)

        dom_rows = self.dom.build(bid_bars, ask_bars)
        stats = self.stats.build(snapshot)
        tape = self.tape.build(snapshot.recent_trades)

        return DepthIndicatorFrame(
            timestamp_ms=snapshot.timestamp_ms,
            symbol=snapshot.symbol,
            bid_bars=bid_bars,
            ask_bars=ask_bars,
            dom_rows=dom_rows,
            tape=tape,
            stats=stats,
            staircase=staircase,
            game_state=game_state,
            force_vector=force_vector,
            authenticity=authenticity,
            regime_weights=regime_weights,
            direction=signal.direction,
            confidence=signal.confidence,
            urgency=signal.urgency,
            size_multiplier=signal.size_multiplier,
        )

    def _render_from_level_states(
        self,
        level_states: List[LevelState],
        default_authenticity: float,
    ) -> tuple[List[DepthBar], List[DepthBar]]:
        """Build DepthBar lists from LevelState objects with lifecycle metadata."""
        bid_levels = [ls for ls in level_states if ls.side == Side.BID]
        ask_levels = [ls for ls in level_states if ls.side == Side.ASK]

        # Sort: bids descending by price, asks ascending by price
        bid_levels.sort(key=lambda ls: ls.price, reverse=True)
        ask_levels.sort(key=lambda ls: ls.price)

        all_volumes = [ls.volume for ls in level_states]
        max_vol = max(all_volumes, default=1.0)

        def _build_bars(levels: List[LevelState]) -> List[DepthBar]:
            bars: List[DepthBar] = []
            cum = 0.0
            for ls in levels:
                cum += ls.volume
                bars.append(DepthBar(
                    price=ls.price,
                    side=ls.side,
                    volume=ls.volume,
                    order_count=ls.order_count,
                    cumulative_volume=cum,
                    bar_length=(ls.volume / max_vol) if max_vol else 0.0,
                    is_round_number=abs(ls.price - round(ls.price)) < 1e-9,
                    authenticity=ls.authenticity,
                    lifecycle=ls.lifecycle,
                    significance=ls.significance,
                    spoof_type=ls.spoof_type,
                    iceberg_suspected=ls.iceberg_suspected,
                ))
            return bars

        return _build_bars(bid_levels), _build_bars(ask_levels)
