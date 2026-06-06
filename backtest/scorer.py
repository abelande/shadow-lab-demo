"""Backtest scoring — compute performance metrics from trade results."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ScoreCard:
    """Complete performance scorecard."""
    total_pnl: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_holding_time_ms: float = 0.0
    regime_hit_rates: Dict[str, dict] = field(default_factory=dict)


class BacktestScorer:
    """Computes performance metrics from backtest results.

    Metrics:
        - Total PnL, Win rate, Avg win / Avg loss
        - Sharpe ratio (annualized from trade returns)
        - Max drawdown (peak-to-trough)
        - Profit factor (gross profit / gross loss)
        - Total trades, avg holding time
        - Hit rate per regime
    """

    def __init__(self, trades: List = None, equity_curve: List[dict] = None):
        self.trades = trades or []
        self.equity_curve = equity_curve or []

    def score(self) -> ScoreCard:
        """Compute all metrics and return a ScoreCard."""
        card = ScoreCard()

        if not self.trades:
            return card

        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        card.total_trades = len(self.trades)
        card.total_pnl = sum(pnls)
        card.win_rate = len(wins) / card.total_trades if card.total_trades > 0 else 0.0
        card.avg_win = sum(wins) / len(wins) if wins else 0.0
        card.avg_loss = sum(losses) / len(losses) if losses else 0.0
        card.win_loss_ratio = (
            abs(card.avg_win / card.avg_loss) if card.avg_loss != 0 else float("inf")
        )

        card.sharpe_ratio = self._sharpe(pnls)

        card.max_drawdown, card.max_drawdown_pct = self._max_drawdown(pnls)

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        card.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        holding_times = [t.holding_time_ms for t in self.trades]
        card.avg_holding_time_ms = sum(holding_times) / len(holding_times) if holding_times else 0.0

        card.regime_hit_rates = self._regime_hit_rates()

        return card

    def _sharpe(self, pnls: List[float], annual_factor: float = 2016.0) -> float:
        """Compute annualized Sharpe ratio."""
        if len(pnls) < 2:
            return 0.0
        mean = sum(pnls) / len(pnls)
        variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(annual_factor)

    def _max_drawdown(self, pnls: List[float]) -> tuple:
        """Compute max drawdown in absolute and percentage terms."""
        if not pnls:
            return 0.0, 0.0

        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        max_dd_pct = 0.0

        for pnl in pnls:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd / peak if peak > 0 else 0.0

        return max_dd, max_dd_pct

    def _regime_hit_rates(self) -> Dict[str, dict]:
        """Compute win rate and trade count per regime."""
        regimes: Dict[str, List] = {}
        for t in self.trades:
            regime = t.regime if hasattr(t, "regime") else "UNKNOWN"
            if regime not in regimes:
                regimes[regime] = []
            regimes[regime].append(t)

        result = {}
        for regime, trades in regimes.items():
            wins = sum(1 for t in trades if t.pnl > 0)
            result[regime] = {
                "trades": len(trades),
                "wins": wins,
                "hit_rate": wins / len(trades) if trades else 0.0,
                "total_pnl": sum(t.pnl for t in trades),
            }
        return result
