"""P6 Backtest Engine — backtesting framework for the Staircase Terminal."""

from .runner import BacktestRunner
from .scorer import BacktestScorer
from .report import generate_report

__all__ = ["BacktestRunner", "BacktestScorer", "generate_report"]
