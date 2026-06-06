"""Backtest report generation — Markdown + JSON equity curve."""
from __future__ import annotations

import datetime
import json
import os
from typing import Optional

from .runner import BacktestResult
from .scorer import BacktestScorer, ScoreCard


def generate_report(
    results: BacktestResult,
    output_path: str,
    title: str = "Staircase Terminal — Backtest Report",
) -> str:
    """Generate a comprehensive Markdown backtest report.

    Args:
        results: BacktestResult from BacktestRunner.run()
        output_path: Path to write the report (e.g. "report.md")
        title: Report title

    Returns:
        The report as a string
    """
    scorer = BacktestScorer(
        trades=results.trades,
        equity_curve=results.equity_curve,
    )
    card = scorer.score()

    lines = []
    lines.append(f"# {title}\n")
    lines.append(f"**Period:** {_fmt_ts(results.start_time)} → {_fmt_ts(results.end_time)}\n")

    # Summary stats
    lines.append("## Summary Statistics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total PnL | {card.total_pnl:.4f} |")
    lines.append(f"| Total Trades | {card.total_trades} |")
    lines.append(f"| Win Rate | {card.win_rate:.1%} |")
    lines.append(f"| Avg Win | {card.avg_win:.4f} |")
    lines.append(f"| Avg Loss | {card.avg_loss:.4f} |")
    lines.append(f"| Win/Loss Ratio | {card.win_loss_ratio:.2f} |")
    lines.append(f"| Sharpe Ratio | {card.sharpe_ratio:.2f} |")
    lines.append(f"| Max Drawdown | {card.max_drawdown:.4f} |")
    lines.append(f"| Max Drawdown % | {card.max_drawdown_pct:.1%} |")
    lines.append(f"| Profit Factor | {card.profit_factor:.2f} |")
    lines.append(f"| Avg Holding Time | {_fmt_duration(card.avg_holding_time_ms)} |")
    lines.append("")

    # Config
    if results.config:
        lines.append("## Configuration\n")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Direction Threshold | {results.config.direction_threshold} |")
        lines.append(f"| Exit Threshold | {results.config.exit_threshold} |")
        lines.append(f"| Min Confidence | {results.config.min_confidence} |")
        lines.append(f"| Stop Loss % | {results.config.stop_loss_pct:.1%} |")
        lines.append(f"| Position Size | {results.config.position_size} |")
        lines.append("")

    # Regime breakdown
    if card.regime_hit_rates:
        lines.append("## Performance by Regime\n")
        lines.append("| Regime | Trades | Win Rate | Total PnL |")
        lines.append("|--------|--------|----------|-----------|")
        for regime, stats in sorted(card.regime_hit_rates.items()):
            lines.append(
                f"| {regime} | {stats['trades']} | "
                f"{stats['hit_rate']:.1%} | {stats['total_pnl']:.4f} |"
            )
        lines.append("")

    # Trade log
    if results.trades:
        lines.append("## Trade Log\n")
        lines.append("| # | Entry Time | Exit Time | Side | Entry | Exit | PnL | Regime |")
        lines.append("|---|-----------|----------|------|-------|------|-----|--------|")
        for i, t in enumerate(results.trades, 1):
            lines.append(
                f"| {i} | {_fmt_ts(t.entry_time)} | {_fmt_ts(t.exit_time)} | "
                f"{t.side.value} | {t.entry_price:.4f} | {t.exit_price:.4f} | "
                f"{t.pnl:+.4f} | {t.regime} |"
            )
        lines.append("")

    report = "\n".join(lines)

    # Write report
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    # Write equity curve JSON alongside
    eq_path = output_path.rsplit(".", 1)[0] + "_equity.json"
    with open(eq_path, "w") as f:
        json.dump(results.equity_curve, f, indent=2)

    return report


def _fmt_ts(ts_ms: int) -> str:
    """Format millisecond timestamp to readable string."""
    if ts_ms == 0:
        return "N/A"
    dt = datetime.datetime.fromtimestamp(ts_ms / 1000.0, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_duration(ms: float) -> str:
    """Format duration in milliseconds to human-readable."""
    if ms == 0:
        return "N/A"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"
