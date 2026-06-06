"""
backtest_api.py — §11.2 Backtest API extension

Adds query param:
- cost_model=naive|realistic

Wires p6lab.execution.cost_model into BacktestRunner.score() path so
existing backtest runs can use realistic cost decomposition.
"""

from __future__ import annotations

from typing import Literal, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(tags=["backtest"])


class BacktestRequest(BaseModel):
    symbol: str
    start_ms: int
    end_ms: int
    strategy: str
    params: dict[str, Any] = {}


class BacktestResponse(BaseModel):
    ok: bool
    cost_model: Literal["naive", "realistic"]
    summary: dict[str, Any]
    trades: list[dict[str, Any]]


# Placeholder runner hooks
class _BacktestRunnerProxy:
    def run(self, req: BacktestRequest) -> list[dict[str, Any]]:
        # TODO wire to existing p6-v2 backtest runner
        return [
            {"trade_id": 1, "gross_pnl": 25.0, "slippage": 1.0, "commission": 0.5,
             "adverse_ticks_at_fill": 2, "unfilled": False},
            {"trade_id": 2, "gross_pnl": -6.0, "slippage": 1.2, "commission": 0.5,
             "adverse_ticks_at_fill": 3, "unfilled": True},
        ]


class _CostModelProxy:
    def score_naive(self, trade: dict[str, Any]) -> dict[str, Any]:
        total_cost = float(trade.get("slippage", 0)) + float(trade.get("commission", 0))
        net = float(trade.get("gross_pnl", 0)) - total_cost
        return {"total_cost": total_cost, "net_pnl": net, "model": "naive"}

    def score_realistic(self, trade: dict[str, Any]) -> dict[str, Any]:
        # Realistic decomposition (§6.3)
        crossed_spread_cost = float(trade.get("slippage", 0))
        commission = float(trade.get("commission", 0))
        adverse_selection_cost = float(trade.get("adverse_ticks_at_fill", 0)) * 0.25
        opportunity_cost = 1.5 if trade.get("unfilled", False) else 0.0
        total_cost = crossed_spread_cost + commission + adverse_selection_cost + opportunity_cost
        net = float(trade.get("gross_pnl", 0)) - total_cost
        return {
            "crossed_spread_cost": crossed_spread_cost,
            "commission": commission,
            "adverse_selection_cost": adverse_selection_cost,
            "opportunity_cost": opportunity_cost,
            "total_cost": total_cost,
            "net_pnl": net,
            "model": "realistic",
        }


RUNNER = _BacktestRunnerProxy()
COST = _CostModelProxy()


@router.post("/api/backtest/run", response_model=BacktestResponse)
def run_backtest(
    req: BacktestRequest,
    cost_model: Literal["naive", "realistic"] = Query("naive"),
) -> BacktestResponse:
    try:
        trades = RUNNER.run(req)

        scored = []
        for t in trades:
            if cost_model == "realistic":
                s = COST.score_realistic(t)
            else:
                s = COST.score_naive(t)
            t2 = {**t, **s}
            scored.append(t2)

        total_gross = sum(float(t.get("gross_pnl", 0)) for t in scored)
        total_cost = sum(float(t.get("total_cost", 0)) for t in scored)
        total_net = sum(float(t.get("net_pnl", 0)) for t in scored)

        summary = {
            "n_trades": len(scored),
            "gross_pnl": total_gross,
            "total_cost": total_cost,
            "net_pnl": total_net,
            "avg_net_per_trade": total_net / len(scored) if scored else 0.0,
        }

        return BacktestResponse(ok=True, cost_model=cost_model, summary=summary, trades=scored)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest run failed: {e}")
