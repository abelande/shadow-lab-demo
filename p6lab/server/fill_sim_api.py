"""
fill_sim_api.py — §11.5 Interactive Fill Simulator API

Endpoint:
- POST /api/fill_sim/interactive

Used by web/js/virtual_order_tool.js for one-order "what if" simulation.
Batch notebook 05 uses fill_simulator.simulate_bulk directly (not this API).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["fill-sim"])


class InteractiveOrderRequest(BaseModel):
    symbol: str = Field(..., description="Instrument symbol, e.g. NQ")
    timestamp_ms: int = Field(..., ge=0)
    side: Literal["buy", "sell"]
    size: float = Field(..., gt=0)
    order_type: Literal["limit", "market", "step-ahead"] = "limit"
    price: float | None = Field(default=None)
    max_horizon_sec: int = Field(default=120, ge=1, le=3600)
    context: dict[str, Any] | None = None


class QueueSnapshotModel(BaseModel):
    timestamp_ms: int
    queue_position: float | None = None
    volume_ahead: float | None = None
    p_fill_estimate: float | None = None
    status: str | None = None


class FillOutcomeResponse(BaseModel):
    filled: bool
    filled_size: float
    fill_timestamp_ms: int | None
    queue_position_at_entry: float
    queue_position_at_fill: float | None
    adverse_ticks_at_fill: int
    realized_pnl: float
    fill_reason: Literal["full", "partial", "adverse_exit", "timeout", "cancelled"]
    trajectory: list[QueueSnapshotModel]


# Placeholder simulator hook; wire to p6lab.execution.fill_simulator in real integration
class _SimulatorProxy:
    def simulate_interactive(self, order_spec: dict[str, Any]) -> dict[str, Any]:
        # TODO: Replace with real call:
        # from p6lab.execution.fill_simulator import FillSimulator
        # outcome = FillSimulator(...).simulate_interactive(order_spec)
        # return asdict(outcome)
        ts = order_spec["timestamp_ms"]
        size = order_spec["size"]
        return {
            "filled": False,
            "filled_size": 0.0,
            "fill_timestamp_ms": None,
            "queue_position_at_entry": 120.0,
            "queue_position_at_fill": None,
            "adverse_ticks_at_fill": 0,
            "realized_pnl": -0.25,
            "fill_reason": "timeout",
            "trajectory": [
                {"timestamp_ms": ts + i * 250, "queue_position": max(0, 120 - i * 8), "volume_ahead": max(0, 120 - i * 10),
                 "p_fill_estimate": min(0.99, i * 0.06), "status": "pending"}
                for i in range(1, 12)
            ] + [{"timestamp_ms": ts + 12 * 250, "queue_position": 32, "volume_ahead": 29, "p_fill_estimate": 0.64, "status": "timeout"}],
        }


SIMULATOR = _SimulatorProxy()


@router.post("/api/fill_sim/interactive", response_model=FillOutcomeResponse)
def simulate_interactive(req: InteractiveOrderRequest) -> FillOutcomeResponse:
    try:
        order_spec = req.model_dump()

        # Validation by order type
        if req.order_type in {"limit", "step-ahead"} and req.price is None:
            raise HTTPException(status_code=400, detail="price is required for limit/step-ahead orders")

        outcome = SIMULATOR.simulate_interactive(order_spec)
        return FillOutcomeResponse(**outcome)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interactive fill sim failed: {e}")
