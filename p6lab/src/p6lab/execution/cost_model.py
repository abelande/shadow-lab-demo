"""
p6lab.execution.cost_model
==========================
Realistic cost decomposition — §6.3 of the P6 Lab Spec.

Replaces the naive ``slippage + commission`` model in
WRAP_P6_BACKTEST.ipynb §07 with a four-component model:

    total_cost = crossed_spread_cost
               + commission
               + adverse_selection_cost
               + opportunity_cost

Components
----------
1. **Crossed-spread cost** (aggressive entry):
       cost = 0.5 × spread_ticks × tick_value × size
   For passive fills this is 0 (you earned the spread instead).

2. **Commission**:
       cost = per_contract_commission × size
   CME fee schedule: ~$2.04/contract all-in for NQ/ES retail.

3. **Adverse selection cost** (from FillOutcome.adverse_ticks_at_fill):
       cost = adverse_ticks × tick_value × size
   Measures how much the price moved against you by the time you got filled.
   Zero for market orders (instant fill, no queue).

4. **Opportunity cost** (for unfilled passive orders):
       cost = expected_edge × (1 − P(fill)) × size
   Captures the alpha you would have earned if the fill had occurred.
   Estimated from the strategy's historical mean edge per trade.

References
----------
- Spec §6.3 — cost decomposition, opportunity cost formula
- Spec §11.2 — backtest_api.py toggle: cost_model=naive|realistic
- Spec §9.3 notebook 05 §08 — realistic PnL backtest using this model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from p6lab.execution.fill_simulator import FillOutcome

# ---------------------------------------------------------------------------
# Constants (NQ defaults — override per instrument)
# ---------------------------------------------------------------------------

#: Commission per contract, all-in (CME + clearing + NFA + platform).
DEFAULT_COMMISSION_PER_CONTRACT: float = 2.04

#: NQ tick value ($5.00 per tick = 0.25 points).
DEFAULT_TICK_VALUE: float = 5.00

#: Default spread assumption in ticks (used when actual spread unavailable).
DEFAULT_SPREAD_TICKS: float = 1.0

#: Default expected edge in ticks per trade (for opportunity cost).
#: Calibrate from backtest output; this is a conservative placeholder.
DEFAULT_EXPECTED_EDGE_TICKS: float = 2.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CostBreakdown:
    """Full cost decomposition for one trade.

    Attributes
    ----------
    crossed_spread_cost:
        Half-spread cost for aggressive entries; 0 for passive fills.
    commission:
        Exchange + clearing + platform fees.
    adverse_selection_cost:
        Cost from adverse price movement at fill time.
    opportunity_cost:
        Alpha lost from unfilled orders.
    total_cost:
        Sum of all four components (always non-negative).
    cost_per_contract:
        total_cost / size.
    entry_type:
        'passive' or 'aggressive' — determines crossed_spread_cost.
    """

    crossed_spread_cost: float
    commission: float
    adverse_selection_cost: float
    opportunity_cost: float
    total_cost: float
    cost_per_contract: float
    entry_type: Literal["passive", "aggressive"]


# ---------------------------------------------------------------------------
# CostModel
# ---------------------------------------------------------------------------


class CostModel:
    """Realistic cost model for strategy backtesting.

    Parameters
    ----------
    commission_per_contract:
        All-in commission.
    tick_value:
        Dollar value per tick.
    default_spread_ticks:
        Assumed spread when actual spread not supplied.
    expected_edge_ticks:
        Expected strategy edge per trade (for opportunity cost).
        Calibrate from historical backtest mean PnL per trade.

    Usage
    -----
    >>> model = CostModel()
    >>> breakdown = model.compute(fill_outcome, entry_type='passive', spread_ticks=1.0)
    >>> print(breakdown.total_cost)

    Notebook 05 (§9.3 §08): computes costs for all bulk fills.
    backtest_api.py (§11.2): wires into BacktestRunner.score() path.
    """

    def __init__(
        self,
        commission_per_contract: float = DEFAULT_COMMISSION_PER_CONTRACT,
        tick_value: float = DEFAULT_TICK_VALUE,
        default_spread_ticks: float = DEFAULT_SPREAD_TICKS,
        expected_edge_ticks: float = DEFAULT_EXPECTED_EDGE_TICKS,
    ) -> None:
        self.commission_per_contract = commission_per_contract
        self.tick_value = tick_value
        self.default_spread_ticks = default_spread_ticks
        self.expected_edge_ticks = expected_edge_ticks

    def compute(
        self,
        fill_outcome: FillOutcome,
        entry_type: Literal["passive", "aggressive"] = "passive",
        spread_ticks: float | None = None,
        size: float | None = None,
        pfill: float | None = None,
    ) -> CostBreakdown:
        """Compute full cost breakdown for a single trade.

        Parameters
        ----------
        fill_outcome:
            Result from FillSimulator.
        entry_type:
            Whether the order was passive (limit) or aggressive (market/cross).
        spread_ticks:
            Actual spread at entry time in ticks.  None → use default.
        size:
            Override fill_outcome.filled_size if you want to cost the full
            intended size (including unfilled portion for opportunity cost).
        pfill:
            P(fill) estimate.  Used for opportunity cost.  If None, uses
            1.0 for filled orders and 0.0 for unfilled.

        Returns
        -------
        CostBreakdown
        """
        spread = spread_ticks if spread_ticks is not None else self.default_spread_ticks
        sz = size if size is not None else max(
            float(getattr(fill_outcome, "filled_size", 0.0) or 0.0), 1.0
        )
        filled_size = float(getattr(fill_outcome, "filled_size", 0.0) or 0.0)
        adverse_ticks = float(getattr(fill_outcome, "adverse_ticks_at_fill", 0.0) or 0.0)

        crossed = 0.0 if entry_type == "passive" else 0.5 * spread * self.tick_value * sz
        commission = self.commission_per_contract * sz
        adverse = adverse_ticks * self.tick_value * filled_size
        if pfill is None:
            pf = 1.0 if getattr(fill_outcome, "filled", False) else 0.0
        else:
            pf = max(0.0, min(1.0, float(pfill)))
        opportunity = self.expected_edge_ticks * self.tick_value * sz * (1.0 - pf)
        total = crossed + commission + max(0.0, adverse) + opportunity
        return CostBreakdown(
            crossed_spread_cost=float(crossed),
            commission=float(commission),
            adverse_selection_cost=float(max(0.0, adverse)),
            opportunity_cost=float(opportunity),
            total_cost=float(total),
            cost_per_contract=float(total / sz) if sz > 0 else 0.0,
            entry_type=entry_type,
        )

    def compute_batch(
        self,
        fill_outcomes: list[FillOutcome],
        entry_types: list[Literal["passive", "aggressive"]] | None = None,
        spread_ticks_series: list[float] | None = None,
    ) -> list[CostBreakdown]:
        """Compute costs for a batch of trades."""
        n = len(fill_outcomes)
        types = entry_types or ["passive"] * n
        spreads = spread_ticks_series or [self.default_spread_ticks] * n
        if len(types) != n or len(spreads) != n:
            raise ValueError("entry_types/spread_ticks_series length must match fill_outcomes")
        return [
            self.compute(fill_outcomes[i], entry_type=types[i], spread_ticks=spreads[i])
            for i in range(n)
        ]

    def compare_to_naive(
        self,
        realistic_costs: list[CostBreakdown],
        naive_slippage_ticks: float = 1.0,
        naive_commission: float = 2.04,
    ) -> dict:
        """Compare realistic costs to the naive (slippage + commission) model."""
        if not realistic_costs:
            return {
                "realistic_mean": 0.0, "naive_mean": 0.0,
                "realistic_total": 0.0, "naive_total": 0.0,
                "realistic_median": 0.0, "naive_median": 0.0,
                "cost_ratio": 1.0, "adverse_selection_pct": 0.0,
            }
        # Naive cost ignores opportunity + adverse; uses fixed slippage.
        # Use cost_per_contract * 1 as the per-trade naive figure.
        naive_per_trade = naive_slippage_ticks * self.tick_value + naive_commission
        realistic_per_trade = [c.cost_per_contract for c in realistic_costs]
        n = len(realistic_costs)
        rt = sum(c.total_cost for c in realistic_costs)
        nt = naive_per_trade * n  # one contract per trade as comparable unit
        adv_total = sum(c.adverse_selection_cost for c in realistic_costs)
        return {
            "realistic_mean": rt / n,
            "naive_mean": float(naive_per_trade),
            "realistic_total": float(rt),
            "naive_total": float(nt),
            "realistic_median": float(sorted(realistic_per_trade)[n // 2]),
            "naive_median": float(naive_per_trade),
            "cost_ratio": float(rt / nt) if nt > 0 else 1.0,
            "adverse_selection_pct": float(adv_total / rt) if rt > 0 else 0.0,
        }
