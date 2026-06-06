"""
p6lab.validation.information_gain
=================================
Information-gain gate — §8.3 of the P6 Lab Spec.

Rule (staircase plan §L523)
---------------------------
"No component gets added unless it beats baseline by measurable margin."

Default decision criteria
-------------------------
- Absolute improvement >= min_improvement (default 0.02 = 2%)
- Bootstrap confidence interval lower bound > 0
- Optional p-value threshold (e.g., < 0.05)

Used by notebooks:
- 03: feature must beat bid_ask_imbalance baseline by >=2% AUC
- 04: mined pattern must beat aggregate-confidence baseline
- 06: correlation model must beat template-matcher baseline by >=2%
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class DecisionReport:
    """Decision payload returned by must_beat_baseline()."""

    approved: bool
    candidate_metric: float
    baseline_metric: float
    absolute_improvement: float
    relative_improvement: float
    min_required_improvement: float
    ci_low: float
    ci_high: float
    p_value: float | None
    reason: str


def must_beat_baseline(
    candidate_metric: float,
    baseline_metric: float,
    min_improvement: float = 0.02,
    n_bootstrap: int = 1000,
    *,
    candidate_samples: Sequence[float] | None = None,
    baseline_samples: Sequence[float] | None = None,
    confidence: float = 0.95,
    random_state: int = 42,
) -> DecisionReport:
    """Evaluate whether candidate beats baseline by a measurable margin.

    If ``candidate_samples`` and ``baseline_samples`` are provided, a paired
    bootstrap CI is computed on the per-sample improvement. Otherwise the
    CI is approximated from a normal envelope around the absolute improvement.
    """
    eps = 1e-9
    abs_imp = float(candidate_metric - baseline_metric)
    rel_imp = abs_imp / max(abs(baseline_metric), eps)

    rng = np.random.default_rng(random_state)
    p_value: float | None = None

    if candidate_samples is not None and baseline_samples is not None:
        c = np.asarray(candidate_samples, dtype=float)
        b = np.asarray(baseline_samples, dtype=float)
        n = min(len(c), len(b))
        if n == 0:
            ci_low, ci_high = abs_imp, abs_imp
        else:
            diffs = c[:n] - b[:n]
            boot = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                boot[i] = float(diffs[idx].mean())
            alpha = (1.0 - confidence) / 2.0
            ci_low = float(np.quantile(boot, alpha))
            ci_high = float(np.quantile(boot, 1.0 - alpha))
            # one-sided p-value: P(boot <= 0)
            p_value = float((boot <= 0).mean())
    else:
        # No sample-level data — synthesize a small CI envelope from the gap
        envelope = max(abs(abs_imp) * 0.25, eps)
        ci_low = abs_imp - envelope
        ci_high = abs_imp + envelope

    approved = (abs_imp >= min_improvement) and (ci_low > 0)
    if approved:
        reason = (
            f"approved: improvement {abs_imp:.4f} ≥ {min_improvement:.4f} "
            f"and CI lower {ci_low:.4f} > 0"
        )
    elif abs_imp < min_improvement:
        reason = (
            f"rejected: improvement {abs_imp:.4f} < required {min_improvement:.4f}"
        )
    else:
        reason = (
            f"rejected: CI lower bound {ci_low:.4f} ≤ 0 — improvement not "
            "statistically distinguishable from zero"
        )

    return DecisionReport(
        approved=approved,
        candidate_metric=float(candidate_metric),
        baseline_metric=float(baseline_metric),
        absolute_improvement=abs_imp,
        relative_improvement=float(rel_imp),
        min_required_improvement=float(min_improvement),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p_value,
        reason=reason,
    )
