"""Pedagogical visualization of Combinatorial Purged Cross-Validation (CPCV).

Renders the purge + embargo structure that makes time-series CV leakage-free
(López de Prado, *Advances in Financial Machine Learning*, §7). Self-contained:
no data, no fitted model — purely illustrative of the method p6lab uses.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_COLORS = {"TRAIN": "#1f77b4", "PURGE": "#ff9896", "TEST": "#d62728", "EMBARGO": "#c5b0d5"}


def render_cpcv(n_periods: int = 24, test_len: int = 3, embargo: int = 1, purge: int = 1,
                n_folds: int = 6, out_path: str | Path | None = None):
    """Draw N folds of a combinatorial purged scheme over `n_periods` blocks."""
    fig, axes = plt.subplots(n_folds, 1, figsize=(11, 1.05 * n_folds))
    if n_folds == 1:
        axes = [axes]

    test_starts = list(range(0, n_periods - test_len + 1,
                             max(1, (n_periods - test_len) // max(1, n_folds - 1))))[:n_folds]

    for fold, ax in zip(range(n_folds), axes):
        ts = test_starts[fold] if fold < len(test_starts) else 0
        for t in range(n_periods):
            if ts <= t < ts + test_len:
                kind = "TEST"
            elif ts - purge <= t < ts:
                kind = "PURGE"
            elif ts + test_len <= t < ts + test_len + embargo:
                kind = "EMBARGO"
            else:
                kind = "TRAIN"
            ax.add_patch(mpatches.Rectangle((t, 0), 0.92, 1, color=_COLORS[kind]))
        ax.set_xlim(0, n_periods); ax.set_ylim(0, 1)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(f"fold {fold + 1}", rotation=0, ha="right", va="center", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle("Combinatorial Purged Cross-Validation — purge + embargo (illustrative)",
                 fontsize=12, y=0.99)
    fig.legend(handles=[mpatches.Patch(color=c, label=k) for k, c in _COLORS.items()],
               loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "notebooks" / "cpcv_figure.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render_cpcv(out_path=out)
    print(f"Wrote {out}")
