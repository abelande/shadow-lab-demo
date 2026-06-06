"""
p6lab.features.cross_asset — Wave 7 Phase 7B

PLASMA-style cross-asset adjacency + network momentum. Ported from
``p4-clones/plasma/adjacency.py`` + ``momentum.py``.

Two pieces:

1. **AdjacencyState** — multi-scale rolling correlation over windows
   [20, 60, 120] with exponential decay weighting the shorter (more
   recent) windows more heavily. ``AdjacencyState.matrix()`` returns the
   current N×N adjacency matrix; zero or below-threshold correlations
   drop to 0 to keep the graph sparse.

2. **MomentumState** — per-symbol z-scored returns stacked by lookback
   plus a network-enhanced variant that spreads each symbol's momentum
   across its adjacency neighbors. Emits per-symbol scalars
   (``network_momentum``, ``peer_correlation_avg``, ``peer_phase_lead``).

Exported:
    CROSS_ASSET_FEATURE_NAMES   tuple[str, ...]
    AdjacencyState              dataclass
    MomentumState               dataclass
    update_cross_asset(state_pack, ts_ms, symbol_to_mid)
    snapshot_cross_asset_features(state_pack, symbol) → dict
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np


ADJACENCY_WINDOWS: tuple[int, ...] = (20, 60, 120)
ADJACENCY_DECAY: float = 0.7
ADJACENCY_THRESHOLD: float = 0.15
DEFAULT_LOOKBACKS: tuple[int, ...] = (5, 10, 21, 63)
BUFFER_LEN: int = max(ADJACENCY_WINDOWS) + max(DEFAULT_LOOKBACKS) + 10
NETWORK_DAMPING: float = 0.5


CROSS_ASSET_FEATURE_NAMES: tuple[str, ...] = (
    "network_momentum",
    "peer_correlation_avg",
    "peer_phase_lead",
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class _SymbolSeries:
    mids: Deque[float] = field(default_factory=lambda: deque(maxlen=BUFFER_LEN))

    def append(self, mid: float) -> None:
        self.mids.append(float(mid))

    def returns(self) -> np.ndarray:
        if len(self.mids) < 2:
            return np.asarray([], dtype=float)
        arr = np.asarray(self.mids, dtype=float)
        return np.diff(np.log(np.where(arr > 0, arr, 1e-9)))


@dataclass
class AdjacencyState:
    """N×N cross-symbol adjacency matrix, rebuilt each ``update``."""
    symbols: list[str] = field(default_factory=list)
    matrix_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    series: dict[str, _SymbolSeries] = field(default_factory=dict)

    def reset(self) -> None:
        self.symbols = []
        self.matrix_ = np.zeros((0, 0))
        self.series = {}


@dataclass
class MomentumState:
    """Individual + network momentum. One vector per symbol."""
    individual: dict[str, np.ndarray] = field(default_factory=dict)
    network: dict[str, np.ndarray] = field(default_factory=dict)

    def reset(self) -> None:
        self.individual = {}
        self.network = {}


@dataclass
class CrossAssetState:
    """Packaged state for the cross-asset runner (Phase 7A)."""
    adjacency: AdjacencyState = field(default_factory=AdjacencyState)
    momentum: MomentumState = field(default_factory=MomentumState)

    def reset(self) -> None:
        self.adjacency.reset()
        self.momentum.reset()


# ---------------------------------------------------------------------------
# Update + snapshot
# ---------------------------------------------------------------------------


def update_cross_asset(
    state: CrossAssetState,
    *,
    ts_ms: int,
    symbol_to_mid: dict[str, float],
) -> None:
    """Ingest one multi-symbol tick and recompute adjacency + momentum.

    ``symbol_to_mid`` must include every symbol that should participate;
    symbols absent from this tick are assumed to have no update (prior
    mid carried forward only if they were previously registered)."""
    for sym, mid in symbol_to_mid.items():
        series = state.adjacency.series.setdefault(sym, _SymbolSeries())
        series.append(mid)
        if sym not in state.adjacency.symbols:
            state.adjacency.symbols.append(sym)

    state.adjacency.matrix_ = _adaptive_adjacency(state.adjacency)
    _update_momentum(state)


def _adaptive_adjacency(adj: AdjacencyState) -> np.ndarray:
    """Return the multi-scale rolling correlation matrix."""
    syms = list(adj.symbols)
    n = len(syms)
    if n < 2:
        return np.zeros((n, n))
    total_weight = 0.0
    agg = np.zeros((n, n))
    for scale_idx, window in enumerate(ADJACENCY_WINDOWS):
        weight = ADJACENCY_DECAY ** scale_idx
        mat = _window_correlation(adj, window)
        if mat is None:
            continue
        agg += weight * mat
        total_weight += weight
    if total_weight <= 0.0:
        return np.zeros((n, n))
    agg /= total_weight
    # Threshold + zero-diagonal
    agg[np.abs(agg) < ADJACENCY_THRESHOLD] = 0.0
    np.fill_diagonal(agg, 0.0)
    return agg


def _window_correlation(adj: AdjacencyState, window: int) -> np.ndarray | None:
    """Correlation matrix computed on the last ``window`` returns. Returns
    ``None`` when any symbol has too few samples."""
    syms = adj.symbols
    cols: list[np.ndarray] = []
    for sym in syms:
        ret = adj.series[sym].returns()
        if ret.size < window:
            return None
        cols.append(ret[-window:])
    mat = np.vstack(cols)   # (n_syms, window)
    # Zero-variance rows break corrcoef
    stds = mat.std(axis=1)
    if np.any(stds <= 0.0):
        return np.zeros((len(syms), len(syms)))
    return np.corrcoef(mat)


def _update_momentum(state: CrossAssetState) -> None:
    syms = state.adjacency.symbols
    indiv: dict[str, np.ndarray] = {}
    for sym in syms:
        ret = state.adjacency.series[sym].returns()
        vec = np.zeros(len(DEFAULT_LOOKBACKS), dtype=float)
        for i, lb in enumerate(DEFAULT_LOOKBACKS):
            if ret.size >= lb:
                tail = ret[-lb:]
                mu = float(tail.mean())
                sigma = float(tail.std())
                vec[i] = mu / sigma if sigma > 0.0 else 0.0
        indiv[sym] = vec
    state.momentum.individual = indiv

    network: dict[str, np.ndarray] = {}
    if not indiv:
        state.momentum.network = {}
        return
    A = state.adjacency.matrix_
    # Stack individual momentum for matrix math
    stacked = np.asarray([indiv[s] for s in syms])   # (n, lookbacks)
    if A.shape == (len(syms), len(syms)) and A.any():
        # Row-normalize to prevent blowup
        row_sum = np.abs(A).sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        A_norm = A / row_sum
        propagated = A_norm @ stacked
        blended = NETWORK_DAMPING * propagated + (1.0 - NETWORK_DAMPING) * stacked
    else:
        blended = stacked
    for i, sym in enumerate(syms):
        network[sym] = blended[i]
    state.momentum.network = network


def snapshot_cross_asset_features(
    state: CrossAssetState, symbol: str,
) -> dict[str, float]:
    """Per-symbol scalars. Returns zeros when ``symbol`` isn't registered."""
    syms = state.adjacency.symbols
    if symbol not in syms:
        return {name: 0.0 for name in CROSS_ASSET_FEATURE_NAMES}

    idx = syms.index(symbol)
    # network_momentum = mean over lookbacks of the network vector
    net_vec = state.momentum.network.get(symbol)
    net_mom = float(np.mean(net_vec)) if net_vec is not None and net_vec.size else 0.0

    # peer_correlation_avg = mean off-diagonal row
    A = state.adjacency.matrix_
    if A.shape == (len(syms), len(syms)) and len(syms) >= 2:
        row = A[idx]
        peer_corr = float(np.mean(np.abs(row)) * len(syms) / max(len(syms) - 1, 1))
    else:
        peer_corr = 0.0

    # peer_phase_lead ≈ this symbol's momentum mean minus the cross-symbol mean
    indiv_vec = state.momentum.individual.get(symbol, np.zeros(1))
    my_mean = float(np.mean(indiv_vec)) if indiv_vec.size else 0.0
    peer_mean = 0.0
    if state.momentum.individual:
        peer_mean = float(
            np.mean([
                np.mean(v) for s, v in state.momentum.individual.items()
                if s != symbol and v.size
            ] or [0.0])
        )
    peer_phase_lead = my_mean - peer_mean

    return {
        "network_momentum": net_mom,
        "peer_correlation_avg": peer_corr,
        "peer_phase_lead": float(peer_phase_lead),
    }
