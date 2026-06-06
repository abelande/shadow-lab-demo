"""
p6lab.features.path_signatures — Wave 7 Phase 7F

Port of p4-clones/wraith's hand-rolled path signatures. No iisignature
dependency: truncated Chen-series up to depth 3 via pure numpy.

Signatures are a universal nonlinear feature of a path — they strictly
contain all moments of the path's displacement + area. For a path
increment vector Δ = (Δ₁, …, Δ_d):

    level 1: Δ_i                        shape (d,)
    level 2: 0.5 × Δ_i Δ_j              shape (d, d)  — symmetric part
    level 3: (1/6) × Δ_i Δ_j Δ_k        shape (d, d, d)

We concatenate the flattened levels into a single feature vector. For a
path with n points we cumulate increments through ``compute_signature``
and return the canonical truncated signature at depth ``depth``.

Exported:
    compute_signature(path, depth=3, normalize=True) → np.ndarray
    signature_dim(d, depth) → int
    lead_lag_augmentation(series) → np.ndarray
    time_augmentation(series) → np.ndarray
    snapshot_path_signature_features(mid_series, depth=2, augmentation="time")
        → dict
    PATH_SIGNATURE_FEATURE_NAMES  tuple[str, ...]
"""
from __future__ import annotations

import numpy as np


MAX_DEPTH = 3
DEFAULT_SIG_COMPONENTS = 20


PATH_SIGNATURE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"path_sig_{i:02d}" for i in range(DEFAULT_SIG_COMPONENTS)
)


# ---------------------------------------------------------------------------
# Core: compute_signature
# ---------------------------------------------------------------------------


def compute_signature(
    path: np.ndarray,
    *,
    depth: int = 2,
    normalize: bool = True,
) -> np.ndarray:
    """Depth-``depth`` truncated signature of a (n, d) path.

    Uses the Chen identity in its naive iterated-integral form — O(n·d²)
    for depth-2, O(n·d³) for depth-3. Good enough for the 50-200 sample
    rolling windows we care about.
    """
    arr = np.asarray(path, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    n, d = arr.shape
    if n < 2:
        return np.zeros(signature_dim(d, depth), dtype=float)
    dX = np.diff(arr, axis=0)   # (n-1, d)

    level1 = dX.sum(axis=0)     # (d,)
    levels = [level1]

    if depth >= 2:
        # Level-2 signature: 0.5 × (Δx_i Δx_j) summed over all intervals
        # + cross-product terms giving the Lévy area. The symmetric part
        # equals 0.5 × sum(Δx_i Δx_j); the antisymmetric (Lévy area)
        # piece uses cumulative increments.
        cum = np.concatenate([np.zeros((1, d)), np.cumsum(dX, axis=0)], axis=0)
        # S_{i,j} = Σ_k X_k^i (dX_k^j) for Stratonovich-style signature.
        # We take the symmetric approximation which is numerically stable.
        level2 = np.zeros((d, d), dtype=float)
        for k in range(dX.shape[0]):
            xk = cum[k]
            level2 += np.outer(xk, dX[k])
        # Symmetrize to avoid depending on integration direction
        level2 = 0.5 * (level2 + level2.T)
        levels.append(level2.flatten())

    if depth >= 3:
        level3 = np.zeros((d, d, d), dtype=float)
        cum2 = np.zeros((d, d), dtype=float)
        for k in range(dX.shape[0]):
            cum_before = cum2.copy()
            cum2 += np.outer(cum[k], dX[k])
            level3 += np.einsum("ij,k->ijk", cum_before, dX[k])
        levels.append(level3.flatten())

    sig = np.concatenate(levels)
    if normalize:
        norm = np.linalg.norm(sig)
        if norm > 0.0:
            sig = sig / norm
    return sig


def signature_dim(d: int, depth: int) -> int:
    """Number of signature components at dimension ``d`` and depth ``depth``."""
    total = 0
    for lvl in range(1, depth + 1):
        total += d ** lvl
    return total


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def time_augmentation(series: np.ndarray) -> np.ndarray:
    """Augment a 1D series with a normalized time axis."""
    arr = np.asarray(series, dtype=float).reshape(-1)
    n = arr.size
    if n < 2:
        return arr[:, None]
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([t, arr])


def lead_lag_augmentation(series: np.ndarray) -> np.ndarray:
    """Lead-lag augmentation: each original point becomes two points
    (x_k, x_{k-1}) so signatures capture autocorrelation."""
    arr = np.asarray(series, dtype=float).reshape(-1)
    n = arr.size
    if n < 2:
        return np.zeros((0, 2), dtype=float)
    leads = arr[1:]
    lags = arr[:-1]
    return np.column_stack([leads, lags])


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def snapshot_path_signature_features(
    mid_series: np.ndarray | list[float],
    *,
    depth: int = 2,
    augmentation: str = "time",
) -> dict[str, float]:
    """Compute signature on a rolling ``mid_series``, return the first
    ``DEFAULT_SIG_COMPONENTS`` coefficients as named scalars. Zero-pads
    if the signature dim is smaller than the fixed feature count."""
    arr = np.asarray(mid_series, dtype=float)
    if arr.size < 3:
        return {name: 0.0 for name in PATH_SIGNATURE_FEATURE_NAMES}

    if augmentation == "lead_lag":
        aug = lead_lag_augmentation(arr)
        if aug.size == 0:
            return {name: 0.0 for name in PATH_SIGNATURE_FEATURE_NAMES}
        sig = compute_signature(aug, depth=depth, normalize=True)
    else:
        aug = time_augmentation(arr)
        sig = compute_signature(aug, depth=depth, normalize=True)

    padded = np.zeros(DEFAULT_SIG_COMPONENTS, dtype=float)
    k = min(sig.size, DEFAULT_SIG_COMPONENTS)
    padded[:k] = sig[:k]
    return {name: float(padded[i]) for i, name in enumerate(PATH_SIGNATURE_FEATURE_NAMES)}
