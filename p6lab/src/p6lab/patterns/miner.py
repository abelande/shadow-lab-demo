"""
Unsupervised Pattern Miner — HDBSCAN Discovery
Spec §5.2 | Used by Notebook 04

Pipeline:
1. Burst-anchor windowing via event_windowing.WindowIterator
2. 30-dim event shape vector per window
3. Instrument normalization (§3.3)  [Phase 5 — pass-through for now]
4. HDBSCAN with min_cluster_size=50, min_samples=25
5. Forward-outcome labeling at 1m/5m/15m/1h
6. Filter: n≥200, hit_rate_5m > 0.55, Sharpe > 0.3,
   cosine distance to existing patterns > 0.3
7. Write candidates to mined_candidates/*.parquet

Promotion decision happens in the web UI (§10.3), NOT here.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..ingestion.event_windowing import (
    BurstDetectorConfig,
    WindowAnchorStrategy,
    WindowIterator,
)
from .labeler import (
    HORIZONS,
    MultiHorizonOutcome,
    OutcomeClass,
    _extract_price_series,
    compute_outcome_statistics,
    label_pattern_instance,
)
from .library import PatternLibrary

logger = logging.getLogger(__name__)


HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES = 25

MIN_OCCURRENCES = 200
MIN_HIT_RATE_5M = 0.55
MIN_SHARPE = 0.3
MIN_COSINE_DISTANCE_TO_EXISTING = 0.3

SHAPE_VECTOR_DIM = 30


@dataclass(frozen=True)
class MinedCandidate:
    cluster_id: int
    centroid: np.ndarray
    member_count: int
    exemplar_timestamps_ms: list[int]
    outcome_stats: dict[str, Any]
    hit_rate_5m: float
    sharpe: float
    cosine_distance_to_nearest_existing: float
    random_state: int


def _event_get(ev, key: str, default=None):
    """Get key from dict event or attribute from object event."""
    if isinstance(ev, dict):
        return ev.get(key, default)
    return getattr(ev, key, default)


def _action_of(ev) -> str:
    return str(_event_get(ev, "action", "") or "").lower()


def _side_of(ev) -> str:
    return str(_event_get(ev, "side", "") or "").lower()


def _price_of(ev) -> float:
    try:
        return float(_event_get(ev, "price", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _size_of(ev) -> float:
    v = _event_get(ev, "size", None)
    if v is None:
        v = _event_get(ev, "volume", 0.0)
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ts_of(ev) -> int:
    return int(_event_get(ev, "timestamp_ms", 0) or 0)


def _linreg_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def _skew(xs: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    m = sum(xs) / n
    s2 = sum((x - m) ** 2 for x in xs) / n
    if s2 <= 0:
        return 0.0
    s = math.sqrt(s2)
    return sum(((x - m) / s) ** 3 for x in xs) / n


def extract_event_shape_vector(window_events: list[Any]) -> np.ndarray:
    """Compute the 30-dim event shape vector for one window."""
    vec = np.zeros(SHAPE_VECTOR_DIM, dtype=float)
    if not window_events:
        return vec

    n = len(window_events)
    actions = [_action_of(e) for e in window_events]
    sides = [_side_of(e) for e in window_events]
    sizes = [_size_of(e) for e in window_events]
    prices = [_price_of(e) for e in window_events]
    timestamps = [_ts_of(e) for e in window_events]

    adds = sum(1 for a in actions if "add" in a)
    cancels = sum(1 for a in actions if "cancel" in a or "remove" in a)
    # [0] add_cancel_ratio
    denom = adds + cancels
    vec[0] = (adds / denom) if denom > 0 else 0.5

    # [1] burst_intensity (events/sec over the window span)
    span_ms = max(1, timestamps[-1] - timestamps[0])
    vec[1] = n * 1000.0 / span_ms

    # [2-4] spread dynamics — need per-event best-bid/ask. We don't carry
    # book state here, so approximate "spread" at each event by
    # |price - rolling_mid|. Rolling mid = mean of last 20 prices.
    spreads: list[float] = []
    recent: list[float] = []
    for p in prices:
        recent.append(p)
        if len(recent) > 20:
            recent.pop(0)
        mid = sum(recent) / len(recent)
        spreads.append(abs(p - mid))
    vec[2] = float(np.mean(spreads)) if spreads else 0.0
    vec[3] = float(np.std(spreads)) if spreads else 0.0
    xs = [float(t - timestamps[0]) / 1000.0 for t in timestamps]
    vec[4] = _linreg_slope(xs, spreads)

    # [5] level_crossings — # distinct prices touched
    distinct_prices = {round(p, 6) for p in prices if p > 0}
    vec[5] = float(len(distinct_prices))

    # [6-9] size percentiles
    if sizes:
        sizes_arr = np.asarray(sizes, dtype=float)
        vec[6] = float(np.percentile(sizes_arr, 25))
        vec[7] = float(np.percentile(sizes_arr, 50))
        vec[8] = float(np.percentile(sizes_arr, 75))
        vec[9] = float(np.percentile(sizes_arr, 95))

    # [10-12] inter-event intervals (ms)
    if n >= 2:
        deltas = [float(timestamps[i] - timestamps[i - 1]) for i in range(1, n)]
        vec[10] = float(np.mean(deltas))
        vec[11] = float(np.std(deltas))
        vec[12] = _skew(deltas)

    # [13] side asymmetry
    bid_ct = sum(1 for s in sides if s in ("bid", "buy", "b"))
    ask_ct = sum(1 for s in sides if s in ("ask", "sell", "s", "a"))
    total_sides = bid_ct + ask_ct
    vec[13] = (bid_ct - ask_ct) / total_sides if total_sides > 0 else 0.0

    # [14] lifecycle entropy — Shannon entropy over action distribution
    counts = Counter(actions)
    probs = [c / n for c in counts.values() if c > 0]
    vec[14] = -sum(p * math.log(p) for p in probs) if probs else 0.0

    # [15-29] reserved — leave as zeros for future extension
    return vec


def run_hdbscan(
    shape_vectors: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    random_state: int = 42,
) -> tuple[np.ndarray, Any]:
    """Run HDBSCAN clustering. Returns (labels, clusterer). Noise label = -1."""
    import hdbscan
    if len(shape_vectors) == 0:
        return np.array([], dtype=int), None
    if len(shape_vectors) < max(min_cluster_size, min_samples + 1):
        # HDBSCAN can't build a KD-tree with fewer points than neighbors.
        # Return all-noise labels; no meaningful clusters with this little data.
        return np.full(len(shape_vectors), -1, dtype=int), None
    # HDBSCAN is deterministic given the same input; np seed ensures tie-break reproducibility.
    np.random.seed(random_state)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(shape_vectors)
    return labels, clusterer


def compute_umap_projection(
    shape_vectors: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """2D UMAP projection for visualization only (NB04 §04)."""
    import umap
    if len(shape_vectors) < n_components + 2:
        return np.zeros((len(shape_vectors), n_components), dtype=float)
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        n_jobs=1,
    )
    return reducer.fit_transform(shape_vectors)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    return 1.0 - sim


def filter_candidates(
    candidates: list[MinedCandidate],
    existing_centroids: np.ndarray | None = None,
    min_occurrences_override: int | None = None,
) -> list[MinedCandidate]:
    """Apply the 4-stage filter pipeline with per-stage rejection logging.

    ``min_occurrences_override`` lets callers (typically smoketest-scale
    notebooks) relax the production-scale ``MIN_OCCURRENCES = 200`` floor.
    """
    reject_counts = {
        "n": 0, "hit_rate_5m": 0, "sharpe": 0, "cosine_distance": 0,
    }
    min_occ = MIN_OCCURRENCES if min_occurrences_override is None else int(min_occurrences_override)
    kept: list[MinedCandidate] = []
    for cand in candidates:
        if cand.member_count < min_occ:
            reject_counts["n"] += 1
            continue
        if cand.hit_rate_5m <= MIN_HIT_RATE_5M:
            reject_counts["hit_rate_5m"] += 1
            continue
        if cand.sharpe <= MIN_SHARPE:
            reject_counts["sharpe"] += 1
            continue
        if existing_centroids is not None and len(existing_centroids) > 0:
            min_dist = min(
                _cosine_distance(cand.centroid, c) for c in existing_centroids
            )
            if min_dist <= MIN_COSINE_DISTANCE_TO_EXISTING:
                reject_counts["cosine_distance"] += 1
                continue
        kept.append(cand)
    logger.info("filter_candidates: kept=%d rejected=%s", len(kept), reject_counts)
    return kept


def _sharpe_from_returns(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns))
    if std == 0:
        return 0.0
    return mean / std


def _build_candidate(
    cluster_id: int,
    centroid: np.ndarray,
    member_vectors: np.ndarray,
    member_timestamps: list[int],
    forward_event_stream: list,
    instrument_atr: float,
    tick_size: float,
    session_end_ms: int | None,
    existing_centroids: np.ndarray | None,
    random_state: int,
) -> MinedCandidate:
    # 5 exemplars = members closest to centroid
    if len(member_vectors) > 0:
        dists = np.linalg.norm(member_vectors - centroid, axis=1)
        order = np.argsort(dists)[:5]
        exemplars = [member_timestamps[i] for i in order]
    else:
        exemplars = []

    # Forward outcomes per member — reuse price series for O(members × log E)
    outcomes: list[MultiHorizonOutcome] = []
    for ts in member_timestamps:
        mh = label_pattern_instance(
            event_stream=forward_event_stream,
            pattern_timestamp_ms=ts,
            pattern_direction="long",
            instrument_atr=instrument_atr,
            tick_size=tick_size,
            session_end_ms=session_end_ms,
            pattern_id=f"cluster_{cluster_id}",
            _price_series=price_series,
        )
        outcomes.append(mh)

    outcome_stats = {h: compute_outcome_statistics(outcomes, h) for h in HORIZONS}
    hit_rate_5m = outcome_stats["5m"]["hit_rate"]
    if math.isnan(hit_rate_5m):
        hit_rate_5m = 0.0

    # Sharpe on 5m atr-normalized returns (non-incomplete only)
    five_min_returns = [
        mh.outcomes["5m"].atr_normalized_return
        for mh in outcomes
        if mh.outcomes.get("5m")
        and mh.outcomes["5m"].classification != OutcomeClass.INCOMPLETE
        and not math.isnan(mh.outcomes["5m"].atr_normalized_return)
    ]
    sharpe = _sharpe_from_returns(five_min_returns)

    # Cosine distance to nearest existing pattern
    if existing_centroids is not None and len(existing_centroids) > 0:
        cos_dist = min(_cosine_distance(centroid, c) for c in existing_centroids)
    else:
        cos_dist = 1.0

    return MinedCandidate(
        cluster_id=cluster_id,
        centroid=centroid,
        member_count=len(member_timestamps),
        exemplar_timestamps_ms=exemplars,
        outcome_stats=outcome_stats,
        hit_rate_5m=hit_rate_5m,
        sharpe=sharpe,
        cosine_distance_to_nearest_existing=cos_dist,
        random_state=random_state,
    )


def _load_events_from_triple_view(path: Path) -> list[dict]:
    """Flatten all l3_events rows from a triple_view parquet."""
    df = pd.read_parquet(path)
    events: list[dict] = []
    for lst in df["l3_events"]:
        if lst is None:
            continue
        for ev in lst:
            events.append(dict(ev))
    events.sort(key=lambda e: int(e.get("timestamp_ms", 0)))
    return events


def mine(
    triple_view_path: Path,
    library_path: Path,
    output_dir: Path,
    symbols: list[str],
    random_state: int = 42,
    *,
    burst_config: BurstDetectorConfig | None = None,
    instrument_atr: float = 2.0,
    tick_size: float = 0.25,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    apply_filters: bool = True,
) -> list[MinedCandidate]:
    """Full mining pipeline.

    Note: per spec §3.3, instrument normalization is a Phase-5 concern;
    the current implementation mines a single instrument at a time and
    treats the `symbols` list as a filter on parquet files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "mined_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    # Load existing pattern centroids for novelty filtering (best-effort)
    existing_centroids: np.ndarray | None = None
    try:
        lib = PatternLibrary(library_path)
        lib.load()
        # Library does not currently store centroids directly; leave as None.
        existing_centroids = None
    except Exception as e:
        logger.warning("Failed to load existing library %s: %s", library_path, e)

    all_candidates: list[MinedCandidate] = []
    for symbol in symbols:
        # Resolve parquet path: either directory containing {symbol}_1s.parquet
        # or a direct file path.
        path = Path(triple_view_path)
        if path.is_dir():
            target = path / f"{symbol}_1s.parquet"
        else:
            target = path
        if not target.exists():
            logger.warning("Triple-view parquet missing for %s: %s", symbol, target)
            continue

        events = _load_events_from_triple_view(target)
        if not events:
            logger.warning("No events for %s in %s", symbol, target)
            continue
        logger.info("Loaded %d events for %s", len(events), symbol)

        # Window with burst anchoring
        iterator = WindowIterator.create(
            strategy=WindowAnchorStrategy.BURST_ANCHORED,
            events=events,
            burst_config=burst_config,
        )
        windows = list(iterator)
        if not windows:
            logger.warning("No burst windows for %s", symbol)
            continue
        logger.info("Extracted %d burst windows for %s", len(windows), symbol)

        vectors = np.stack([extract_event_shape_vector(w.events) for w in windows])
        anchor_ts = [w.anchor_ms for w in windows]

        labels, _ = run_hdbscan(
            vectors, min_cluster_size=min_cluster_size,
            min_samples=min_samples, random_state=random_state,
        )
        n_clusters = len(set(l for l in labels if l >= 0))
        logger.info("HDBSCAN: %d clusters (+ noise=%d) for %s",
                    n_clusters, int(np.sum(labels == -1)), symbol)

        session_end_ms = events[-1]["timestamp_ms"]

        # Pre-extract price series ONCE — reused across every cluster member.
        # Without this, _extract_price_series is O(E) per call × members × clusters.
        price_series = _extract_price_series(events)
        logger.info("Price series: %d points", len(price_series[0]))

        symbol_candidates: list[MinedCandidate] = []
        for cid in sorted(set(int(l) for l in labels if l >= 0)):
            mask = labels == cid
            members = vectors[mask]
            centroid = members.mean(axis=0)
            member_ts = [anchor_ts[i] for i, m in enumerate(mask) if m]
            cand = _build_candidate(
                cluster_id=cid,
                centroid=centroid,
                member_vectors=members,
                member_timestamps=member_ts,
                forward_event_stream=events,
                instrument_atr=instrument_atr,
                tick_size=tick_size,
                session_end_ms=session_end_ms,
                existing_centroids=existing_centroids,
                random_state=random_state,
                price_series=price_series,
            )
            symbol_candidates.append(cand)

        if apply_filters:
            kept = filter_candidates(symbol_candidates, existing_centroids)
        else:
            kept = symbol_candidates

        _write_candidates_parquet(candidates_dir / f"{symbol}_candidates.parquet", kept)
        all_candidates.extend(kept)

    logger.info("mine: wrote %d total candidates across %d symbols",
                len(all_candidates), len(symbols))
    return all_candidates


def _write_candidates_parquet(path: Path, candidates: list[MinedCandidate]) -> None:
    if not candidates:
        logger.info("No candidates to write to %s", path)
        return
    rows = [
        {
            "cluster_id": c.cluster_id,
            "centroid": c.centroid.tolist(),
            "member_count": c.member_count,
            "exemplar_timestamps_ms": c.exemplar_timestamps_ms,
            "outcome_stats": c.outcome_stats,
            "hit_rate_5m": c.hit_rate_5m,
            "sharpe": c.sharpe,
            "cosine_distance_to_nearest_existing": c.cosine_distance_to_nearest_existing,
            "random_state": c.random_state,
        }
        for c in candidates
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    logger.info("Wrote %d candidates to %s", len(rows), path)
