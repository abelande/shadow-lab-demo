"""Label construction (Lopez de Prado, "Advances in Financial ML" §3.4).

The naive forward-return-sign label (``mid[t+H] > mid[t]``) ignores whether
the move ever reached a meaningful size, treats a 0.1-tick drift as a
1-tick move, and produces hideously imbalanced classes in low-volatility
windows. This module implements:

- ``triple_barrier_labels``: walk forward up to ``horizon_ms`` and return
  which barrier (up/down/timeout) hit first. Side ∈ {+1, -1, 0}.

- ``cost_thresholded_binary``: binary label that masks out sub-cost moves
  as NaN (drop them from the training set).

- ``cusum_events`` / ``activity_mask``: Wave 9-A.1 — flag snapshots where
  *something is happening* so the model trains on informative rows only
  (López de Prado, AFML §17.2). The mask is a soft selection: we keep all
  rows in ``X`` for diagnostic + threshold tuning but train only on
  ``X.loc[mask]``. At inference the engine emits a base-rate prior on
  ``~mask`` rows.

- ``mfe_mae_labels``: Wave 9 §H.1.c — path-aware 5-class label
  ``{-2, -1, 0, +1, +2}`` encoding both direction (sign) and cleanness
  (magnitude). Built on top of MFE/MAE walk-forward per López de Prado
  AFML §3.6. Distinguishes "clean directional move" (no significant
  adverse excursion along the way) from "wicky resolution" (barrier hit
  but a stop at ±X would have triggered first). The ``{±2}`` subset is
  the trade-actionable population.

- ``LabelSpec`` / ``compute_label_set``: Wave 9 §H.1.b — multi-target
  dispatcher. Define multiple ``LabelSpec`` entries (e.g. TB at 60s/120s
  + MFE/MAE at 60s/120s/180s/300s) and compute all label vectors in one
  call. Used by NB06 §04 to train one model per (label_spec, horizon)
  pair side-by-side, then compare §04d/§04e diagnostics across them.

- ``pattern_firing_labels``: Wave 10-A — per-row label answering "in the
  next K snapshots, did any library pattern fire?" (binary) or "which
  pattern fired" (multi-class). Built on top of pre-computed firing
  events so the labeler is decoupled from the matcher's compute path —
  caller runs the matcher in batch (NB06 §02 pattern), feeds firings in.
  Produces labels usable as both: (a) Strategy B's primary target —
  trade when the matcher recognizes a known opportunity, and (b)
  Strategy C's matcher leg — fused with the calibrated directional
  proba in Wave 10-C.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TripleBarrierLabel:
    """One snapshot's triple-barrier outcome.

    ``barrier_hit`` distinguishes four outcomes:

    - ``"up"`` / ``"down"``: profit barrier resolved within the observed
      window. ``side`` ∈ {+1, -1}.
    - ``"timeout"``: full horizon observed AND no barrier fired. ``side`` = 0.
      A real outcome — informative for training.
    - ``"unknown"``: data ended before the deadline so we never saw whether
      a barrier would have fired. ``side`` = 0 by convention but the row
      should be excluded from training. Detect via ``barrier_hit ==
      "unknown"`` (or convert to NaN via ``compute_label_set``).
    """
    side: int          # +1 up-target hit / 0 timeout|unknown / -1 down-target hit
    ret: float         # realized return when barrier hit (signed, in price units)
    horizon_ms: int    # wall-clock from entry to resolution
    barrier_hit: str   # "up" | "down" | "timeout" | "unknown"


def triple_barrier_labels(
    mid: np.ndarray,
    ts_ms: np.ndarray,
    *,
    horizon_ms: int = 60_000,
    up_target_ticks: float = 4.0,
    down_target_ticks: float = 4.0,
    tick_size: float = 0.25,
) -> list[TripleBarrierLabel]:
    """For each entry, find the first of {up-barrier, down-barrier, timeout} hit.

    Parameters
    ----------
    mid : np.ndarray
        1D array of mid prices (one per snapshot).
    ts_ms : np.ndarray
        1D array of timestamps in milliseconds. Must be same length as ``mid``.
    horizon_ms : int
        Vertical barrier (timeout). Default 60 seconds.
    up_target_ticks : float
        Up barrier = entry + ``up_target_ticks * tick_size``.
    down_target_ticks : float
        Down barrier = entry - ``down_target_ticks * tick_size``.
    tick_size : float
        Instrument tick size (e.g. 0.25 for NQ).

    Returns
    -------
    list[TripleBarrierLabel]
        One entry per input row. Outcome semantics — see ``TripleBarrierLabel``:
        ``"up"`` / ``"down"`` are barrier hits, ``"timeout"`` is a full-horizon
        observed non-hit (still a real outcome), ``"unknown"`` flags rows
        whose deadline extends past the data — exclude these from training.
    """
    if len(mid) != len(ts_ms):
        raise ValueError(f"mid and ts_ms length mismatch: {len(mid)} vs {len(ts_ms)}")

    n = len(mid)
    up_move = up_target_ticks * tick_size
    down_move = down_target_ticks * tick_size
    out: list[TripleBarrierLabel] = []
    last_ts = int(ts_ms[-1]) if n > 0 else 0

    for i in range(n):
        entry = mid[i]
        entry_ts = ts_ms[i]
        deadline = entry_ts + horizon_ms
        up_bar = entry + up_move
        down_bar = entry - down_move

        side = 0
        ret = 0.0
        hit = "timeout"
        hit_ts = deadline

        # Walk forward until we hit a barrier
        for j in range(i + 1, n):
            if ts_ms[j] > deadline:
                hit_ts = deadline
                ret = float(mid[j - 1] - entry) if j > 0 else 0.0
                break
            if mid[j] >= up_bar:
                side, hit, hit_ts = 1, "up", int(ts_ms[j])
                ret = float(mid[j] - entry)
                break
            if mid[j] <= down_bar:
                side, hit, hit_ts = -1, "down", int(ts_ms[j])
                ret = float(mid[j] - entry)
                break
        else:
            # Loop completed without break → either a timeout (we observed
            # through the deadline but no barrier fired) or an unknown
            # outcome (data ended before the deadline). Distinguish by
            # comparing the dataset's last timestamp against the deadline.
            if last_ts >= deadline:
                ret = float(mid[-1] - entry)
            else:
                hit = "unknown"
                hit_ts = entry_ts  # horizon_ms = 0 carries the unobserved tag
                ret = 0.0

        out.append(TripleBarrierLabel(
            side=side, ret=ret,
            horizon_ms=int(hit_ts - entry_ts),
            barrier_hit=hit,
        ))

    return out


def cost_thresholded_binary(
    mid: np.ndarray,
    horizon_snapshots: int,
    *,
    cost_ticks: float,
    tick_size: float = 0.25,
) -> np.ndarray:
    """Binary label with NaN for sub-cost moves.

    For each row ``i``, ``fwd = mid[i + horizon] - mid[i]``. If
    ``abs(fwd) < cost_ticks * tick_size``, the label is NaN (drop from
    training). Otherwise it is 1 if ``fwd > 0`` else 0.

    Parameters
    ----------
    mid : np.ndarray
        1D array of mid prices.
    horizon_snapshots : int
        Forward horizon in rows.
    cost_ticks : float
        Round-trip transaction cost in ticks. Moves smaller than this are
        unprofitable regardless of direction, so their "direction" is noise.
    tick_size : float
        Instrument tick size.

    Returns
    -------
    np.ndarray of shape ``(len(mid),)``, dtype float with values in {0, 1, NaN}.
    """
    n = len(mid)
    out = np.full(n, np.nan, dtype=float)
    if n <= horizon_snapshots:
        return out
    cost = cost_ticks * tick_size
    fwd = mid[horizon_snapshots:] - mid[:-horizon_snapshots]
    # Only label rows where the move exceeded transaction cost
    mask = np.abs(fwd) >= cost
    out[:-horizon_snapshots] = np.where(mask, (fwd > 0).astype(float), np.nan)
    return out


# ---------------------------------------------------------------------------
# Activity detection — Wave 9 §H.1.a
# ---------------------------------------------------------------------------
#
# CUSUM filter from López de Prado (AFML §17.2). Tracks two-sided cumulative
# deviations of price changes from zero; fires an event whenever either
# accumulator crosses ±threshold. Resets both accumulators after each event.
#
# The activity mask selects snapshots in a window around each event. Used to
# concentrate training on rows where the order book is doing something —
# rather than the ~95% of equilibrium snapshots that drown signal in noise.
# See `reports/P6LAB-WAVE-9-10-BUILD-PHASES.md` §H.1.a for sequencing context.


def cusum_events(
    price: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Two-sided CUSUM event indices (López de Prado, AFML §17.2).

    Walks forward through first differences of ``price``, accumulating
    signed changes. When the upper accumulator exceeds ``+threshold`` or
    the lower accumulator falls below ``-threshold``, an event is recorded
    at that index and *both* accumulators reset to zero. Pure NumPy/Python
    loop — fine for the snapshot counts seen in NB06 (millions per day).

    Parameters
    ----------
    price : np.ndarray
        1D array of price values (mid, weighted-mid, or any drift-bearing
        series). Length ``n``.
    threshold : float
        Event firing threshold in the same units as ``price`` (e.g. 0.5 for
        NQ would mean two-tick accumulated drift). Higher = fewer, larger
        events. Tune per instrument.

    Returns
    -------
    np.ndarray of int
        Sorted event indices into ``price``. Empty array if no events fire
        (e.g. flat series or threshold too high for the data scale).
    """
    n = len(price)
    if n < 2 or threshold <= 0:
        return np.array([], dtype=np.int64)
    diffs = np.diff(price)
    s_pos = 0.0
    s_neg = 0.0
    events: list[int] = []
    for i, r in enumerate(diffs, start=1):
        s_pos = max(0.0, s_pos + float(r))
        s_neg = min(0.0, s_neg + float(r))
        if s_pos >= threshold:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg <= -threshold:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0
    return np.asarray(events, dtype=np.int64)


def activity_mask(
    price: np.ndarray,
    ts_ms: np.ndarray,
    *,
    method: str = "cusum",
    cusum_threshold: float = 0.5,
    lookback_ms: int = 60_000,
    lookforward_ms: int = 0,
    volume: np.ndarray | None = None,
    volume_floor: float = 100.0,
) -> np.ndarray:
    """Boolean per-row mask flagging snapshots that pass the activity filter.

    Two MVP methods, both returning a per-row ``bool`` array of length
    ``len(price)``:

    - ``"cusum"`` (default): runs :func:`cusum_events` on ``price`` to
      detect drift events, then masks rows whose timestamp falls within
      ``[event_ts - lookback_ms, event_ts + lookforward_ms]`` of any
      event. Default ``lookforward_ms=0`` — pure lookback — concentrates
      training on the lead-up to detected activity.
    - ``"volume_bar"``: masks rows whose ``volume[i] >= volume_floor``.
      Requires the optional ``volume`` array — usually a rolling-window
      contract count from the L2 features.

    Compose multiple methods externally:

    .. code-block:: python

        m_cusum  = activity_mask(price, ts_ms, method="cusum")
        m_volume = activity_mask(price, ts_ms, method="volume_bar",
                                  volume=v, volume_floor=200)
        combined = m_cusum & m_volume   # AND, both must agree

    Parameters
    ----------
    price : np.ndarray
        Drift-bearing price series (typically mid). Length ``n``.
    ts_ms : np.ndarray
        Per-row timestamps in milliseconds. Length ``n``.
    method : {"cusum", "volume_bar"}
        Detector flavor.
    cusum_threshold : float
        Forwarded to :func:`cusum_events` when ``method="cusum"``.
    lookback_ms, lookforward_ms : int
        Window around each CUSUM event to mark as "active." Pure
        lookback (default) avoids leaking post-event price information
        into training-row features.
    volume : np.ndarray, optional
        Per-row recent volume (e.g. last-K-second contract count).
        Required when ``method="volume_bar"``.
    volume_floor : float
        Minimum per-row volume to qualify as active.

    Returns
    -------
    np.ndarray of bool
        ``True`` where the row is in an "active" window.
    """
    n = len(price)
    if len(ts_ms) != n:
        raise ValueError(
            f"price and ts_ms length mismatch: {n} vs {len(ts_ms)}",
        )

    mask = np.zeros(n, dtype=bool)

    if method == "cusum":
        events = cusum_events(price, cusum_threshold)
        if events.size == 0:
            return mask
        ts_ms_arr = np.asarray(ts_ms, dtype=np.int64)
        for ev_idx in events:
            ev_ts = int(ts_ms_arr[ev_idx])
            window_start = ev_ts - int(lookback_ms)
            window_end = ev_ts + int(lookforward_ms)
            mask |= (ts_ms_arr >= window_start) & (ts_ms_arr <= window_end)
        return mask

    if method == "volume_bar":
        if volume is None:
            raise ValueError(
                "method='volume_bar' requires the `volume` argument",
            )
        volume_arr = np.asarray(volume)
        if len(volume_arr) != n:
            raise ValueError(
                f"volume length {len(volume_arr)} != price length {n}",
            )
        return volume_arr >= volume_floor

    raise ValueError(
        f"unknown method: {method!r}; must be 'cusum' or 'volume_bar'",
    )


# ---------------------------------------------------------------------------
# Path-aware 5-class label — Wave 9 §H.1.c
# ---------------------------------------------------------------------------
#
# Standard triple-barrier merges "clean directional ride" (good trade) and
# "wicky resolution after stop-out" (bad trade) into the same ±1 label
# because TB is path-blind. The 5-class scheme tracks max-favorable and
# max-adverse excursion (MFE / MAE) per row and bakes a stop-aware
# distinction into the label:
#
#   +2  clean bull   : up barrier hit AND |MAE| < stop
#   +1  wicky bull   : up barrier hit AND |MAE| >= stop
#    0  timeout      : neither barrier hit before time barrier
#   -1  wicky bear   : down barrier hit AND |MFE| >= stop
#   -2  clean bear   : down barrier hit AND |MFE| < stop
#
# The {±2} subset is the trade-actionable population — a model trained
# to predict {±2} directly answers "would my trade have completed
# without hitting a stop?" rather than "did price eventually move?"


@dataclass(frozen=True)
class MFEMAELabel:
    """One snapshot's path-aware 5-class outcome.

    Encodes both direction (sign of ``label``) and cleanness (magnitude:
    ``±2`` clean, ``±1`` wicky). ``mfe`` and ``mae`` are kept as raw
    excursion values in price units so downstream code can derive
    secondary labels (binary clean/wicky, MFE/MAE ratio regression, etc.)
    without re-running the labeler.

    ``barrier_hit`` distinguishes:

    - ``"up"`` / ``"down"``: profit barrier resolved within the observed
      window. ``label`` ∈ {±1, ±2}.
    - ``"timeout"``: full horizon observed AND no barrier fired. ``label``
      = 0. A real outcome — informative for training.
    - ``"unknown"``: data ended before the deadline. ``label`` = 0 by
      convention; exclude from training (or use ``compute_label_set``
      which converts to NaN).
    """
    label: int            # ∈ {-2, -1, 0, +1, +2}
    mfe: float            # max favorable excursion (price units, ≥ 0)
    mae: float            # max adverse excursion (price units, ≤ 0)
    barrier_hit: str      # "up" | "down" | "timeout" | "unknown"
    horizon_ms: int       # actual time-to-resolution (= horizon_ms on timeout)


def mfe_mae_labels(
    mid: np.ndarray,
    ts_ms: np.ndarray,
    *,
    horizon_ms: int = 60_000,
    up_target_ticks: float = 4.0,
    down_target_ticks: float = 4.0,
    stop_threshold_ticks: float = 1.5,
    tick_size: float = 0.25,
) -> list[MFEMAELabel]:
    """Path-aware 5-class triple barrier with stop-survivability gating.

    Walks forward from each row up to ``horizon_ms`` tracking the
    running MFE (max favorable) and MAE (max adverse) excursions. When
    either profit barrier hits, classifies the result by whether the
    *opposite* excursion ever exceeded ``stop_threshold_ticks`` —
    distinguishing clean directional moves (``±2``) from wicky
    resolutions (``±1``).

    Parameters
    ----------
    mid : np.ndarray
        1D array of mid prices, one per snapshot.
    ts_ms : np.ndarray
        1D array of timestamps in milliseconds. Must be same length as ``mid``.
    horizon_ms : int
        Time barrier; default 60 seconds.
    up_target_ticks, down_target_ticks : float
        Profit-target distances from entry, in ticks. Symmetric by default.
    stop_threshold_ticks : float
        The trader's stop. If the opposite excursion reaches this magnitude
        before the profit barrier, the resolution is labeled wicky (±1)
        instead of clean (±2). Default 1.5 ticks ≈ 0.15 ATR for NQ at
        typical volatility.
    tick_size : float
        Instrument tick size (NQ = 0.25).

    Returns
    -------
    list[MFEMAELabel]
        One entry per input row.
    """
    if len(mid) != len(ts_ms):
        raise ValueError(
            f"mid and ts_ms length mismatch: {len(mid)} vs {len(ts_ms)}",
        )

    n = len(mid)
    up_target = up_target_ticks * tick_size
    down_target = down_target_ticks * tick_size
    stop = stop_threshold_ticks * tick_size
    out: list[MFEMAELabel] = []
    last_ts = int(ts_ms[-1]) if n > 0 else 0

    for i in range(n):
        entry = float(mid[i])
        entry_ts = int(ts_ms[i])
        deadline = entry_ts + horizon_ms

        mfe = 0.0
        mae = 0.0
        label = 0
        barrier_hit = "timeout"
        hit_ts = deadline

        # Walk forward until a barrier hits or we time out.
        for j in range(i + 1, n):
            if ts_ms[j] > deadline:
                hit_ts = deadline
                break
            diff = float(mid[j]) - entry
            if diff > mfe:
                mfe = diff
            if diff < mae:
                mae = diff
            if diff >= up_target:
                # Upper barrier hit. Clean if MAE never reached -stop.
                label = 2 if abs(mae) < stop else 1
                barrier_hit = "up"
                hit_ts = int(ts_ms[j])
                break
            if diff <= -down_target:
                # Lower barrier hit. Clean if MFE never reached +stop.
                label = -2 if abs(mfe) < stop else -1
                barrier_hit = "down"
                hit_ts = int(ts_ms[j])
                break
        else:
            # End-of-data without hitting any barrier. Real timeout iff
            # we observed at least through the deadline; otherwise the
            # outcome is unknown (data ran out first).
            if last_ts < deadline:
                barrier_hit = "unknown"
                hit_ts = entry_ts

        out.append(MFEMAELabel(
            label=label,
            mfe=float(mfe),
            mae=float(mae),
            barrier_hit=barrier_hit,
            horizon_ms=int(hit_ts - entry_ts),
        ))

    return out


# ---------------------------------------------------------------------------
# Multi-target dispatcher — Wave 9 §H.1.b
# ---------------------------------------------------------------------------
#
# NB06 §04 trains one model per (label_kind, horizon) pair so the §04d/§04e
# diagnostic can compare which question the features actually answer.
# Rather than hand-rolling separate calls per spec, ``LabelSpec`` +
# ``compute_label_set`` accepts a list of specs and returns one np.ndarray
# per spec. Caller wraps in ``pd.DataFrame(...)`` to align with NB06's
# `concat(L1, L2, FI)` X-matrix.


@dataclass(frozen=True)
class LabelSpec:
    """One labeling target for the multi-target labeler.

    Each spec produces a single integer label column per row. Pick
    ``kind="tb"`` for the classical 3-class triple barrier or
    ``kind="mfe_mae"`` for the path-aware 5-class scheme.

    Attributes
    ----------
    name : str
        Output column name. Must be unique within a spec list.
    kind : str
        ``"tb"`` (triple_barrier_labels) or ``"mfe_mae"`` (mfe_mae_labels).
    horizon_ms : int
        Time barrier in milliseconds.
    up_target_ticks, down_target_ticks : float
        Profit barrier distances in ticks.
    stop_threshold_ticks : float
        Used only when ``kind="mfe_mae"`` — magnitude in ticks at which
        the opposite excursion turns a barrier hit into a "wicky"
        (±1) classification rather than "clean" (±2).
    tick_size : float
        Instrument tick size. NQ = 0.25.
    """
    name: str
    kind: str
    horizon_ms: int = 60_000
    up_target_ticks: float = 4.0
    down_target_ticks: float = 4.0
    stop_threshold_ticks: float = 1.5
    tick_size: float = 0.25


def compute_label_set(
    mid: np.ndarray,
    ts_ms: np.ndarray,
    specs: list[LabelSpec],
) -> dict[str, np.ndarray]:
    """Compute multiple label vectors in one call.

    Dispatches each ``LabelSpec`` to the appropriate underlying labeler
    (``triple_barrier_labels`` or ``mfe_mae_labels``) and returns one
    integer array per spec, keyed by ``spec.name``. Output integer
    arrays have shape ``(n,)`` and dtype ``int8`` (sufficient for both
    label schemes — values fit in {-2, -1, 0, +1, +2}).

    NB06 §04 wraps the result in ``pd.DataFrame(compute_label_set(...))``
    and trains one LightGBM model per column.

    Parameters
    ----------
    mid : np.ndarray
        1D array of mid prices.
    ts_ms : np.ndarray
        1D array of timestamps in milliseconds. Same length as ``mid``.
    specs : list[LabelSpec]
        One spec per output column. ``spec.name`` must be unique.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``spec.name → label_vector``. Vectors are ``float64`` with
        ``np.nan`` for rows whose horizon extends past the end of the
        observed data (``barrier_hit == "unknown"``). Resolved + timeout
        rows carry their integer label as a float (e.g. ``+1.0``,
        ``-2.0``, ``0.0``). Downstream callers filter via
        ``~pd.isna(...)`` before training.

    Raises
    ------
    ValueError
        If specs share a name, if a spec has unknown ``kind``, or if
        ``mid`` and ``ts_ms`` have mismatched lengths.
    """
    if len(mid) != len(ts_ms):
        raise ValueError(
            f"mid and ts_ms length mismatch: {len(mid)} vs {len(ts_ms)}",
        )
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate spec names in: {names}")

    out: dict[str, np.ndarray] = {}
    for spec in specs:
        if spec.kind == "tb":
            tb_labels = triple_barrier_labels(
                mid, ts_ms,
                horizon_ms=spec.horizon_ms,
                up_target_ticks=spec.up_target_ticks,
                down_target_ticks=spec.down_target_ticks,
                tick_size=spec.tick_size,
            )
            out[spec.name] = np.asarray(
                [
                    float("nan") if lbl.barrier_hit == "unknown"
                    else float(lbl.side)
                    for lbl in tb_labels
                ],
                dtype=np.float64,
            )
        elif spec.kind == "mfe_mae":
            mm_labels = mfe_mae_labels(
                mid, ts_ms,
                horizon_ms=spec.horizon_ms,
                up_target_ticks=spec.up_target_ticks,
                down_target_ticks=spec.down_target_ticks,
                stop_threshold_ticks=spec.stop_threshold_ticks,
                tick_size=spec.tick_size,
            )
            out[spec.name] = np.asarray(
                [
                    float("nan") if lbl.barrier_hit == "unknown"
                    else float(lbl.label)
                    for lbl in mm_labels
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(
                f"unknown LabelSpec.kind: {spec.kind!r}; "
                "must be 'tb' or 'mfe_mae'",
            )
    return out


# ---------------------------------------------------------------------------
# Pattern-firing labels — Wave 10-A
# ---------------------------------------------------------------------------
#
# For each row i, the label encodes whether (and which) library patterns
# fire within the next ``horizon_ms``. Caller runs the matcher in batch
# beforehand and supplies the resulting list of firing events; this
# labeler aggregates them into per-row labels under one of three
# encodings:
#
#   binary   →  0 / 1    (any pattern fires in window?)
#   first    →  0 / k    (k = ordinal of the first firing pattern)
#   best     →  0 / k    (k = ordinal of the highest-scoring firing)
#
# Produced labels feed two downstream paths:
#   * Strategy B — train a classifier to predict "will a pattern fire
#     soon?" so the engine can ride pattern fires with anticipation.
#   * Strategy C (10-C) — fused with the calibrated proba from Strategy A
#     to demand both legs agree before trade.
#
# Decoupled from the matcher to keep this module dependency-light.
# Caller (NB06 §02) runs ``matcher.match_all`` in a batch loop, collects
# results above the threshold, and passes them in as firing tuples.


@dataclass(frozen=True)
class FiringEvent:
    """One pattern firing at a specific snapshot.

    A typed alternative to the ``(int, str, float)`` tuple input — use
    whichever feels cleaner at the call site. Both are accepted by
    ``pattern_firing_labels``.
    """
    snapshot_idx: int   # row index into the snapshot/ts_ms array
    pattern_id: str
    score: float


def pattern_firing_labels(
    ts_ms: np.ndarray,
    firings: list,
    *,
    horizon_ms: int = 60_000,
    encoding: str = "binary",
    pattern_id_order: list[str] | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    """Per-row pattern-firing label given pre-computed firing events.

    For each row ``i`` (timestamp ``ts_ms[i]``), walks forward through
    sorted firings whose ``snapshot_idx >= i`` and aggregates those
    whose ``ts_ms[snapshot_idx] <= ts_ms[i] + horizon_ms``.

    Parameters
    ----------
    ts_ms : np.ndarray
        Per-row timestamps in milliseconds; length ``n``.
    firings : list[FiringEvent | tuple[int, str, float]]
        Pattern firings to aggregate. Each entry must be either a
        ``FiringEvent`` instance or a 3-tuple ``(snapshot_idx,
        pattern_id, score)``. ``snapshot_idx`` indexes into
        ``ts_ms``; out-of-range indices are skipped silently.
    horizon_ms : int
        Forward window in milliseconds. Default 60 seconds.
    encoding : {"binary", "first", "best"}
        - ``binary``: label ∈ {0, 1}; 1 if any firing in window.
        - ``first``: label ∈ {0, 1..K}; first firing's pattern ordinal.
        - ``best``: label ∈ {0, 1..K}; highest-score firing's ordinal.
    pattern_id_order : list[str], optional
        Stable ordinal encoding for multi-class modes. When ``None``
        (default), inferred from ``firings`` as the sorted unique
        ``pattern_id`` set. Pass an explicit order to keep encodings
        stable across runs (e.g. derived from
        ``library.get_active_patterns().keys()``).

    Returns
    -------
    labels : np.ndarray
        Shape ``(n,)``, ``float64``. Values are integer ordinals as floats
        (``0.0``, ``1.0``, ``2.0``, ...) for observable rows and
        ``np.nan`` for rows whose horizon extends past the end of the
        observed data **and** no firing was seen within the partial
        window. (If a firing is observed inside the partial window we
        already know the row is positive and emit the ordinal.)
    encoding_map : dict[int, str]
        ``ordinal -> pattern_id`` mapping. Empty for binary mode.

    Raises
    ------
    ValueError
        If ``encoding`` is not one of ``{"binary", "first", "best"}``.
    """
    if encoding not in ("binary", "first", "best"):
        raise ValueError(
            f"unknown encoding: {encoding!r}; "
            "must be 'binary', 'first', or 'best'",
        )

    n = len(ts_ms)
    ts_arr = np.asarray(ts_ms, dtype=np.int64)
    last_ts = int(ts_arr[-1]) if n > 0 else 0

    # Normalize firings → list of (snap_idx, pattern_id, score) tuples,
    # filter out-of-range, sort ascending by snap_idx for binary search.
    norm: list[tuple[int, str, float]] = []
    for f in firings:
        if isinstance(f, FiringEvent):
            si, pid, sc = f.snapshot_idx, f.pattern_id, f.score
        else:
            si, pid, sc = f[0], f[1], f[2]
        si = int(si)
        if 0 <= si < n:
            norm.append((si, str(pid), float(sc)))
    norm.sort(key=lambda t: t[0])

    # Build encoding map for multi-class modes.
    if encoding == "binary":
        encoding_map: dict[int, str] = {}
    else:
        if pattern_id_order is None:
            unique_ids = sorted({pid for _, pid, _ in norm})
        else:
            unique_ids = list(pattern_id_order)
        encoding_map = {i + 1: pid for i, pid in enumerate(unique_ids)}
        id_to_int = {pid: i + 1 for i, pid in enumerate(unique_ids)}
    labels = np.zeros(n, dtype=np.float64)

    # When firings list is empty, every row is either observable=0 or
    # unknown=NaN depending on whether its full horizon was observed.
    if not norm:
        for i in range(n):
            if int(ts_arr[i]) + int(horizon_ms) > last_ts:
                labels[i] = np.nan
        return labels, encoding_map

    # Build snap_idx index array for binary search per row.
    snap_idx_arr = np.asarray([t[0] for t in norm], dtype=np.int64)

    for i in range(n):
        deadline = int(ts_arr[i]) + int(horizon_ms)
        full_observable = last_ts >= deadline

        # First firing with snap_idx >= i.
        start = int(np.searchsorted(snap_idx_arr, i, side="left"))
        fired_value: float | None = None

        if start < len(norm):
            if encoding == "binary":
                for k in range(start, len(norm)):
                    snap_i = norm[k][0]
                    if int(ts_arr[snap_i]) > deadline:
                        break
                    fired_value = 1.0
                    break  # one firing satisfies binary
            elif encoding == "first":
                for k in range(start, len(norm)):
                    snap_i, pid, _ = norm[k]
                    if int(ts_arr[snap_i]) > deadline:
                        break
                    if pid in id_to_int:
                        fired_value = float(id_to_int[pid])
                        break
            else:  # encoding == "best"
                best_score = float("-inf")
                best_pid: str | None = None
                for k in range(start, len(norm)):
                    snap_i, pid, sc = norm[k]
                    if int(ts_arr[snap_i]) > deadline:
                        break
                    if sc > best_score and pid in id_to_int:
                        best_score = sc
                        best_pid = pid
                if best_pid is not None:
                    fired_value = float(id_to_int[best_pid])

        if fired_value is not None:
            # A firing was observed inside the window — label is known
            # positive whether or not the rest of the window is observed.
            labels[i] = fired_value
        elif full_observable:
            labels[i] = 0.0
        else:
            labels[i] = np.nan

    return labels, encoding_map
