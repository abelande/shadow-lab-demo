"""
p6lab.patterns.cascade_taxonomy
================================

Cascade Type A/B/C/D classifier — replaces the 8-tick-drop heuristic
used by NB07 §01 with a real behaviour-aware taxonomy backed by the
``cup_flip`` state machine.

Type definitions (p6-v2 SPEC.md §4, lines 76-79):

  - **A. LIQUIDITY_WITHDRAWAL** — slow-burn cascade driven by resting
    bids/asks being pulled rather than hit. Manifests as sustained
    *STALL* states with low ``pressure_acceleration`` and rising
    ``streak_exhaustion``.

  - **B. MOMENTUM_IGNITION** — fast directional move, minutes to hours.
    Manifests as *STREAK* states with high ``streak_velocity`` and
    positive ``pressure_acceleration`` (energy ratio > 1.0 — flow
    accelerating rather than decaying).

  - **C. GRINDING_CORRECTION** — multi-day drift under repeated failed
    breakouts. Requires multi-day input and is therefore only emitted
    when ``classify_snapshots`` is called with snapshots spanning ≥ 2
    distinct trading days.

  - **D. CROSS_INSTRUMENT_CONTAGION** — cascade in one instrument
    triggering correlated moves in others. **Placeholder** until
    cross-instrument input streams are wired (Wave 3). Never fires
    from single-instrument input.

The classifier reuses the existing ``cup_flip.TapeReader`` to turn an
event stream into a sequence of ``GameState`` objects, then pattern-
matches on the state sequence.

Typical use (replace NB07 §01 heuristic):

    from p6lab.patterns.cascade_taxonomy import CascadeClassifier
    clf = CascadeClassifier(tick_size=0.25)
    cascades = clf.classify_snapshots(snaps)
    for c in cascades:
        if c.cascade_type == CascadeType.MOMENTUM_IGNITION:
            ...
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

# Add projects/ to sys.path so the ``p6v2`` package (symlink of p6-v2) resolves.
# The cup_flip subpackage uses relative imports (``from ..models import ...``)
# so it must be imported as ``p6v2.cup_flip``, not a bare ``cup_flip``.
_PROJECTS = Path(__file__).resolve().parents[5]  # .../projects/
if str(_PROJECTS) not in sys.path:
    sys.path.insert(0, str(_PROJECTS))

from p6v2.cup_flip import TapeReader                              # noqa: E402
from p6v2.models import CupFlipState, GameState                   # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class CascadeType(str, Enum):
    """Four-way cascade taxonomy. String-valued so serializers round-trip cleanly."""
    LIQUIDITY_WITHDRAWAL = "A_liquidity_withdrawal"
    MOMENTUM_IGNITION = "B_momentum_ignition"
    GRINDING_CORRECTION = "C_grinding_correction"
    CROSS_INSTRUMENT_CONTAGION = "D_cross_instrument_contagion"


@dataclass(frozen=True)
class CascadeEvent:
    """One detected cascade with anchor timestamp + metadata."""
    cascade_type: CascadeType
    anchor_ts_ms: int
    end_ts_ms: int
    confidence: float            # [0, 1] — classifier's own confidence
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detection thresholds (exposed as class-level defaults so tests can override)
# ---------------------------------------------------------------------------

@dataclass
class CascadeThresholds:
    """All tunable thresholds in one place — easier to sweep than inline constants."""
    # Type A (liquidity withdrawal / slow burn)
    liquidity_min_stall_duration_ms: int = 60_000     # 60s of sustained stall
    liquidity_min_stall_count: int = 3                 # minimum failed fills
    liquidity_max_pressure_accel: float = 0.9          # flow NOT accelerating
    liquidity_min_streak_exhaustion: float = 0.5       # momentum is dying

    # Type B (momentum ignition)
    momentum_min_streak_length: int = 3
    momentum_min_streak_velocity: float = 2.0          # levels/sec
    momentum_min_pressure_accel: float = 1.0           # flow accelerating
    momentum_cooldown_ms: int = 5_000                  # min gap between Type B events

    # Type C (grinding correction) — multi-day only
    grinding_min_duration_ms: int = 86_400_000         # 1 day (ms)
    grinding_min_stall_transitions: int = 8            # many failed-breakout bars

    # Type A cooldown (prevents flooding on sustained withdrawal)
    liquidity_cooldown_ms: int = 30_000


# ---------------------------------------------------------------------------
# CascadeClassifier
# ---------------------------------------------------------------------------

class CascadeClassifier:
    """Wraps cup_flip.TapeReader + state-sequence analysis → cascade events."""

    def __init__(
        self,
        *,
        tick_size: float = 0.25,
        stop_run_levels: int = 5,
        thresholds: CascadeThresholds | None = None,
        min_streak_length: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        min_streak_length
            Wave 4 Phase 1C: override the TapeReader's internal
            StreakDetector(min_streak_length=3) for smoketest-scale runs.
            When None, uses the cup_flip default (3). Smoketest runs (<5k
            snapshots) typically pass 2 alongside the relaxed
            ``CascadeThresholds.momentum_min_streak_length``.
        """
        self.tick_size = tick_size
        self.tape_reader = TapeReader(stop_run_levels=stop_run_levels)
        if min_streak_length is not None and min_streak_length > 0:
            # Patch the TapeReader's streak detector to a smaller threshold.
            # cup_flip lives under p6v2 in this repo; use the same path the
            # module-level TapeReader import relies on.
            from p6v2.cup_flip.streak_detector import StreakDetector
            self.tape_reader.streak_detector = StreakDetector(
                min_streak_length=int(min_streak_length)
            )
        self.thresholds = thresholds or CascadeThresholds()
        self._history: list[GameState] = []

    # ------------------------------------------------------------------
    # Batch entry point (used by NB07)
    # ------------------------------------------------------------------

    def classify_snapshots(self, snapshots: Iterable[Any]) -> list[CascadeEvent]:
        """Walk a sequence of ``OrderBookSnapshot`` objects, classify cascades.

        Consumes each snapshot's ``recent_events`` list through the
        TapeReader to produce one GameState per snapshot; then pattern-
        matches on the full GameState sequence.
        """
        # Wave 4 Phase 1C: some ingestion paths emit events whose .action
        # and .side are ``p6.models`` enum instances, while TapeReader's
        # StreakDetector compares against ``p6v2.models`` enums. Enum
        # identity fails across modules even when the values match. Cache
        # the expected enum classes once so we can rebuild events.
        from p6v2.models import Order as _P6V2Order
        from p6v2.models import OrderAction as _P6V2Action
        from p6v2.models import Side as _P6V2Side

        def _normalize(ev):
            """Return the event with .action/.side normalized to p6v2 enums."""
            action = getattr(ev, "action", None)
            side = getattr(ev, "side", None)
            action_name = getattr(action, "name", None) or str(action).split(".")[-1]
            side_name = getattr(side, "name", None) or (str(side).split(".")[-1] if side is not None else None)
            # Fast-path: already a p6v2 Order
            if isinstance(ev, _P6V2Order):
                return ev
            try:
                new_action = _P6V2Action[action_name] if action_name else None
                new_side = _P6V2Side[side_name] if side_name else None
            except (KeyError, TypeError):
                return ev
            if new_action is None:
                return ev
            try:
                return _P6V2Order(
                    timestamp_ms=int(getattr(ev, "timestamp_ms", 0)),
                    order_id=str(getattr(ev, "order_id", "")),
                    action=new_action,
                    side=new_side if new_side is not None else _P6V2Side.BID,
                    price=float(getattr(ev, "price", 0.0) or 0.0),
                    size=float(getattr(ev, "size", 0.0) or 0.0),
                )
            except Exception:
                return ev

        self._history.clear()
        snaps = list(snapshots)
        for s in snaps:
            raw_events = list(getattr(s, "recent_events", []) or [])
            events = [_normalize(ev) for ev in raw_events]
            # Fall back to best_bid/ask from the snapshot if present
            best_bid = getattr(s, "best_bid", None)
            best_ask = getattr(s, "best_ask", None)
            gs = self.tape_reader.update(
                events=events,
                timestamp_ms=int(getattr(s, "timestamp_ms", 0)),
                best_bid=best_bid, best_ask=best_ask,
                snapshot=s,
            )
            self._history.append(gs)

        return self._detect_all(self._history)

    # ------------------------------------------------------------------
    # Detection rules — one method per type so they can be unit tested
    # ------------------------------------------------------------------

    def _detect_all(self, history: list[GameState]) -> list[CascadeEvent]:
        events: list[CascadeEvent] = []
        events.extend(self._detect_momentum_ignition(history))
        events.extend(self._detect_liquidity_withdrawal(history))
        events.extend(self._detect_grinding_correction(history))
        events.extend(self._detect_cross_instrument_contagion(history))
        events.sort(key=lambda e: e.anchor_ts_ms)
        return events

    # ------------------------------------------------------------------
    # Type D — cross-instrument contagion (Wave 7 Phase 7G)
    # ------------------------------------------------------------------

    #: Coherence needed between two symbols (7C) for them to count as
    #: a co-moving pair. Deliberately permissive — peer correlation is
    #: already gating upstream.
    CROSS_INSTRUMENT_MIN_COHERENCE: float = 0.55
    CROSS_INSTRUMENT_MIN_ADJACENCY: float = 0.45
    CROSS_INSTRUMENT_COOLDOWN_MS: int = 15_000

    def detect_cross_instrument_contagion(
        self,
        *,
        events_by_symbol: dict[str, list["CascadeEvent"]],
        coherence_matrix: dict[tuple[str, str], float] | None = None,
        adjacency_matrix: dict[tuple[str, str], float] | None = None,
    ) -> list["CascadeEvent"]:
        """Given per-symbol Type-A/B/C event timelines + optional peer
        coherence/adjacency dicts, emit one Type-D event per detected
        contagion cluster.

        The algorithm: sort every symbol's events by anchor timestamp,
        then walk forward looking for clusters where ≥ 2 symbols fire
        within 1 second of each other AND the pair's coherence (or
        adjacency) clears the Wave 7 threshold. The output carries the
        contributing symbols in ``metadata['symbols']``.
        """
        all_events: list[tuple[int, str, CascadeEvent]] = []
        for sym, evs in events_by_symbol.items():
            for ev in evs:
                all_events.append((ev.anchor_ts_ms, sym, ev))
        all_events.sort(key=lambda x: x[0])

        cluster_window_ms = 1_000
        out: list[CascadeEvent] = []
        last_fire_ts = -10**18
        i = 0
        while i < len(all_events):
            anchor_ts, anchor_sym, _anchor_ev = all_events[i]
            cluster_syms: set[str] = {anchor_sym}
            cluster_end_ts = anchor_ts
            j = i + 1
            while j < len(all_events) and all_events[j][0] - anchor_ts <= cluster_window_ms:
                _, sym_j, _ = all_events[j]
                if sym_j != anchor_sym:
                    # Gate on peer coherence / adjacency if supplied
                    pair = tuple(sorted((anchor_sym, sym_j)))
                    ok = self._pair_passes_gate(pair, coherence_matrix, adjacency_matrix)
                    if ok:
                        cluster_syms.add(sym_j)
                        cluster_end_ts = all_events[j][0]
                j += 1
            if len(cluster_syms) >= 2 and anchor_ts - last_fire_ts >= self.CROSS_INSTRUMENT_COOLDOWN_MS:
                out.append(CascadeEvent(
                    cascade_type=CascadeType.CROSS_INSTRUMENT_CONTAGION,
                    anchor_ts_ms=anchor_ts,
                    end_ts_ms=cluster_end_ts,
                    confidence=min(1.0, 0.5 + 0.15 * (len(cluster_syms) - 2)),
                    metadata={
                        "symbols": sorted(cluster_syms),
                        "cluster_size": len(cluster_syms),
                    },
                ))
                last_fire_ts = anchor_ts
            i += 1
        return out

    def _detect_cross_instrument_contagion(
        self, history: list[GameState],
    ) -> list["CascadeEvent"]:
        """Single-instrument input never triggers Type D — the multi-
        symbol variant is the one above, called from
        ``MultiSymbolRunner`` (Phase 7A)."""
        return []

    def _pair_passes_gate(
        self,
        pair: tuple[str, str],
        coherence_matrix: dict[tuple[str, str], float] | None,
        adjacency_matrix: dict[tuple[str, str], float] | None,
    ) -> bool:
        """Accept the pair if either the coherence OR the adjacency
        matrix entry clears its threshold. When both are None we pass
        (keeps the detector usable in unit tests without Wave-7 plumbing)."""
        if coherence_matrix is None and adjacency_matrix is None:
            return True
        coh = coherence_matrix.get(pair, 0.0) if coherence_matrix else 0.0
        adj = adjacency_matrix.get(pair, 0.0) if adjacency_matrix else 0.0
        return coh >= self.CROSS_INSTRUMENT_MIN_COHERENCE or \
               adj >= self.CROSS_INSTRUMENT_MIN_ADJACENCY

    def _detect_momentum_ignition(
        self, history: list[GameState],
    ) -> list[CascadeEvent]:
        """Type B — streak with high velocity AND accelerating pressure."""
        th = self.thresholds
        out: list[CascadeEvent] = []
        in_streak = False
        streak_start_ts: int = 0
        last_fire_ts: int = -10**18

        for gs in history:
            is_streak = gs.state in (CupFlipState.BULL_STREAK, CupFlipState.BEAR_STREAK)
            if is_streak and not in_streak:
                in_streak = True
                streak_start_ts = gs.timestamp_ms
            elif not is_streak and in_streak:
                in_streak = False

            if not is_streak:
                continue
            if gs.streak_length < th.momentum_min_streak_length:
                continue
            if gs.streak_velocity < th.momentum_min_streak_velocity:
                continue
            if gs.pressure_acceleration < th.momentum_min_pressure_accel:
                continue
            if gs.timestamp_ms - last_fire_ts < th.momentum_cooldown_ms:
                continue

            # Confidence blends velocity + accel normalization
            conf = min(1.0, 0.5 * (gs.streak_velocity / max(th.momentum_min_streak_velocity, 1))
                           + 0.5 * (gs.pressure_acceleration / max(th.momentum_min_pressure_accel, 1)))
            out.append(CascadeEvent(
                cascade_type=CascadeType.MOMENTUM_IGNITION,
                anchor_ts_ms=streak_start_ts,
                end_ts_ms=gs.timestamp_ms,
                confidence=min(conf, 1.0),
                metadata={
                    "state": gs.state.value,
                    "streak_length": gs.streak_length,
                    "streak_velocity": gs.streak_velocity,
                    "pressure_acceleration": gs.pressure_acceleration,
                },
            ))
            last_fire_ts = gs.timestamp_ms
        return out

    def _detect_liquidity_withdrawal(
        self, history: list[GameState],
    ) -> list[CascadeEvent]:
        """Type A — sustained stall, momentum dying, flow not accelerating."""
        th = self.thresholds
        out: list[CascadeEvent] = []
        stall_start_ts: Optional[int] = None
        last_fire_ts: int = -10**18

        for gs in history:
            is_stall = gs.state in (CupFlipState.BULL_STALL, CupFlipState.BEAR_STALL)
            if is_stall and stall_start_ts is None:
                stall_start_ts = gs.timestamp_ms
            elif not is_stall and stall_start_ts is not None:
                stall_start_ts = None

            if not is_stall or stall_start_ts is None:
                continue
            duration_ms = gs.timestamp_ms - stall_start_ts
            if duration_ms < th.liquidity_min_stall_duration_ms:
                continue
            if gs.stall_count < th.liquidity_min_stall_count:
                continue
            if gs.pressure_acceleration >= th.liquidity_max_pressure_accel:
                continue
            if gs.streak_exhaustion < th.liquidity_min_streak_exhaustion:
                continue
            if gs.timestamp_ms - last_fire_ts < th.liquidity_cooldown_ms:
                continue

            conf = min(1.0, 0.5 * (gs.stall_count / max(th.liquidity_min_stall_count, 1))
                           + 0.5 * gs.streak_exhaustion)
            out.append(CascadeEvent(
                cascade_type=CascadeType.LIQUIDITY_WITHDRAWAL,
                anchor_ts_ms=stall_start_ts,
                end_ts_ms=gs.timestamp_ms,
                confidence=conf,
                metadata={
                    "state": gs.state.value,
                    "stall_count": gs.stall_count,
                    "streak_exhaustion": gs.streak_exhaustion,
                    "pressure_acceleration": gs.pressure_acceleration,
                    "duration_ms": duration_ms,
                },
            ))
            last_fire_ts = gs.timestamp_ms
        return out

    def _detect_grinding_correction(
        self, history: list[GameState],
    ) -> list[CascadeEvent]:
        """Type C — multi-day drift with many failed breakouts.

        Only fires when the input spans >= 2 distinct calendar dates.
        """
        th = self.thresholds
        if not history:
            return []
        # Check multi-day span
        ts_min = history[0].timestamp_ms
        ts_max = history[-1].timestamp_ms
        span_ms = ts_max - ts_min
        if span_ms < th.grinding_min_duration_ms:
            return []

        # Count stall transitions (entered-stall events)
        transitions = 0
        prev_stall = False
        for gs in history:
            is_stall = gs.state in (CupFlipState.BULL_STALL, CupFlipState.BEAR_STALL)
            if is_stall and not prev_stall:
                transitions += 1
            prev_stall = is_stall

        if transitions < th.grinding_min_stall_transitions:
            return []

        return [CascadeEvent(
            cascade_type=CascadeType.GRINDING_CORRECTION,
            anchor_ts_ms=ts_min,
            end_ts_ms=ts_max,
            confidence=min(1.0, transitions / (th.grinding_min_stall_transitions * 2)),
            metadata={
                "span_ms": span_ms,
                "stall_transitions": transitions,
                "n_snapshots": len(history),
            },
        )]


# ---------------------------------------------------------------------------
# Convenience helpers for the notebook
# ---------------------------------------------------------------------------

def cascade_events_to_df(events: list[CascadeEvent]):
    """Convert a list of CascadeEvent to a ``pd.DataFrame`` for tabular reports."""
    import pandas as pd
    if not events:
        return pd.DataFrame(columns=[
            "cascade_type", "anchor_ts_ms", "end_ts_ms", "confidence", "metadata",
        ])
    return pd.DataFrame([
        {
            "cascade_type": e.cascade_type.value,
            "anchor_ts_ms": e.anchor_ts_ms,
            "end_ts_ms":    e.end_ts_ms,
            "confidence":   round(e.confidence, 3),
            "metadata":     e.metadata,
        }
        for e in events
    ])
