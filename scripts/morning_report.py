#!/usr/bin/env python3
"""
Morning Prep Report — L3 Replay Forensics

Replays yesterday's L3 MBO data through Layer 4 (spoof detection) + Layer 1 (fragility)
+ Layer 2 (cup flip / game state) and outputs a structured trading prep summary.

Usage:
    python morning_report.py data/nq-mbo-2026-03-20.dbn.zst
    python morning_report.py data/es-mbo-2026-03-20.dbn.zst --rth-only
    python morning_report.py --files data/nq-mbo-2026-03-18.dbn.zst data/nq-mbo-2026-03-19.dbn.zst data/nq-mbo-2026-03-20.dbn.zst
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Add project root to path so p6-v2 can be imported as a package
_project_root = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_project_root)
_pkg_link = os.path.join(_parent, "p6v2")
if not os.path.exists(_pkg_link):
    os.symlink(_project_root, _pkg_link)
sys.path.insert(0, _parent)

from p6v2.models import (
    OrderBookSnapshot, Side, SpoofType, SpoofEvent,
    FragilityState, StaircaseProfile, GameState, CupFlipState,
)
from p6v2.ingestion.databento_feed import DatabentoReplayFeed
from p6v2.spoof_detection.pull_before_touch import PullBeforeTouchDetector
from p6v2.spoof_detection.layering_detector import LayeringDetector
from p6v2.spoof_detection.iceberg_inference import IcebergInference
from p6v2.spoof_detection.phantom_wall import PhantomWallDetector
from p6v2.spoof_detection.authenticity_scorer import AuthenticityScorer
from p6v2.staircase_analyzer.fragility_scorer import FragilityScorer
from p6v2.cup_flip.tape_reader import TapeReader

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

_ET = ZoneInfo("America/New_York")


# ── Instrument Config ──────────────────────────────────────────────

@dataclass
class InstrumentConfig:
    symbol: str
    tick_size: float         # NQ=0.25, ES=0.25
    point_value: float       # NQ=20, ES=50
    bucket_size: float       # price bucket for spoof clustering
    zone_bucket: float       # auth zone size (NQ=5.0, ES=2.0)
    persistence_pct: float   # fragility map threshold (0.01 = 1%)
    persistence_min: int     # minimum snapshot count for persistence
    iceberg_vol_threshold: float  # min contracts for "actionable" iceberg
    stop_run_levels: int     # StopRunDetector threshold
    session_tz: str = "America/New_York"


_INSTRUMENT_CONFIGS: Dict[str, InstrumentConfig] = {
    "NQ": InstrumentConfig(
        symbol="NQ",
        tick_size=0.25,
        point_value=20.0,
        bucket_size=0.25,
        zone_bucket=5.0,
        persistence_pct=0.01,
        persistence_min=50,
        iceberg_vol_threshold=10.0,
        stop_run_levels=5,
    ),
    "ES": InstrumentConfig(
        symbol="ES",
        tick_size=0.25,
        point_value=50.0,
        bucket_size=0.25,
        zone_bucket=2.0,
        persistence_pct=0.01,
        persistence_min=50,
        iceberg_vol_threshold=30.0,
        stop_run_levels=3,
    ),
    "CL": InstrumentConfig(
        symbol="CL",
        tick_size=0.01,          # crude oil = 1 cent
        point_value=1000.0,      # $1000 per point ($10 per tick)
        bucket_size=0.01,        # cluster spoofs by tick
        zone_bucket=0.25,        # auth zones every 25 cents
        persistence_pct=0.01,
        persistence_min=50,
        iceberg_vol_threshold=20.0,  # CL has decent volume
        stop_run_levels=5,       # CL can rip — need higher bar
    ),
    "GC": InstrumentConfig(
        symbol="GC",
        tick_size=0.10,          # gold = 10 cents
        point_value=100.0,       # $100 per point ($10 per tick)
        bucket_size=0.10,        # cluster by tick
        zone_bucket=2.0,         # auth zones every $2
        persistence_pct=0.01,
        persistence_min=50,
        iceberg_vol_threshold=15.0,  # GC thinner than ES
        stop_run_levels=4,
    ),
    "SI": InstrumentConfig(
        symbol="SI",
        tick_size=0.005,         # silver = half cent
        point_value=5000.0,      # $5000 per point ($25 per tick)
        bucket_size=0.005,       # cluster by tick
        zone_bucket=0.10,        # auth zones every 10 cents
        persistence_pct=0.02,    # SI is thinner — raise threshold
        persistence_min=30,      # fewer snapshots needed
        iceberg_vol_threshold=5.0,   # SI much thinner book
        stop_run_levels=4,
    ),
}


def _detect_instrument(file_path: str, symbol_override: Optional[str] = None) -> InstrumentConfig:
    """Auto-detect instrument from filename, or use explicit --symbol override."""
    if symbol_override:
        key = symbol_override.upper()
        if key in _INSTRUMENT_CONFIGS:
            return _INSTRUMENT_CONFIGS[key]
        # Try stripping month code (e.g. "ESH6" → "ES")
        root = "".join(c for c in key if c.isalpha())
        if root in _INSTRUMENT_CONFIGS:
            return _INSTRUMENT_CONFIGS[root]
        raise ValueError(f"Unknown symbol '{symbol_override}'. Available: {list(_INSTRUMENT_CONFIGS.keys())}")

    name = os.path.basename(file_path).upper()
    for sym, cfg in _INSTRUMENT_CONFIGS.items():
        if name.startswith(sym):
            return cfg
    # For full-exchange files (glbx-mdp3-...), require --symbol
    if name.startswith("GLBX"):
        raise ValueError(
            "Full-exchange file detected. Use --symbol to specify instrument "
            f"(e.g. --symbol NQ). Available: {list(_INSTRUMENT_CONFIGS.keys())}"
        )
    # Default to NQ
    return _INSTRUMENT_CONFIGS["NQ"]


def _ms_to_et(ts_ms: int) -> str:
    """Convert millisecond UTC timestamp to HH:MM:SS ET string."""
    if ts_ms == 0:
        return "??:??:??"
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).astimezone(_ET)
    return dt.strftime("%H:%M:%S")


def _ms_to_et_date(ts_ms: int) -> str:
    """Convert millisecond UTC timestamp to 'DayName Month DD YYYY' ET string."""
    if ts_ms == 0:
        return ""
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).astimezone(_ET)
    return dt.strftime("%A %B %d %Y")


# ── Accumulation tracking ──────────────────────────────────────────

@dataclass
class IcebergZone:
    price: float
    side: Side
    total_refills: int = 0
    total_volume: float = 0.0
    first_seen_ms: int = 0
    last_seen_ms: int = 0
    detections: int = 0
    max_confidence: float = 0.0


@dataclass
class SpoofZone:
    price: float
    side: Side
    spoof_type: SpoofType
    event_count: int = 0
    max_confidence: float = 0.0
    first_seen_ms: int = 0
    last_seen_ms: int = 0
    game_states: List[str] = field(default_factory=list)  # CupFlipState names seen


@dataclass
class FragilityRecord:
    price: float
    side: Side
    fragile_count: int = 0
    solid_count: int = 0
    moderate_count: int = 0
    total_snapshots: int = 0
    avg_volume: float = 0.0
    avg_order_count: float = 0.0
    overlapping_spoof: bool = False

    @property
    def fragile_pct(self) -> float:
        return self.fragile_count / max(self.total_snapshots, 1) * 100

    @property
    def solid_pct(self) -> float:
        return self.solid_count / max(self.total_snapshots, 1) * 100

    @property
    def dominant_state(self) -> str:
        m = max(self.fragile_count, self.solid_count, self.moderate_count)
        if m == self.fragile_count:
            return "FRAGILE"
        elif m == self.solid_count:
            return "SOLID"
        return "MODERATE"


@dataclass
class AuthZone:
    """Authenticity tracking bucketed by (price_bucket, time_window)."""
    price_bucket: float
    time_window: int   # 30-minute block index
    scores: List[float] = field(default_factory=list)

    @property
    def mean_auth(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 1.0


# ── Report Builder ─────────────────────────────────────────────────

class MorningReportBuilder:
    def __init__(self, cfg: InstrumentConfig):
        self.cfg = cfg

        # Detectors
        self.pull = PullBeforeTouchDetector()
        self.layering = LayeringDetector()
        self.iceberg = IcebergInference()
        self.phantom = PhantomWallDetector()
        self.auth_scorer = AuthenticityScorer()
        self.fragility = FragilityScorer()
        self.tape_reader = TapeReader(stop_run_levels=cfg.stop_run_levels)

        # Accumulators
        self.iceberg_zones: Dict[Tuple[float, Side], IcebergZone] = {}
        self.spoof_zones: Dict[Tuple[float, Side, SpoofType], SpoofZone] = {}
        self.fragility_records: Dict[Tuple[float, Side], FragilityRecord] = {}
        self.auth_zones: Dict[Tuple[float, int], AuthZone] = {}
        self.authenticity_scores: List[float] = []
        self.snapshot_count = 0
        self.price_range: List[float] = []
        self.first_ts_ms: int = 0
        self.last_ts_ms: int = 0

        # Current game state for annotation
        self._current_game_state: Optional[GameState] = None

    def _time_window(self, ts_ms: int) -> int:
        """Return 30-minute window index from epoch ms."""
        return ts_ms // (30 * 60 * 1000)

    def process_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        self.snapshot_count += 1

        ts = snapshot.timestamp_ms
        if self.first_ts_ms == 0:
            self.first_ts_ms = ts
        self.last_ts_ms = ts

        if snapshot.mid_price:
            self.price_range.append(snapshot.mid_price)

        # ── Layer 2: Tape / Cup Flip ──
        self._current_game_state = self.tape_reader.update(
            snapshot.recent_events, ts,
            best_bid=snapshot.best_bid, best_ask=snapshot.best_ask,
        )

        # ── Layer 4: Spoof Detection ──
        spoof_events: List[SpoofEvent] = []
        spoof_events += self.pull.detect(
            snapshot.recent_events, snapshot.best_bid, snapshot.best_ask
        )
        spoof_events += self.layering.detect(snapshot.recent_events)
        spoof_events += self.iceberg.detect(snapshot.recent_events)
        spoof_events += self.phantom.detect(snapshot.recent_events, snapshot.mid_price)

        auth = self.auth_scorer.score(spoof_events, ts)
        self.authenticity_scores.append(auth.authenticity_score)

        # Track zone-based authenticity
        if snapshot.mid_price:
            bucket = round(snapshot.mid_price / self.cfg.zone_bucket) * self.cfg.zone_bucket
            win = self._time_window(ts)
            az_key = (bucket, win)
            if az_key not in self.auth_zones:
                self.auth_zones[az_key] = AuthZone(price_bucket=bucket, time_window=win)
            self.auth_zones[az_key].scores.append(auth.authenticity_score)

        for ev in spoof_events:
            if ev.spoof_type == SpoofType.ICEBERG:
                # Track icebergs separately
                key = (ev.price, ev.side)
                if key not in self.iceberg_zones:
                    self.iceberg_zones[key] = IcebergZone(
                        price=ev.price, side=ev.side, first_seen_ms=ts,
                    )
                zone = self.iceberg_zones[key]
                zone.detections += 1
                zone.last_seen_ms = ts
                zone.max_confidence = max(zone.max_confidence, ev.confidence)
                if "total" in ev.details:
                    try:
                        zone.total_volume = max(zone.total_volume, float(ev.details.split("total ")[-1]))
                    except (ValueError, IndexError):
                        pass
                if "fills" in ev.details:
                    try:
                        zone.total_refills = max(zone.total_refills, int(ev.details.split(" fills")[0]))
                    except (ValueError, IndexError):
                        pass
            else:
                # Cluster by (price_bucket, side, type)
                bucket = round(ev.price / self.cfg.bucket_size) * self.cfg.bucket_size
                key = (bucket, ev.side, ev.spoof_type)
                if key not in self.spoof_zones:
                    self.spoof_zones[key] = SpoofZone(
                        price=bucket, side=ev.side, spoof_type=ev.spoof_type,
                        first_seen_ms=ts,
                    )
                sz = self.spoof_zones[key]
                sz.event_count += 1
                sz.last_seen_ms = ts
                sz.max_confidence = max(sz.max_confidence, ev.confidence)
                if self._current_game_state:
                    gs_name = self._current_game_state.state.value
                    if gs_name not in sz.game_states:
                        sz.game_states.append(gs_name)

        # ── Layer 1: Fragility ──
        staircase = self.fragility.build_profile(snapshot)
        for lp in staircase.bid_levels + staircase.ask_levels:
            key = (lp.price, lp.side)
            if key not in self.fragility_records:
                self.fragility_records[key] = FragilityRecord(price=lp.price, side=lp.side)
            rec = self.fragility_records[key]
            rec.total_snapshots += 1
            rec.avg_volume += lp.volume
            rec.avg_order_count += lp.order_count
            if lp.fragility == FragilityState.FRAGILE:
                rec.fragile_count += 1
            elif lp.fragility == FragilityState.SOLID:
                rec.solid_count += 1
            else:
                rec.moderate_count += 1

    def _finalize(self) -> None:
        """Finalize averages and cross-reference spoof/fragility overlap."""
        for rec in self.fragility_records.values():
            if rec.total_snapshots > 0:
                rec.avg_volume /= rec.total_snapshots
                rec.avg_order_count /= rec.total_snapshots

        # Mark fragility records that overlap with spoof zones
        spoof_prices = {sz.price for sz in self.spoof_zones.values()}
        for rec in self.fragility_records.values():
            if round(rec.price / self.cfg.bucket_size) * self.cfg.bucket_size in spoof_prices:
                rec.overlapping_spoof = True

    def generate_report(self, file_path: str, date_override: Optional[str] = None) -> str:
        self._finalize()
        cfg = self.cfg
        lines = []

        # ── Header ──
        sym = cfg.symbol
        if date_override:
            date_str = date_override
        else:
            date_str = _ms_to_et_date(self.first_ts_ms) if self.first_ts_ms else "Unknown Date"
        start_t = _ms_to_et(self.first_ts_ms)
        end_t = _ms_to_et(self.last_ts_ms)
        low = min(self.price_range) if self.price_range else 0.0
        high = max(self.price_range) if self.price_range else 0.0
        open_p = self.price_range[0] if self.price_range else 0.0
        close_p = self.price_range[-1] if self.price_range else 0.0
        change = close_p - open_p

        lines.append("═" * 63)
        lines.append(f"  {sym} MORNING PREP — {date_str}")
        lines.append(f"  L3 Forensics: Spoof Detection + Fragility Analysis")
        lines.append("═" * 63)
        if self.price_range:
            lines.append(f"  Session:   {low:.2f} → {high:.2f}  (Δ {change:+.2f})")
            lines.append(f"  Open: {open_p:.2f}  Close: {close_p:.2f}")
        lines.append(f"  Time:      {start_t} → {end_t} ET")
        lines.append(f"  Snapshots: {self.snapshot_count:,} @ 500ms intervals")
        lines.append("")

        # ── Spoof Activity — BID side ──
        bid_spoof = sorted(
            [sz for sz in self.spoof_zones.values() if sz.side == Side.BID],
            key=lambda s: s.event_count, reverse=True
        )
        ask_spoof = sorted(
            [sz for sz in self.spoof_zones.values() if sz.side == Side.ASK],
            key=lambda s: s.event_count, reverse=True
        )

        _type_emoji = {
            SpoofType.PULL_BEFORE_TOUCH: "⚡",
            SpoofType.LAYERING: "📚",
            SpoofType.PHANTOM_WALL: "👻",
            SpoofType.STUFFING: "📦",
        }

        def _spoof_line(sz: SpoofZone) -> str:
            emoji = _type_emoji.get(sz.spoof_type, "❓")
            conf_pct = int(sz.max_confidence * 100)
            t0 = _ms_to_et(sz.first_seen_ms)
            t1 = _ms_to_et(sz.last_seen_ms)
            warn = "  ⚠️ HIGH" if conf_pct > 60 else ""
            gs_label = ""
            if sz.game_states:
                gs_label = f"  [{', '.join(sz.game_states)}]"
            return (
                f"  {emoji} {sz.price:.2f}  {sz.spoof_type.value} × {sz.event_count}"
                f"  conf {conf_pct}%  {t0}–{t1}{warn}{gs_label}"
            )

        lines.append("🎭 SPOOF ACTIVITY — BID SIDE (fake support)")
        if bid_spoof:
            for sz in bid_spoof[:15]:
                lines.append(_spoof_line(sz))
        else:
            lines.append("  None detected.")
        lines.append("")

        lines.append("🎭 SPOOF ACTIVITY — ASK SIDE (fake resistance)")
        if ask_spoof:
            for sz in ask_spoof[:15]:
                lines.append(_spoof_line(sz))
        else:
            lines.append("  None detected.")
        lines.append("")

        # ── Iceberg Accumulation ──
        icebergs = sorted(
            self.iceberg_zones.values(),
            key=lambda z: z.total_volume, reverse=True
        )
        lines.append("─" * 63)
        lines.append("🧊 ICEBERG ACCUMULATION ZONES")
        lines.append("─" * 63)
        if not icebergs:
            lines.append("  No icebergs detected.")
        else:
            for z in icebergs[:15]:
                side_label = "BID (buying)" if z.side == Side.BID else "ASK (selling)"
                dur_s = (z.last_seen_ms - z.first_seen_ms) / 1000
                lines.append(
                    f"  ★ {z.price:.2f} — {side_label}  "
                    f"{z.total_refills} refills  ~{z.total_volume:.0f} contracts  "
                    f"conf {z.max_confidence:.0%}  active {dur_s:.0f}s  "
                    f"({z.detections} snaps)"
                )
        lines.append("")

        # ── Fragility Ladder ──
        threshold = max(
            int(self.snapshot_count * cfg.persistence_pct),
            cfg.persistence_min
        )
        persistent = [
            r for r in self.fragility_records.values()
            if r.total_snapshots >= threshold
        ]
        bid_levels = sorted(
            [r for r in persistent if r.side == Side.BID],
            key=lambda r: r.price, reverse=True
        )
        ask_levels = sorted(
            [r for r in persistent if r.side == Side.ASK],
            key=lambda r: r.price
        )

        _state_emoji = {"FRAGILE": "🔴", "SOLID": "🟢", "MODERATE": "🟡"}

        def _frag_line(r: FragilityRecord) -> str:
            e = _state_emoji.get(r.dominant_state, "⚪")
            spoof_tag = "  ← SPOOFED" if r.overlapping_spoof else ""
            return (
                f"  {e} {r.price:.2f}  {r.dominant_state}  "
                f"{r.fragile_pct:.0f}% fragile  "
                f"avg {r.avg_volume:.0f} vol / {r.avg_order_count:.0f} orders"
                f"{spoof_tag}"
            )

        lines.append("─" * 63)
        lines.append("🏗️  FRAGILITY LADDER (persistent levels, nearest to close)")
        lines.append("─" * 63)
        lines.append("")

        # ASK side (above close)
        ask_near_close = [r for r in ask_levels if r.price >= close_p][:10]
        if ask_near_close:
            lines.append("  ASK ────────────────────────────────────────────")
            for r in ask_near_close:
                lines.append(_frag_line(r))

        if close_p > 0:
            lines.append(f"  ─── {close_p:.2f} ── CLOSE ──────────────────────────")

        # BID side (below close)
        bid_near_close = [r for r in bid_levels if r.price <= close_p][:10]
        if bid_near_close:
            for r in bid_near_close:
                lines.append(_frag_line(r))
            lines.append("  BID ────────────────────────────────────────────")
        lines.append("")

        # ── Actionable Levels ──
        lines.append("─" * 63)
        lines.append("🎯 ACTIONABLE LEVELS")
        lines.append("─" * 63)
        lines.append("")

        high_conf: List[str] = []
        watch: List[str] = []

        # Icebergs
        bid_icebergs = [z for z in icebergs if z.side == Side.BID and z.total_volume >= cfg.iceberg_vol_threshold]
        ask_icebergs = [z for z in icebergs if z.side == Side.ASK and z.total_volume >= cfg.iceberg_vol_threshold]
        for z in bid_icebergs[:5]:
            entry = f"  📈 {z.price:.2f}  Institutional BUYING  (~{z.total_volume:.0f} contracts, {z.total_refills} refills)"
            (high_conf if z.max_confidence > 0.6 else watch).append(entry)
        for z in ask_icebergs[:5]:
            entry = f"  📉 {z.price:.2f}  Institutional SELLING  (~{z.total_volume:.0f} contracts, {z.total_refills} refills)"
            (high_conf if z.max_confidence > 0.6 else watch).append(entry)

        # Cross-layer: spoofed + fragile
        for rec in persistent:
            if rec.overlapping_spoof and rec.dominant_state == "FRAGILE":
                tag = "support" if rec.side == Side.BID else "resistance"
                entry = f"  ⚠️  {rec.price:.2f}  FRAGILE {tag} + SPOOFED  (fragile {rec.fragile_pct:.0f}% of session)"
                high_conf.append(entry)
            elif rec.dominant_state == "FRAGILE" and abs(rec.price - close_p) <= 20:
                tag = "support" if rec.side == Side.BID else "resistance"
                entry = f"  ⚠️  {rec.price:.2f}  Fragile {tag}  (fragile {rec.fragile_pct:.0f}% of session)"
                watch.append(entry)

        # High-conf spoof zones
        for sz in bid_spoof + ask_spoof:
            if sz.max_confidence > 0.6 and sz.event_count >= 5:
                side_label = "BID" if sz.side == Side.BID else "ASK"
                entry = f"  🎭 {sz.price:.2f}  {sz.spoof_type.value} on {side_label}  (conf {int(sz.max_confidence*100)}%, × {sz.event_count})"
                high_conf.append(entry)

        lines.append("  HIGH CONFIDENCE:")
        if high_conf:
            for e in high_conf[:10]:
                lines.append(e)
        else:
            lines.append("  None.")
        lines.append("")
        lines.append("  WATCH:")
        if watch:
            for e in watch[:10]:
                lines.append(e)
        else:
            lines.append("  None.")
        lines.append("")

        # ── Authenticity Zones ──
        lines.append("─" * 63)
        lines.append("🔍 AUTHENTICITY — LOW ZONES (< 60%)")
        lines.append("─" * 63)

        low_zones = [
            az for az in self.auth_zones.values()
            if az.mean_auth < 0.60 and len(az.scores) >= 5
        ]
        low_zones.sort(key=lambda z: z.mean_auth)

        if not low_zones:
            lines.append("  All zones ≥ 60% authentic.")
        else:
            from datetime import datetime, timezone, timedelta
            for az in low_zones[:10]:
                win_start_ms = az.time_window * 30 * 60 * 1000
                t_str = _ms_to_et(win_start_ms)
                lines.append(
                    f"  📍 {az.price_bucket:.2f}  auth {az.mean_auth:.0%}  "
                    f"({len(az.scores)} snaps)  window ~{t_str} ET"
                )

        if self.authenticity_scores:
            avg_auth = sum(self.authenticity_scores) / len(self.authenticity_scores)
            low_auth = sum(1 for s in self.authenticity_scores if s < 0.4)
            lines.append("")
            lines.append(f"  Session avg: {avg_auth:.0%}  |  Low (<40%) snaps: {low_auth} ({low_auth/len(self.authenticity_scores)*100:.1f}%)")
        lines.append("")

        lines.append("═" * 63)
        lines.append("  Mark these levels on your chart before the session.")
        lines.append("═" * 63)

        return "\n".join(lines)


# ── Session Results (for multi-file comparison) ──────────────────────

@dataclass
class SessionResult:
    file_path: str
    cfg: InstrumentConfig
    spoof_zones: Dict[Tuple[float, Side, SpoofType], SpoofZone]
    fragility_records: Dict[Tuple[float, Side], FragilityRecord]
    iceberg_zones: Dict[Tuple[float, Side], IcebergZone]
    snapshot_count: int
    price_range: List[float]
    first_ts_ms: int
    last_ts_ms: int
    report_text: str


def _comparative_summary(results: List[SessionResult]) -> str:
    if len(results) < 2:
        return ""

    lines = []
    lines.append("")
    lines.append("═" * 63)
    lines.append("  COMPARATIVE SUMMARY")
    lines.append("═" * 63)
    lines.append("")

    # Count how many sessions each spoof price appeared in
    spoof_price_counts: Dict[float, int] = defaultdict(int)
    for r in results:
        seen_prices = set()
        for (price, side, stype), sz in r.spoof_zones.items():
            seen_prices.add(price)
        for p in seen_prices:
            spoof_price_counts[p] += 1

    persistent_spoof = sorted(
        [(p, c) for p, c in spoof_price_counts.items() if c >= 2],
        key=lambda x: x[1], reverse=True
    )

    lines.append("🎭 PERSISTENT SPOOF ZONES (2+ sessions):")
    if persistent_spoof:
        for price, count in persistent_spoof[:10]:
            lines.append(f"  📌 {price:.2f}  seen in {count}/{len(results)} sessions")
    else:
        lines.append("  None found.")
    lines.append("")

    # Chronic fragility: levels fragile in 2+ sessions
    frag_price_counts: Dict[float, int] = defaultdict(int)
    for r in results:
        threshold = max(
            int(r.snapshot_count * r.cfg.persistence_pct),
            r.cfg.persistence_min
        )
        for (price, side), rec in r.fragility_records.items():
            if rec.total_snapshots >= threshold and rec.dominant_state == "FRAGILE":
                frag_price_counts[price] += 1

    chronic_frag = sorted(
        [(p, c) for p, c in frag_price_counts.items() if c >= 2],
        key=lambda x: x[1], reverse=True
    )

    lines.append("🏗️  CHRONIC FRAGILITY (2+ sessions):")
    if chronic_frag:
        for price, count in chronic_frag[:10]:
            lines.append(f"  🔴 {price:.2f}  fragile in {count}/{len(results)} sessions")
    else:
        lines.append("  None found.")
    lines.append("")

    # Recurrent icebergs
    ice_price_counts: Dict[float, int] = defaultdict(int)
    for r in results:
        for (price, side), z in r.iceberg_zones.items():
            if z.total_volume >= r.cfg.iceberg_vol_threshold:
                ice_price_counts[price] += 1

    recurrent_ice = sorted(
        [(p, c) for p, c in ice_price_counts.items() if c >= 2],
        key=lambda x: x[1], reverse=True
    )

    lines.append("🧊 INSTITUTIONAL INTEREST LEVELS (icebergs 2+ sessions):")
    if recurrent_ice:
        for price, count in recurrent_ice[:10]:
            lines.append(f"  ★  {price:.2f}  iceberg in {count}/{len(results)} sessions")
    else:
        lines.append("  None found.")
    lines.append("")

    lines.append("═" * 63)
    return "\n".join(lines)


# ── Main runner ────────────────────────────────────────────────────

async def run_report(
    file_path: str,
    rth_only: bool = False,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    symbol_override: Optional[str] = None,
) -> SessionResult:
    cfg = _detect_instrument(file_path, symbol_override=symbol_override)
    feed = DatabentoReplayFeed(
        file_path=file_path,
        symbol=f"{cfg.symbol}.c.0",
        filter_symbol=cfg.symbol,
        snapshot_interval_ms=500,
        num_levels=10,
        event_window=500,
        trade_window=200,
        time_start=time_start,
        time_end=time_end,
    )

    print(f"Connecting to replay data: {os.path.basename(file_path)} ({cfg.symbol})")
    await feed.connect()
    print(f"Streaming mode. Running analysis...\n")

    builder = MorningReportBuilder(cfg)
    processed = 0
    skipped_rth = 0

    # ── RTH filter: determine target date ──
    # If rth_only and a date is inferrable from the filename or time_start,
    # lock to that single calendar date so multi-day files don't mix sessions.
    _rth_target_date = None
    if rth_only:
        # Try to extract date from filename (e.g. nq-mbo-2026-03-24.dbn.zst)
        import re as _re
        _m = _re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(file_path))
        if _m:
            from datetime import date as _date_cls
            _rth_target_date = _date_cls.fromisoformat(_m.group(1))
        elif time_start:
            _m2 = _re.search(r'(\d{4}-\d{2}-\d{2})', time_start)
            if _m2:
                from datetime import date as _date_cls
                _rth_target_date = _date_cls.fromisoformat(_m2.group(1))

    while True:
        snapshot = await feed.next()
        if snapshot is None:
            break

        if rth_only:
            from datetime import datetime as _dt, timezone as _tz
            _snap_dt = _dt.fromtimestamp(
                snapshot.timestamp_ms / 1000.0, tz=_tz.utc
            ).astimezone(_ET)
            _snap_time = _snap_dt.hour * 60 + _snap_dt.minute
            _rth_start = 9 * 60 + 30   # 9:30 AM ET
            _rth_end = 16 * 60          # 4:00 PM ET
            # Skip weekends
            if _snap_dt.weekday() >= 5:
                skipped_rth += 1
                continue
            # Skip outside RTH hours
            if _snap_time < _rth_start or _snap_time >= _rth_end:
                skipped_rth += 1
                continue
            # Skip wrong date (if target date is known)
            if _rth_target_date and _snap_dt.date() != _rth_target_date:
                skipped_rth += 1
                continue

        builder.process_snapshot(snapshot)
        processed += 1

        if processed % 1000 == 0:
            scanned = feed.records_scanned
            matched = feed.events_processed
            print(f"  {processed:,} snapshots | {matched:,} matched / {scanned:,} scanned", end="\r")

    await feed.disconnect()

    if rth_only and skipped_rth > 0:
        print(f"\n  RTH filter: {processed:,} RTH snapshots kept, {skipped_rth:,} non-RTH skipped")
        if _rth_target_date:
            print(f"  Target date: {_rth_target_date}")
    if processed == 0:
        print("\n⚠️  No snapshots matched the filter. Check data coverage and date range.")

    # Override the builder's date if we have an RTH target date
    rth_date_override = None
    if rth_only and _rth_target_date:
        rth_date_override = _rth_target_date.strftime("%A %B %d %Y")

    report = builder.generate_report(file_path, date_override=rth_date_override)
    print("\n")
    print(report)

    # Save report to reports/ directory
    _reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(_reports_dir, exist_ok=True)
    _report_basename = os.path.basename(file_path).replace(".dbn.zst", "-report.txt")
    report_path = os.path.join(_reports_dir, _report_basename)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    return SessionResult(
        file_path=file_path,
        cfg=cfg,
        spoof_zones=builder.spoof_zones,
        fragility_records=builder.fragility_records,
        iceberg_zones=builder.iceberg_zones,
        snapshot_count=builder.snapshot_count,
        price_range=builder.price_range,
        first_ts_ms=builder.first_ts_ms,
        last_ts_ms=builder.last_ts_ms,
        report_text=report,
    )


async def run_multi(
    file_paths: List[str],
    rth_only: bool = False,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    symbol_override: Optional[str] = None,
) -> None:
    results: List[SessionResult] = []
    for fp in file_paths:
        result = await run_report(fp, rth_only=rth_only, time_start=time_start,
                                   time_end=time_end, symbol_override=symbol_override)
        results.append(result)

    if len(results) >= 2:
        comp = _comparative_summary(results)
        print(comp)
        # Append comparative summary to last report file
        _reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
        _last_basename = os.path.basename(file_paths[-1]).replace(".dbn.zst", "-report.txt")
        last_report_path = os.path.join(_reports_dir, _last_basename)
        with open(last_report_path, "a") as f:
            f.write(comp)
        print(f"Comparative summary appended to: {last_report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Morning Prep Report — L3 Forensics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("file", nargs="?", help="Path to .dbn.zst file")
    group.add_argument("--files", nargs="+", help="Multiple .dbn.zst files for comparative analysis")
    parser.add_argument("--symbol", help="Instrument to extract (ES, NQ, CL, GC, SI). Required for full-exchange files.")
    parser.add_argument("--rth-only", action="store_true", help="Only analyze RTH (9:30-4:00 ET)")
    parser.add_argument("--date", help="Extract single date from multi-day file (YYYY-MM-DD)")
    parser.add_argument("--dates", nargs="+", help="Extract multiple dates for comparison (YYYY-MM-DD ...)")
    parser.add_argument("--start", dest="time_start", help="Start time filter (YYYY-MM-DDTHH:MM or YYYY-MM-DD)")
    parser.add_argument("--end", dest="time_end", help="End time filter (YYYY-MM-DDTHH:MM or YYYY-MM-DD)")
    args = parser.parse_args()

    sym = args.symbol

    if args.dates and args.file:
        async def _run_dates():
            results = []
            for d in args.dates:
                r = await run_report(args.file, rth_only=args.rth_only,
                                     time_start=d, time_end=d, symbol_override=sym)
                results.append(r)
            if len(results) >= 2:
                comp = _comparative_summary(results)
                print(comp)
        asyncio.run(_run_dates())
    elif args.date and args.file:
        asyncio.run(run_report(args.file, rth_only=args.rth_only,
                               time_start=args.date, time_end=args.date, symbol_override=sym))
    elif args.files:
        asyncio.run(run_multi(args.files, rth_only=args.rth_only,
                              time_start=args.time_start, time_end=args.time_end, symbol_override=sym))
    else:
        asyncio.run(run_report(args.file, rth_only=args.rth_only,
                               time_start=args.time_start, time_end=args.time_end, symbol_override=sym))
