"""End-to-end pipeline: OrderBookSnapshot -> all layers -> DepthIndicatorFrame."""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Optional, List

from .models import OrderBookSnapshot, DepthIndicatorFrame, LevelState
from .level_tracker import LevelTracker
from .staircase_analyzer.fragility_scorer import FragilityScorer
from .cup_flip.tape_reader import TapeReader
from .spectral_force.volume_delta_series import VolumeDeltaSeries
from .spectral_force.fft_decomposer import FFTDecomposer
from .spectral_force.band_splitter import BandSplitter
from .spectral_force.energy_per_band import EnergyPerBand
from .spectral_force.force_aggregator import ForceAggregator
from .spectral_force.institutional_score import InstitutionalScore
from .spoof_detection.pull_before_touch import PullBeforeTouchDetector
from .spoof_detection.layering_detector import LayeringDetector
from .spoof_detection.iceberg_inference import IcebergInference
from .spoof_detection.phantom_wall import PhantomWallDetector
from .spoof_detection.authenticity_scorer import AuthenticityScorer
from .regime_context.regime_classifier import RegimeClassifier
from .regime_context.regime_weights import get_regime_weights
from .regime_context.abstain_policy import AbstainPolicy
from .depth_indicator.chart_overlay import ChartOverlay
from .aggregator import SignalAggregator
from .ofi import OFITracker

logger = logging.getLogger(__name__)


class OrderBookMetaPipeline:
    def __init__(
        self,
        *,
        correlation_engine: Any = None,
        feature_accumulator: Any = None,
        match_interval_ms: int = 1_000,
    ):
        # Wave 5 Phase 5A — optional L6 correlation engine. When ``correlation_engine``
        # and ``feature_accumulator`` are both provided, run() feeds each snapshot
        # to the accumulator and calls ``engine.match()`` at ``match_interval_ms``
        # cadence. Resulting ``PatternMatch`` list is attached to the frame so the
        # existing WebSocket broadcaster forwards it to the browser. If either is
        # None (default), correlation is skipped — preserves backward compat.
        self._correlation_engine = correlation_engine
        self._feature_accumulator = feature_accumulator
        self._match_interval_ms = int(match_interval_ms)
        self._last_match_ts_ms: int = 0
        # Wave 8.5-A: observability counters on the correlation swallow sites.
        # Plain dict[str, int] (not collections.Counter) so _serialize_value in
        # server/websocket.py handles it natively as JSON.
        self._correlation_stats: dict[str, int] = {
            "ingest_errors": 0,
            "match_errors": 0,
        }
        self.l1 = FragilityScorer()
        self.tape = TapeReader()

        self.vds = VolumeDeltaSeries()
        self.fft = FFTDecomposer()
        self.split = BandSplitter()
        self.epb = EnergyPerBand()
        self.force = ForceAggregator()
        self.inst_score = InstitutionalScore()

        self.pull = PullBeforeTouchDetector()
        self.layering = LayeringDetector()
        self.iceberg = IcebergInference()
        self.phantom = PhantomWallDetector()
        self.auth_scorer = AuthenticityScorer()

        self.regime_classifier = RegimeClassifier()
        self.abstain = AbstainPolicy()
        self.agg = SignalAggregator()
        self.overlay = ChartOverlay()
        self.level_tracker = LevelTracker()
        self.ofi_tracker = OFITracker()

    def run(
        self,
        snapshot: OrderBookSnapshot,
        combined_regime_output: dict | None = None,
    ) -> DepthIndicatorFrame:
        # Layer 1
        staircase = self.l1.build_profile(snapshot)

        # OFI (between L1 and L2 — feeds enriched pressure to cup flip)
        ofi, vpin = self.ofi_tracker.update(snapshot)

        # Layer 2
        game_state = self.tape.update(
            snapshot.recent_events or snapshot.recent_trades,
            snapshot.timestamp_ms,
            snapshot=snapshot,
            ofi=ofi,
        )

        # Layer 3
        series = self.vds.build(snapshot.recent_trades)
        freqs, coeffs = self.fft.decompose(series)
        bands = self.split.split(freqs)
        band_energy = self.epb.compute(bands, coeffs, series)
        force_vector = self.force.aggregate(band_energy, timestamp_ms=snapshot.timestamp_ms)
        force_vector.institutional_score = self.inst_score.score(force_vector)

        # Layer 4
        spoof_events = []
        spoof_events += self.pull.detect(snapshot.recent_events, snapshot.best_bid, snapshot.best_ask)
        spoof_events += self.layering.detect(snapshot.recent_events)
        spoof_events += self.iceberg.detect(snapshot.recent_events)
        spoof_events += self.phantom.detect(snapshot.recent_events, snapshot.mid_price)
        authenticity = self.auth_scorer.score(spoof_events, snapshot.timestamp_ms)

        # Layer 5
        regime = self.regime_classifier.classify(combined_regime_output)
        regime_weights = get_regime_weights(regime)

        # Aggregation
        signal = self.agg.aggregate(
            staircase=staircase,
            game_state=game_state,
            force_vector=force_vector,
            authenticity=authenticity,
            regime_weights=regime_weights,
            timestamp_ms=snapshot.timestamp_ms,
        )
        signal.abstain = signal.abstain or self.abstain.should_abstain(
            regime_weights=regime_weights,
            confidence=signal.confidence,
            authenticity_score=authenticity.authenticity_score,
            pressure_abs=abs(game_state.pressure),
        )

        # Update LevelTracker and get level states
        spoof_events_for_tracker = authenticity.spoof_events if authenticity else []
        auth_score = authenticity.authenticity_score if authenticity else 1.0
        level_states: List[LevelState] = self.level_tracker.update(
            snapshot, spoof_events=spoof_events_for_tracker, authenticity_score=auth_score
        )

        frame = self.overlay.render(
            snapshot=snapshot,
            staircase=staircase,
            game_state=game_state,
            force_vector=force_vector,
            authenticity=authenticity,
            regime_weights=regime_weights,
            signal=signal,
            level_states=level_states,
        )
        frame.level_states = level_states
        frame.ofi = ofi
        frame.vpin = vpin

        # Wave 5 Phase 5A — L6 correlation matching.
        frame.correlation_matches = self._maybe_correlate(snapshot)

        return frame

    def _maybe_correlate(self, snapshot: OrderBookSnapshot) -> List[dict]:
        """Feed the accumulator and, at cadence, call engine.match().

        Returns serialized PatternMatch dicts (empty when the engine isn't
        wired or the window is mid-cadence). Failures degrade silently —
        correlation is additive; a bad match cycle must never corrupt the
        upstream frame.
        """
        engine = self._correlation_engine
        accumulator = self._feature_accumulator
        if engine is None or accumulator is None:
            return []

        try:
            accumulator.ingest(snapshot)
        except Exception:
            # Wave 8.5-A: increment counter inside except block so it's clear
            # the increment is exceptional, not routine; traceback line refs stay intact.
            self._correlation_stats["ingest_errors"] += 1
            logger.exception("correlation accumulator.ingest raised; skipping frame")
            return []

        ts_ms = int(getattr(snapshot, "timestamp_ms", 0) or 0)
        if ts_ms - self._last_match_ts_ms < self._match_interval_ms:
            return []

        windows = accumulator.window()
        if windows is None:
            return []
        l2_window, l1_window = windows
        if len(l2_window) < 5:
            return []

        context = self._build_match_context(snapshot)

        try:
            matches = engine.match(
                l2_window=l2_window,
                l1_window=l1_window,
                context=context,
            )
        except Exception:
            # Wave 8.5-A: counter increment; see _correlation_stats docstring.
            self._correlation_stats["match_errors"] += 1
            logger.exception("correlation engine.match raised; skipping")
            return []

        self._last_match_ts_ms = ts_ms
        return [self._serialize_match(m) for m in matches]

    def _build_match_context(self, snapshot: OrderBookSnapshot):
        """Construct a MatchContext for the current snapshot. Defaults are
        conservative (calm FI, neutral regime) — Wave 5 Phase 5C+ replaces
        these with live VIX / volume signals."""
        from p6lab.patterns.template_matcher import MatchContext  # local import — optional dep

        ts_ms = int(getattr(snapshot, "timestamp_ms", 0) or 0)
        tod_minutes = int((ts_ms // 60_000) % (60 * 24))
        symbol = getattr(snapshot, "symbol", "") or ""
        instrument = symbol.split(".")[0].upper() if symbol else ""
        return MatchContext(
            time_of_day_minutes=tod_minutes,
            vix_level=18.0,
            vix_regime="normal",
            relative_volume=1.0,
            instrument=instrument,
        )

    @property
    def correlation_stats(self) -> dict[str, int]:
        """Wave 8.5-A: shallow copy so callers can't mutate the live counter.

        Read-only snapshot of exception counts for the correlation path.
        Surfaced via engine_runner.get_status() so the operator can
        distinguish 'zero matches because warming up' from 'zero matches
        because every engine.match() call is crashing'."""
        return dict(self._correlation_stats)

    @staticmethod
    def _serialize_match(m) -> dict:
        """PatternMatch dataclass → plain dict for JSON serialization."""
        try:
            return asdict(m)
        except Exception:
            return {
                "pattern_id": getattr(m, "pattern_id", "?"),
                "ensemble_score": float(getattr(m, "ensemble_score", 0.0) or 0.0),
                "confidence_tier": getattr(m, "confidence_tier", "?"),
                "expected_direction": getattr(m, "expected_direction", "neutral"),
                "expected_move_atr": float(getattr(m, "expected_move_atr", 0.0) or 0.0),
                "instrument": getattr(m, "instrument", ""),
                "regime": getattr(m, "regime", ""),
            }
