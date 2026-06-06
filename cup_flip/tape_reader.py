"""Tape Reader: converts trade/events stream into Cup Flip game state updates.

Enriched version: integrates OFI pressure blend, energy-ratio acceleration,
Kalman velocity for stop-run confirmation, Markov P(up|state) for state
machine confidence, entropy-based confidence gating, and FLUX+STARK
exhaustion detection.

All enrichment signals are optional — backward compatible when called
without snapshot/ofi arguments.
"""
from __future__ import annotations
from typing import List, Optional
from ..models import Order, OrderAction, GameState, Side, OrderBookSnapshot
from .streak_detector import StreakDetector
from .stall_detector import StallDetector
from .stop_run_detector import StopRunDetector
from .pressure_scorer import PressureScorer
from .state_machine import CupFlipMachine
from .exhaustion_detector import ExhaustionDetector
from .signals import EnergyRatio, KalmanVelocity, EntropyGate, MarkovStateTracker


class TapeReader:
    def __init__(self, stop_run_levels: int = 5):
        """
        Args:
            stop_run_levels: Minimum levels cleared to trigger a stop-run signal.
                             Default 5 (NQ needs a higher bar than ES's 3).
        """
        self.streak_detector = StreakDetector(min_streak_length=3)
        self.stall_detector = StallDetector(min_failed_attempts=3, window_ms=1500)
        self.stop_run_detector = StopRunDetector(
            levels_threshold=stop_run_levels, velocity_threshold=2.0
        )
        self.pressure_scorer = PressureScorer()
        self.machine = CupFlipMachine()
        self.state: GameState | None = None

        # Enrichment signal trackers
        self.energy = EnergyRatio(short_window=5, long_window=20)
        self.kalman = KalmanVelocity()
        self.entropy_gate = EntropyGate(window=30)
        self.markov = MarkovStateTracker()
        self.exhaustion = ExhaustionDetector()

    def update(
        self,
        events: List[Order],
        timestamp_ms: int,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
        snapshot: Optional[OrderBookSnapshot] = None,
        ofi: float = 0.0,
    ) -> GameState:
        """Process a batch of recent events and return the current game state.

        Args:
            events: Recent order events from the snapshot window.
            timestamp_ms: Current snapshot timestamp.
            best_bid: Current best bid price (used for aggressive-add detection).
            best_ask: Current best ask price (used for aggressive-add detection).
            snapshot: Full OrderBookSnapshot for enrichment signals. Optional.
            ofi: Order Flow Imbalance value from OFITracker. Optional (0 = unused).
        """
        current_streak = self.streak_detector.current_streak

        for e in events:
            if e.action == OrderAction.FILL:
                self.streak_detector.process_fill(e)
                current_streak = self.streak_detector.current_streak

        # Core pressure with optional OFI blend
        pressure = self.pressure_scorer.score(events, ofi=ofi)

        # Enrichment signals (neutral defaults when snapshot unavailable)
        pressure_accel = self.energy.update(abs(pressure))
        entropy = self.entropy_gate.update(pressure)

        markov_p_up = 0.5
        kalman_vel: Optional[float] = None
        kalman_std: Optional[float] = None
        exhaustion_conf = 0.0

        if snapshot is not None:
            markov_p_up = self.markov.update(snapshot)
            mid = snapshot.mid_price
            if mid is not None:
                _, kalman_vel = self.kalman.update(mid)
                kalman_std = self.kalman.velocity_std
                ex = self.exhaustion.update(mid)
                if ex is not None:
                    exhaustion_conf = ex.confidence

        # Infer best_bid/ask from snapshot if not explicitly provided
        if best_bid is None and snapshot is not None:
            best_bid = snapshot.best_bid
        if best_ask is None and snapshot is not None:
            best_ask = snapshot.best_ask

        push_side = Side.BID if pressure >= 0 else Side.ASK
        stall = self.stall_detector.detect(
            events, push_side=push_side, best_bid=best_bid, best_ask=best_ask
        )
        stop_run = self.stop_run_detector.detect(
            current_streak,
            kalman_velocity=kalman_vel,
            kalman_velocity_std=kalman_std,
        )

        self.state = self.machine.transition(
            prev=self.state,
            pressure=pressure,
            streak=current_streak,
            stall=stall,
            stop_run=stop_run,
            timestamp_ms=timestamp_ms,
            markov_p_up=markov_p_up,
        )

        # Apply enrichment fields
        self.state.pressure_acceleration = pressure_accel
        self.state.streak_exhaustion = exhaustion_conf

        # Entropy-based confidence decay: high entropy = choppy = less reliable
        if self.state.state_confidence > 0 and entropy > 0.3:
            self.state.state_confidence *= (1.0 - 0.3 * entropy)
            self.state.state_confidence = max(0.0, self.state.state_confidence)

        return self.state
