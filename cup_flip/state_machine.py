"""Cup Flip state machine with 6 states and transition logic."""
from __future__ import annotations
from ..models import GameState, CupFlipState, Side
from .streak_detector import Streak
from .stall_detector import StallSignal
from .stop_run_detector import StopRunSignal


class CupFlipMachine:
    def __init__(self, pressure_enter: float = 0.40, pressure_stall: float = 0.10):
        self.pressure_enter = pressure_enter
        self.pressure_stall = pressure_stall

    def _streak_to_state(self, streak: Streak) -> CupFlipState:
        # Streak consuming asks => bullish
        return CupFlipState.BULL_STREAK if streak.side == Side.ASK else CupFlipState.BEAR_STREAK

    def transition(
        self,
        prev: GameState | None,
        pressure: float,
        streak: Streak | None,
        stall: StallSignal | None,
        stop_run: StopRunSignal | None,
        timestamp_ms: int,
        markov_p_up: float = 0.5,
    ) -> GameState:
        if prev is None:
            prev = GameState(state=CupFlipState.BALANCED, timestamp_ms=timestamp_ms)

        state = prev.state

        # Highest priority
        if stop_run is not None:
            state = CupFlipState.STOP_RUN
        elif stall is not None:
            state = CupFlipState.BULL_STALL if stall.side == Side.BID else CupFlipState.BEAR_STALL
        elif streak is not None and streak.length >= 3:
            state = self._streak_to_state(streak)
        else:
            if pressure >= self.pressure_enter:
                state = CupFlipState.BULL_STREAK
            elif pressure <= -self.pressure_enter:
                state = CupFlipState.BEAR_STREAK
            elif abs(pressure) <= self.pressure_stall:
                state = CupFlipState.BALANCED

        # Compute state confidence from threshold distance + markov agreement
        is_bullish = state in (CupFlipState.BULL_STREAK, CupFlipState.STOP_RUN)
        is_bearish = state in (CupFlipState.BEAR_STREAK,)

        if is_bullish or is_bearish:
            threshold_conf = min(1.0, abs(pressure) / max(self.pressure_enter, 0.01))
            markov_agrees = (is_bullish and markov_p_up > 0.55) or (is_bearish and markov_p_up < 0.45)
            markov_score = markov_p_up if is_bullish else (1.0 - markov_p_up)
            state_confidence = 0.7 * threshold_conf + 0.3 * markov_score
            if not markov_agrees:
                state_confidence *= 0.7  # disagreement penalty
        else:
            state_confidence = 0.0

        return GameState(
            state=state,
            streak_length=streak.length if streak else 0,
            streak_velocity=streak.velocity if streak else 0.0,
            streak_depth=streak.depth if streak else 0,
            pressure=pressure,
            stall_count=stall.failed_attempts if stall else 0,
            stop_run_side=stop_run.aggressor_side if stop_run else None,
            timestamp_ms=timestamp_ms,
            state_confidence=min(1.0, max(0.0, state_confidence)),
        )
