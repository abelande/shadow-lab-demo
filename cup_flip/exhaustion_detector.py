"""Streak exhaustion detection via FLUX + STARK.

Detects when a directional streak is losing momentum by combining
two signals from FluxStarkTracker:

1. FLUX > threshold → a structural shift has occurred (the streak was real)
2. STARK < 0 → the shift rate is decelerating (the streak is exhausting)

Both conditions must be true simultaneously for an exhaustion signal.
Confidence scales with flux magnitude × |stark| magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .signals.flux_stark import FluxStarkTracker


@dataclass(frozen=True)
class ExhaustionSignal:
    """Emitted when a directional streak shows signs of exhaustion."""
    confidence: float           # 0-1
    bars_since_peak_flux: int   # how long ago flux peaked (recency of shift)
    flux: float                 # current flux level
    stark: float                # current d(flux)/dt (negative = decelerating)


class ExhaustionDetector:
    """Monitors FLUX + STARK to detect streak exhaustion.

    Wire into TapeReader. Call update(mid_price) every snapshot.
    When the detector fires, the tape_reader should set
    game_state.streak_exhaustion = signal.confidence.
    """

    def __init__(
        self,
        flux_threshold: float = 0.3,
        flux_window: int = 30,
        baseline_window: int = 120,
    ) -> None:
        self._flux_threshold = flux_threshold
        self._tracker = FluxStarkTracker(
            flux_window=flux_window,
            baseline_window=baseline_window,
        )
        self._peak_flux: float = 0.0
        self._bars_since_peak: int = 0

    def update(self, mid_price: float) -> Optional[ExhaustionSignal]:
        """Feed a mid price. Returns ExhaustionSignal if exhaustion detected."""
        flux, stark = self._tracker.update(mid_price)

        if flux > self._peak_flux:
            self._peak_flux = flux
            self._bars_since_peak = 0
        else:
            self._bars_since_peak += 1

        # Exhaustion: structural shift happened AND is now decelerating
        if flux > self._flux_threshold and stark < 0:
            confidence = min(1.0, flux) * min(1.0, abs(stark) * 5.0)
            confidence = min(1.0, confidence)
            return ExhaustionSignal(
                confidence=confidence,
                bars_since_peak_flux=self._bars_since_peak,
                flux=flux,
                stark=stark,
            )
        return None

    def reset(self) -> None:
        self._tracker.reset()
        self._peak_flux = 0.0
        self._bars_since_peak = 0
