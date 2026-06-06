"""Wave 8.5-K — unit tests for PercentileTierFilter."""
import numpy as np
import pytest

from p6lab.live.tier_filter import PercentileTierFilter, TierFilterConfig


def test_warmup_returns_none() -> None:
    """Filter must return None until warmup_samples observations."""
    filt = PercentileTierFilter(TierFilterConfig(warmup_samples=100))
    for i in range(99):
        assert filt.observe(0.99) is None, f"emitted tier at i={i}"
    assert filt.is_warm is False


def test_top_percentile_classified_strictest() -> None:
    """A prediction at the 99.5th percentile of history → A_strict tier."""
    filt = PercentileTierFilter(TierFilterConfig(warmup_samples=1000))
    rng = np.random.default_rng(42)
    # Warmup with uniform [0, 1]
    for _ in range(1000):
        filt.observe(float(rng.uniform()))
    # A high-percentile observation
    tier = filt.observe(0.999)
    assert tier == "A_strict", f"expected A_strict, got {tier}"


def test_low_score_returns_none() -> None:
    """A prediction below all thresholds returns None."""
    filt = PercentileTierFilter(TierFilterConfig(warmup_samples=100))
    for _ in range(100):
        filt.observe(0.5)
    tier = filt.observe(0.4)
    assert tier is None


def test_regime_adaptive() -> None:
    """When score distribution shifts up, thresholds shift up."""
    filt = PercentileTierFilter(TierFilterConfig(warmup_samples=100, history_size=200))
    # Phase 1: low-score regime
    for _ in range(100):
        filt.observe(0.3)
    low_thresholds = filt.current_thresholds()
    # Phase 2: high-score regime
    for _ in range(100):
        filt.observe(0.8)
    high_thresholds = filt.current_thresholds()
    assert all(high_thresholds[k] > low_thresholds[k] for k in low_thresholds), (
        "thresholds did not shift up after regime change"
    )


def test_strictest_tier_first() -> None:
    """Prediction qualifying for multiple tiers gets assigned the strictest."""
    cfg = TierFilterConfig(
        warmup_samples=100,
        tier_percentiles={"A": 0.99, "B": 0.90},
    )
    filt = PercentileTierFilter(cfg)
    for _ in range(100):
        filt.observe(0.5)  # uniform-ish
    # Score that's far above any percentile of 0.5 history
    tier = filt.observe(0.99)
    assert tier == "A", "should pick strictest qualifying tier"
