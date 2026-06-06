"""
p6lab.live
==========

Live-trading integration — ties together ``DatabentoLiveFeed``, the
``CorrelationEngine``, ``MatchBroker``, and the Tier-1 renderers.

Exposes two classes:

- ``FeatureAccumulator`` — maintains rolling L1History / L2History /
  FragilityIndex state, converts each ``OrderBookSnapshot`` from the
  feed into the ``(l2_window, l1_window)`` DataFrames the engine's
  ``match()`` expects.

- ``LiveRunner`` — orchestrator. Drives ``feed.next()`` in a loop,
  feeds the accumulator, calls ``engine.match()``, and relies on the
  engine's ``broker=`` kwarg to fan out matches to renderers.
"""
from p6lab.live.feature_accumulator import FeatureAccumulator
from p6lab.live.runner import LiveRunner

__all__ = ["FeatureAccumulator", "LiveRunner"]
