"""
p6lab.risk — Pre-trade risk + intraday position tracking.

Keeps the domain logic that decides whether a signal is allowed to turn
into an order separate from the router that actually submits it. The
router calls into this module on every ``submit_from_match``; the
outcome tracker calls into ``on_exit`` once a trade's horizon closes.

Submodules:
  position_manager.py  — one-entry-per-pattern throttle, instrument
                         exposure cap, daily-loss circuit breaker.
"""
