"""
p6lab.execution — Execution Simulation (Spec §6)

Shared between batch notebook 05 and interactive UI tool (§10.2).
Same library, two consumers.

Submodules:
  queue_tracker.py  — per-order queue position tracking (§6.1)
  fill_simulator.py — passive order fill model (§6.2)
  cost_model.py     — realistic cost decomposition (§6.3, Phase 5)

Import from the specific submodule to keep the package lazy:
    from p6lab.execution.queue_tracker import QueueTracker
    from p6lab.execution.fill_simulator import FillSimulator, FillOutcome
"""
