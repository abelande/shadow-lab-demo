"""
p6lab.correlation
=================
Correlation Engine — §7 of the P6 Lab Design Spec.

The knowledge-distillation bridge: matches live L2/L1 state to L3 pattern
labels.  L3 is the omniscient teacher; L1/L2 are the deployable students.

Sub-modules
-----------
engine.py              §7.1 — runtime matcher (live L2 → pattern matches)
scorer.py              §7.2 — ensemble score + confidence tier assignment
regime_conditioner.py  §7.3 — VIX-tagged, per-instrument template selection

Import from submodules directly to keep this package lazy.
"""
