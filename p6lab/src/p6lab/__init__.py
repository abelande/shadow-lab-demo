"""
p6lab — Shared research library for P6 Research Lab.

Spec: p6-notebook-lab-spec.md §0 — Purpose & Scope
Revision: 2 (split batch notebooks from interactive UI)

Architecture:
  - p6lab.ingestion   — data prep, windowing, normalization (§3)
  - p6lab.features    — L1/L2 feature sets, fragility index, VPIN (§4)
  - p6lab.patterns    — library schema, miner, labeler, matcher (§5)
  - p6lab.execution   — queue tracker, fill simulator, cost model (§6)
  - p6lab.correlation — engine, scorer, regime conditioner (§7)
  - p6lab.validation  — CPCV, augmentation, information-gain gate (§8)

Guiding principle (spec §0):
  L3 is the omniscient teacher; L1/L2 are the deployable students.
  Every component either labels with L3 truth, trains on L1/L2 features,
  or renders the gap for review.
  Ref: OB-reference.md §5 knowledge distillation (L698-973)
"""

__version__ = "0.1.0"
__spec_revision__ = 2
