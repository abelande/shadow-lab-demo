"""
p6lab.features — L1/L2 feature sets, Fragility Index, and VPIN.

Spec: p6-notebook-lab-spec.md §4 Shared Infrastructure — p6lab.features
  §4.1 l1_features.py    — 16-feature L1 set (OB-reference L830-847)
  §4.2 l2_features.py    — 12-feature L2 set (OB-reference L432-451)
  §4.3 fragility_index.py — 6 sub-indices + 2 composites (OB-reference L1552-1696)
  §4.4 vpin.py           — Volume-synchronized PIN (OB-reference L1625-1644)

Note: Submodules expose their public surface directly. Import from the
specific submodule (e.g. ``from p6lab.features.l1_features import ...``)
rather than re-exporting here — this keeps the package lazy and avoids
forcing unused dependencies (scipy, hdbscan, etc.) when only a subset
is needed.
"""