"""
p6lab.validation
================
Validation tooling — §8 of the P6 Lab Design Spec.

Sub-modules
-----------
cpcv.py              §8.1 — Combinatorial Purged Cross-Validation + 14d embargo
augmentation.py      §8.2 — synthetic augmentation transforms
information_gain.py  §8.3 — must-beat-baseline decision gate
"""

from p6lab.validation.cpcv import CascadeAwareCPCV, CPCVFold
from p6lab.validation.augmentation import AugmentationEngine, AugmentedSample
from p6lab.validation.information_gain import DecisionReport, must_beat_baseline

__all__ = [
    "CascadeAwareCPCV",
    "CPCVFold",
    "AugmentationEngine",
    "AugmentedSample",
    "DecisionReport",
    "must_beat_baseline",
]
