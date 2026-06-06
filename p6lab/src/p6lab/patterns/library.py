"""
Pattern Library — YAML Registry + Pydantic Schema
Spec §5.1 | OB-reference.md:805-809

Versioned YAML at artifacts/p6lab/pattern_library/library.yaml.
Consumed by notebooks 04, 06, pattern review queue UI (§10.3),
and the live correlation engine (§7.1, §10.4).

Minimum sample size rule: 200 occurrences minimum.
Below threshold → status: candidate, no outcome distribution.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PatternStatus(str, Enum):
    CANDIDATE = "candidate"
    MINED_APPROVED = "mined_approved"
    ACTIVE = "active"
    RETIRED = "retired"
    REJECTED = "rejected"


# Valid status transitions. Forms a DAG: any node may go to REJECTED,
# and the linear promotion path is CANDIDATE → MINED_APPROVED → ACTIVE → RETIRED.
_ALLOWED_TRANSITIONS: dict[PatternStatus, set[PatternStatus]] = {
    PatternStatus.CANDIDATE: {PatternStatus.MINED_APPROVED, PatternStatus.REJECTED},
    PatternStatus.MINED_APPROVED: {PatternStatus.ACTIVE, PatternStatus.REJECTED, PatternStatus.RETIRED},
    PatternStatus.ACTIVE: {PatternStatus.RETIRED, PatternStatus.REJECTED},
    PatternStatus.RETIRED: set(),   # terminal
    PatternStatus.REJECTED: set(),  # terminal
}


class OutcomeDistribution(BaseModel):
    mean_atr: float
    std: float
    hit_rate: float
    n: int


class ConfidenceTierCutoffs(BaseModel):
    A: float = 0.85
    B: float = 0.72
    C: float = 0.60


class PatternDefinition(BaseModel):
    """Full pattern definition per §5.1 YAML schema."""
    name: str
    l3_signature: str
    l2_manifestation: str
    l1_footprint: str
    outcome_distribution: dict[str, OutcomeDistribution] = Field(default_factory=dict)
    min_sample_size: int = 200
    regime_specific: bool = True
    instruments: list[str] = Field(default_factory=list)
    confidence_tier_cutoffs: ConfidenceTierCutoffs = Field(default_factory=ConfidenceTierCutoffs)
    status: PatternStatus = PatternStatus.CANDIDATE
    validation_hash: str = ""

    def compute_hash(self) -> str:
        """SHA-256 of defining features (signature + manifestation + footprint)."""
        content = f"{self.l3_signature}|{self.l2_manifestation}|{self.l1_footprint}"
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


class PatternLibraryFile(BaseModel):
    """Top-level library.yaml schema."""
    version: int = 1
    generated: str = ""
    patterns: dict[str, PatternDefinition] = Field(default_factory=dict)


class PatternLibrary:
    """
    In-memory representation of the pattern library.

    Concurrency: writes acquire a filelock on `<path>.lock` and use
    write-to-temp + atomic-rename (§15.5). Safe for parallel miners.
    """

    def __init__(self, library_path: Path | str):
        self.library_path = Path(library_path)
        self._data: PatternLibraryFile | None = None

    @property
    def _lock_path(self) -> Path:
        return self.library_path.with_suffix(self.library_path.suffix + ".lock")

    def load(self) -> PatternLibraryFile:
        """Load library.yaml into memory. Returns an empty library if missing."""
        if not self.library_path.exists():
            self._data = PatternLibraryFile()
            return self._data
        with self._lock_path.parent.joinpath(self._lock_path.name).open("a"):
            pass  # ensure lock file exists
        with FileLock(str(self._lock_path)):
            raw = yaml.safe_load(self.library_path.read_text()) or {}
        self._data = PatternLibraryFile.model_validate(raw)
        return self._data

    def save(self) -> None:
        """Atomic write: file-lock + write to temp + rename. Auto-bumps version."""
        if self._data is None:
            raise RuntimeError("PatternLibrary.save() called before load() / add_pattern()")
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        # refresh metadata
        self._data.version += 1
        self._data.generated = datetime.now(timezone.utc).isoformat()
        payload = self._data.model_dump(mode="json")
        with FileLock(str(self._lock_path)):
            # write-to-temp-and-rename — atomic on POSIX
            fd, tmp_path = tempfile.mkstemp(
                prefix=self.library_path.name + ".",
                suffix=".tmp",
                dir=str(self.library_path.parent),
            )
            try:
                with os.fdopen(fd, "w") as f:
                    yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False)
                os.replace(tmp_path, self.library_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        logger.info("PatternLibrary saved: %s (v%d, %d patterns)",
                    self.library_path, self._data.version, len(self._data.patterns))

    def get_active_patterns(self) -> dict[str, PatternDefinition]:
        """Return patterns with status in {active, mined_approved}."""
        if self._data is None:
            self.load()
        assert self._data is not None
        active = {PatternStatus.ACTIVE, PatternStatus.MINED_APPROVED}
        return {k: p for k, p in self._data.patterns.items() if p.status in active}

    def promote(self, pattern_name: str, new_status: PatternStatus) -> None:
        """Transition a pattern; validates against _ALLOWED_TRANSITIONS."""
        if self._data is None:
            self.load()
        assert self._data is not None
        if pattern_name not in self._data.patterns:
            raise KeyError(f"Pattern not found: {pattern_name}")
        pattern = self._data.patterns[pattern_name]
        current = pattern.status
        if new_status not in _ALLOWED_TRANSITIONS[current]:
            raise ValueError(
                f"Invalid transition for {pattern_name}: {current.value} → {new_status.value}. "
                f"Allowed: {[s.value for s in _ALLOWED_TRANSITIONS[current]]}"
            )
        pattern.status = new_status
        logger.info("Promoted %s: %s → %s", pattern_name, current.value, new_status.value)

    def add_pattern(self, name: str, pattern: PatternDefinition) -> None:
        """Add a new pattern. Computes validation_hash automatically."""
        if self._data is None:
            self.load()
        assert self._data is not None
        if name in self._data.patterns:
            raise ValueError(f"Pattern already exists: {name}")
        if not pattern.validation_hash:
            pattern.validation_hash = pattern.compute_hash()
        self._data.patterns[name] = pattern
