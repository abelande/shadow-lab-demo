"""
pattern_api.py — §11.3 Pattern Library + Candidate Review API

Endpoints:
- GET  /api/patterns/library
- GET  /api/patterns/candidates
- GET  /api/patterns/candidate/{id}
- POST /api/patterns/candidate/{id}/accept
- POST /api/patterns/candidate/{id}/reject

Implements human-in-the-loop gate between Notebook 04 (mining) and
Notebook 06 (correlation training). Accepted candidates are promoted to
library.yaml with status='mined_approved'. Rejected candidates recorded
in mined_decisions.parquet.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import json
import os
import tempfile

import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

router = APIRouter(tags=["patterns"])

ARTIFACTS_ROOT = Path("workspace/myquantlab/artifacts/p6lab")
PATTERN_LIB_DIR = ARTIFACTS_ROOT / "pattern_library"
LIBRARY_PATH = PATTERN_LIB_DIR / "library.yaml"
CANDIDATES_DIR = PATTERN_LIB_DIR / "mined_candidates"
DECISIONS_PATH = CANDIDATES_DIR / "mined_decisions.parquet"
LOCK_PATH = PATTERN_LIB_DIR / ".library.lock"


# ── Schemas ──────────────────────────────────────────────────────────

class CandidateDecisionRequest(BaseModel):
    reviewer: str = Field(default="human")
    decision_reason: str = Field(default="")
    confidence: float | None = Field(default=None, ge=0, le=1)


class PatternLibraryResponse(BaseModel):
    version: int
    generated: str
    patterns: dict[str, Any]


class CandidateSummary(BaseModel):
    id: str
    symbol: str | None = None
    n_occurrences: int | None = None
    hit_rate_5m: float | None = None
    sharpe_5m: float | None = None
    min_cosine_dist: float | None = None
    nearest_known_pattern: str | None = None
    status: str | None = None
    exemplar_timestamps: list[int] = Field(default_factory=list)


@dataclass
class DecisionRow:
    candidate_id: str
    decision: Literal["accept", "reject"]
    reviewer: str
    decision_reason: str
    confidence: float | None
    timestamp_utc: str


# ── Helpers ─────────────────────────────────────────────────────────

def _read_library() -> dict[str, Any]:
    if not LIBRARY_PATH.exists():
        raise HTTPException(status_code=404, detail=f"library.yaml not found: {LIBRARY_PATH}")
    with LIBRARY_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            yaml.safe_dump(data, tmp, sort_keys=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_candidates_df() -> pd.DataFrame:
    if not CANDIDATES_DIR.exists():
        return pd.DataFrame()
    files = sorted(CANDIDATES_DIR.glob("candidates_*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in files]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if "id" not in df.columns:
        # fallback: create id from cluster_id and run timestamp
        if "cluster_id" in df.columns:
            df["id"] = df["cluster_id"].astype(str)
    return df


def _append_decision(row: DecisionRow) -> None:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([asdict(row)])
    if DECISIONS_PATH.exists():
        old = pd.read_parquet(DECISIONS_PATH)
        out = pd.concat([old, new_row], ignore_index=True)
    else:
        out = new_row
    out.to_parquet(DECISIONS_PATH, index=False)


def _promote_candidate_to_library(candidate: dict[str, Any]) -> dict[str, Any]:
    lib = _read_library()
    patterns = lib.setdefault("patterns", {})
    pid = str(candidate["id"])

    pattern_def = {
        "l3_signature": candidate.get("l3_signature", "TODO: derive from cluster centroid / exemplar"),
        "l2_manifestation": candidate.get("l2_manifestation", "TODO: derive from 40-dim book shape profile"),
        "l1_footprint": candidate.get("l1_footprint", "TODO: derive from L1 feature profile"),
        "outcome_distribution": candidate.get("outcome_distribution", {
            "horizon_1m": {
                "mean_atr": float(candidate.get("mean_atr_1m", 0.0)),
                "std": float(candidate.get("std_atr_1m", 0.0)),
                "hit_rate": float(candidate.get("hit_rate_1m", 0.0)),
                "n": int(candidate.get("n_occurrences", 0)),
            },
            "horizon_5m": {
                "mean_atr": float(candidate.get("mean_atr_5m", 0.0)),
                "std": float(candidate.get("std_atr_5m", 0.0)),
                "hit_rate": float(candidate.get("hit_rate_5m", 0.0)),
                "n": int(candidate.get("n_occurrences", 0)),
            },
        }),
        "min_sample_size": int(candidate.get("n_occurrences", 0)),
        "regime_specific": True,
        "instruments": [candidate.get("symbol", "NQ")],
        "confidence_tier_cutoffs": {"A": 0.85, "B": 0.72, "C": 0.60},
        "status": "mined_approved",
        "validation_hash": candidate.get("validation_hash", "sha256:TODO"),
        "source_candidate_id": pid,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }

    patterns[pid] = pattern_def
    lib["generated"] = datetime.now(timezone.utc).date().isoformat()
    lib["version"] = int(lib.get("version", 1)) + 1

    _atomic_write_yaml(LIBRARY_PATH, lib)
    return lib


# ── Endpoints ─────────────────────────────────────────────────────

@router.get("/api/patterns/library", response_model=PatternLibraryResponse)
def get_library() -> PatternLibraryResponse:
    lib = _read_library()
    return PatternLibraryResponse(**{
        "version": int(lib.get("version", 1)),
        "generated": str(lib.get("generated", "")),
        "patterns": lib.get("patterns", {}),
    })


@router.get("/api/patterns/candidates", response_model=list[CandidateSummary])
def get_candidates(status: str | None = None) -> list[CandidateSummary]:
    df = _load_candidates_df()
    if df.empty:
        return []

    # show only status='candidate' by default
    if status is None:
        df = df[df.get("status", "candidate") == "candidate"] if "status" in df.columns else df
    else:
        if "status" in df.columns:
            df = df[df["status"] == status]

    out = []
    for _, r in df.iterrows():
        exemplars = r.get("exemplar_timestamps", r.get("exemplar_timestamps_json", []))
        if isinstance(exemplars, str):
            try:
                exemplars = json.loads(exemplars)
            except Exception:
                exemplars = []

        out.append(CandidateSummary(
            id=str(r.get("id", r.get("cluster_id", ""))),
            symbol=r.get("symbol"),
            n_occurrences=int(r["n_occurrences"]) if pd.notna(r.get("n_occurrences")) else None,
            hit_rate_5m=float(r["hit_rate_5m"]) if pd.notna(r.get("hit_rate_5m")) else None,
            sharpe_5m=float(r["sharpe_5m"]) if pd.notna(r.get("sharpe_5m")) else None,
            min_cosine_dist=float(r["min_cosine_dist"]) if pd.notna(r.get("min_cosine_dist")) else None,
            nearest_known_pattern=r.get("nearest_known_pattern"),
            status=r.get("status"),
            exemplar_timestamps=[int(x) for x in (exemplars or [])],
        ))
    return out


@router.get("/api/patterns/candidate/{candidate_id}")
def get_candidate(candidate_id: str) -> dict[str, Any]:
    df = _load_candidates_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No candidates found")

    m = df[(df.get("id", df.get("cluster_id")).astype(str) == str(candidate_id))]
    if m.empty:
        raise HTTPException(status_code=404, detail=f"Candidate not found: {candidate_id}")

    row = m.iloc[0].to_dict()

    # optionally load exemplar window parquet for this candidate
    exemplar_dir = CANDIDATES_DIR / "exemplars" / str(candidate_id)
    exemplars = sorted(exemplar_dir.glob("*.parquet")) if exemplar_dir.exists() else []
    row["exemplar_files"] = [str(p) for p in exemplars]
    return row


@router.post("/api/patterns/candidate/{candidate_id}/accept")
def accept_candidate(candidate_id: str, body: CandidateDecisionRequest = Body(default=CandidateDecisionRequest())) -> dict[str, Any]:
    df = _load_candidates_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No candidates found")

    m = df[(df.get("id", df.get("cluster_id")).astype(str) == str(candidate_id))]
    if m.empty:
        raise HTTPException(status_code=404, detail=f"Candidate not found: {candidate_id}")

    candidate = m.iloc[0].to_dict()
    lib = _promote_candidate_to_library(candidate)

    _append_decision(DecisionRow(
        candidate_id=str(candidate_id),
        decision="accept",
        reviewer=body.reviewer,
        decision_reason=body.decision_reason,
        confidence=body.confidence,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    ))

    return {
        "ok": True,
        "candidate_id": candidate_id,
        "decision": "accept",
        "library_version": lib.get("version"),
        "status": "mined_approved",
    }


@router.post("/api/patterns/candidate/{candidate_id}/reject")
def reject_candidate(candidate_id: str, body: CandidateDecisionRequest = Body(default=CandidateDecisionRequest())) -> dict[str, Any]:
    df = _load_candidates_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No candidates found")

    m = df[(df.get("id", df.get("cluster_id")).astype(str) == str(candidate_id))]
    if m.empty:
        raise HTTPException(status_code=404, detail=f"Candidate not found: {candidate_id}")

    _append_decision(DecisionRow(
        candidate_id=str(candidate_id),
        decision="reject",
        reviewer=body.reviewer,
        decision_reason=body.decision_reason,
        confidence=body.confidence,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    ))

    return {
        "ok": True,
        "candidate_id": candidate_id,
        "decision": "reject",
        "status": "rejected",
        "decisions_path": str(DECISIONS_PATH),
    }
