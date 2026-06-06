"""
replay_api.py — §11.1 Replay API extension for Triple View

Adds:
- GET /api/triple_view?symbol&start_ms&end_ms&granularity
- WebSocket message helper: triple_frame
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["replay"])

ARTIFACTS_ROOT = Path("workspace/myquantlab/artifacts/p6lab")
TRIPLE_VIEW_DIR = ARTIFACTS_ROOT / "triple_view"
ALLOWED_GRANULARITIES = {"100ms", "1s", "5s"}


def _parquet_path(symbol: str, granularity: str) -> Path:
    return TRIPLE_VIEW_DIR / f"{symbol}_{granularity}.parquet"


@router.get("/api/triple_view")
def get_triple_view(
    symbol: str = Query(...),
    start_ms: int = Query(..., ge=0),
    end_ms: int = Query(..., ge=0),
    granularity: str = Query("1s"),
    limit: int = Query(5000, ge=1, le=50000),
) -> list[dict[str, Any]]:
    if end_ms < start_ms:
        raise HTTPException(status_code=400, detail="end_ms must be >= start_ms")
    if granularity not in ALLOWED_GRANULARITIES:
        raise HTTPException(status_code=400, detail=f"granularity must be one of {sorted(ALLOWED_GRANULARITIES)}")

    p = _parquet_path(symbol, granularity)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"triple_view parquet not found: {p}")

    try:
        df = pd.read_parquet(p)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed reading parquet: {e}")

    if "timestamp_ms" not in df.columns:
        raise HTTPException(status_code=500, detail="Invalid triple_view schema: missing timestamp_ms")

    df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)].sort_values("timestamp_ms")
    if len(df) > limit:
        df = df.iloc[:limit]

    # Ensure JSON-safe output
    out = []
    for _, r in df.iterrows():
        row = r.to_dict()
        # normalize potentially non-serializable numpy arrays
        for k in ["l2_features", "l2_book_vector", "l1_features"]:
            if k in row and hasattr(row[k], "tolist"):
                row[k] = row[k].tolist()
        # keep l3_events as-is (already JSON-like or stringified)
        out.append(row)

    return out


def build_triple_frame_message(frame: dict[str, Any]) -> dict[str, Any]:
    """WebSocket message schema for event type `triple_frame`."""
    # Frame should mirror TripleFrame dataclass schema in JSON form (§3.1)
    return {
        "type": "triple_frame",
        "timestamp_ms": int(frame.get("timestamp_ms", 0)),
        "symbol": frame.get("symbol"),
        "l3_events": frame.get("l3_events", []),
        "l3_book_snapshot": frame.get("l3_book_snapshot"),
        "l2_features": frame.get("l2_features", []),
        "l2_book_vector": frame.get("l2_book_vector", []),
        "l1_features": frame.get("l1_features", []),
    }


# Integration notes for existing replay websocket loop:
"""
# Inside engine_runner / replay broadcaster:
for frame in replay_frames:
    triple_msg = build_triple_frame_message(frame)
    await ws_broadcast(triple_msg)
"""
