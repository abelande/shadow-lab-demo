"""
correlation_api.py — §11.4 Correlation Engine API + WS publishers

Endpoints:
- GET  /api/correlation/status
- POST /api/correlation/reload

WebSocket messages pushed by engine loop:
- correlation_match
- fragility_update
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pickle

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["correlation"])

ARTIFACTS_ROOT = Path("workspace/myquantlab/artifacts/p6lab")
MODELS_DIR = ARTIFACTS_ROOT / "correlation_runs" / "models"
FI_MODELS_DIR = ARTIFACTS_ROOT / "cascade_models"


class CorrelationStatus(BaseModel):
    ready: bool
    loaded_model_path: str | None
    loaded_at_utc: str | None
    model_version: str | None
    library_version: int | None
    latency_target_ms: int
    notes: str | None = None


@dataclass
class CorrelationRuntimeState:
    model: Any = None
    model_path: Path | None = None
    loaded_at: str | None = None
    model_version: str | None = None
    library_version: int | None = None
    ready: bool = False


STATE = CorrelationRuntimeState()
LATENCY_TARGET_MS = 50


def _latest_model_path() -> Path | None:
    if not MODELS_DIR.exists():
        return None
    files = sorted(MODELS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_model(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Model artifact must be dict")
    return obj


def reload_latest_model() -> CorrelationRuntimeState:
    p = _latest_model_path()
    if p is None:
        raise FileNotFoundError(f"No model files found in {MODELS_DIR}")

    model = _load_model(p)
    STATE.model = model
    STATE.model_path = p
    STATE.loaded_at = datetime.now(timezone.utc).isoformat()
    STATE.model_version = str(model.get("model_version", p.stem))
    STATE.library_version = model.get("library_version")
    STATE.ready = True
    return STATE


@router.get("/api/correlation/status", response_model=CorrelationStatus)
def get_status() -> CorrelationStatus:
    return CorrelationStatus(
        ready=STATE.ready,
        loaded_model_path=str(STATE.model_path) if STATE.model_path else None,
        loaded_at_utc=STATE.loaded_at,
        model_version=STATE.model_version,
        library_version=STATE.library_version,
        latency_target_ms=LATENCY_TARGET_MS,
        notes="Engine target <50ms/match. Messages: correlation_match, fragility_update",
    )


@router.post("/api/correlation/reload")
def reload_model() -> dict[str, Any]:
    try:
        st = reload_latest_model()
        return {
            "ok": True,
            "ready": st.ready,
            "loaded_model_path": str(st.model_path),
            "loaded_at_utc": st.loaded_at,
            "model_version": st.model_version,
            "library_version": st.library_version,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")


# ── WebSocket publishing helpers (for engine_runner integration) ─────────────

def build_correlation_match_message(
    timestamp_ms: int,
    pattern_id: str,
    ensemble_score: float,
    tier: str,
    expected_direction: str,
    expected_move_atr: float,
    context: dict[str, Any] | None = None,
    *,
    template_similarity: float | None = None,
    mahalanobis_score: float | None = None,
    contextual_score: float | None = None,
    stage1_score: float | None = None,
    match_window_start_ms: int | None = None,
    match_window_end_ms: int | None = None,
    regime: str | None = None,
    instrument: str | None = None,
) -> dict[str, Any]:
    """Message schema for websocket event type `correlation_match`.

    Extended fields (``template_similarity`` etc.) feed the Live Signal
    Dock's click-to-expand score breakdown. All extended fields are
    optional — older consumers that only read the core fields keep working.
    """
    msg: dict[str, Any] = {
        "type": "correlation_match",
        "timestamp_ms": timestamp_ms,
        "pattern_id": pattern_id,
        "ensemble_score": float(ensemble_score),
        "tier": tier,
        "expected_direction": expected_direction,
        "expected_move_atr": float(expected_move_atr),
        "context": context or {},
    }
    if template_similarity   is not None: msg["template_similarity"]   = float(template_similarity)
    if mahalanobis_score     is not None: msg["mahalanobis_score"]     = float(mahalanobis_score)
    if contextual_score      is not None: msg["contextual_score"]      = float(contextual_score)
    if stage1_score          is not None: msg["stage1_score"]          = float(stage1_score)
    if match_window_start_ms is not None: msg["match_window_start_ms"] = int(match_window_start_ms)
    if match_window_end_ms   is not None: msg["match_window_end_ms"]   = int(match_window_end_ms)
    if regime                is not None: msg["regime"]                = regime
    if instrument            is not None: msg["instrument"]            = instrument
    return msg


def match_to_ws_message(pm) -> dict[str, Any]:
    """Convert a ``PatternMatch`` dataclass to the dock-compatible WS payload.

    Use this as the target of ``MatchBroker.subscribe(...)`` so every dock
    consumer + chart overlay + audit log sees the same normalized shape.
    """
    return build_correlation_match_message(
        timestamp_ms=pm.match_window_end_ms,
        pattern_id=pm.pattern_id,
        ensemble_score=pm.ensemble_score,
        tier=pm.confidence_tier,
        expected_direction=pm.expected_direction,
        expected_move_atr=pm.expected_move_atr,
        template_similarity=pm.template_similarity,
        mahalanobis_score=pm.mahalanobis_score,
        contextual_score=pm.contextual_score,
        stage1_score=pm.stage1_score,
        match_window_start_ms=pm.match_window_start_ms,
        match_window_end_ms=pm.match_window_end_ms,
        regime=pm.regime,
        instrument=pm.instrument,
    )


def install_broker_subscribers(
    broker,
    *,
    ws_broadcast,
    audit_log_path=None,
    enable_metrics: bool = True,
    discord_webhook_url: str | None = None,
    slack_webhook_url: str | None = None,
    webhook_tier_filter: set[str] | None = None,
    webhook_min_score: float | None = None,
    webhook_max_per_minute: int = 20,
):
    """Wire the ``MatchBroker`` to the server's live consumers.

    This is the single place where ``CorrelationEngine`` output fans out
    into the rest of the application. Adding a new consumer means adding
    one ``broker.subscribe(...)`` line here — zero changes to the engine.

    Parameters
    ----------
    broker
        The ``MatchBroker`` instance the engine is emitting into.
    ws_broadcast
        Server's async WebSocket broadcaster. Signature ``(msg: dict) -> Awaitable``.
    audit_log_path
        Where to write the JSONL audit trail. None = disabled.
    enable_metrics
        If True, attach a ``MetricsRenderer``. Prometheus if the client is
        installed, in-memory fallback otherwise. Returned on the handle below.
    discord_webhook_url, slack_webhook_url
        Optional chat-webhook URLs. Typically passed from environment
        variables ``P6LAB_DISCORD_WEBHOOK_URL`` / ``P6LAB_SLACK_WEBHOOK_URL``.
    webhook_tier_filter
        Tier set to forward to chat webhooks. Default ``{"A"}``.
    webhook_max_per_minute
        Rate cap per webhook. Excess matches dropped with a warning.

    Returns
    -------
    dict
        Handles to the installed renderers, e.g. ``{"metrics": MetricsRenderer}``,
        so callers can query ``metrics.snapshot()`` or start the Prom HTTP
        server.
    """
    handles: dict[str, Any] = {}

    # --- WebSocket fanout ---
    def _ws_subscriber(pm):
        msg = match_to_ws_message(pm)
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(asyncio.create_task, ws_broadcast(msg))
        except RuntimeError:
            ws_broadcast(msg)   # type: ignore[func-returns-value]
    broker.subscribe(_ws_subscriber)

    # --- Audit log (JSONL) ---
    if audit_log_path:
        from p6lab.correlation.renderers import AuditLogRenderer
        audit = AuditLogRenderer(audit_log_path, include_run_meta=True, fsync=False)
        broker.subscribe(audit)
        handles["audit"] = audit

    # --- Metrics (Prometheus or in-memory) ---
    if enable_metrics:
        from p6lab.correlation.renderers import MetricsRenderer
        metrics = MetricsRenderer()
        broker.subscribe(metrics)
        handles["metrics"] = metrics

    # --- Discord webhook ---
    if discord_webhook_url:
        from p6lab.correlation.renderers import WebhookRenderer
        discord = WebhookRenderer(
            discord_webhook_url, platform="discord",
            tier_filter=webhook_tier_filter or {"A", "B"},
            min_score=webhook_min_score,
            max_per_minute=webhook_max_per_minute,
        )
        broker.subscribe(discord)
        handles["discord"] = discord

    # --- Slack webhook ---
    if slack_webhook_url:
        from p6lab.correlation.renderers import WebhookRenderer
        slack = WebhookRenderer(
            slack_webhook_url, platform="slack",
            tier_filter=webhook_tier_filter or {"A", "B"},
            min_score=webhook_min_score,
            max_per_minute=webhook_max_per_minute,
        )
        broker.subscribe(slack)
        handles["slack"] = slack

    return handles


def build_fragility_update_message(
    timestamp_ms: int,
    FI_fast: float,
    FI_full: float,
    DF: float,
    CF: float,
    RF: float,
    SF: float,
    FT: float,
    CIS: float,
) -> dict[str, Any]:
    """Message schema for websocket event type `fragility_update`."""
    return {
        "type": "fragility_update",
        "timestamp_ms": timestamp_ms,
        "FI_fast": float(FI_fast),
        "FI_full": float(FI_full),
        "DF": float(DF),
        "CF": float(CF),
        "RF": float(RF),
        "SF": float(SF),
        "FT": float(FT),
        "CIS": float(CIS),
    }


# Example engine loop pseudocode (for integration reference):
"""
async def engine_runner_loop(ws_broadcast, replay_stream):
    if not STATE.ready:
        try:
            reload_latest_model()
        except Exception:
            pass

    for frame in replay_stream:  # each snapshot
        # 1) run correlation engine
        matches = []
        if STATE.ready:
            engine = STATE.model['engine'] if 'engine' in STATE.model else None
            # matches = engine.match(l2_window=..., l1_window=..., context=...)

        # 2) push correlation matches above tier C (>=0.60)
        for m in matches:
            if m.ensemble_score >= 0.60:
                msg = build_correlation_match_message(
                    timestamp_ms=frame.timestamp_ms,
                    pattern_id=m.pattern_id,
                    ensemble_score=m.ensemble_score,
                    tier=m.tier,
                    expected_direction=m.expected_direction,
                    expected_move_atr=m.expected_move_atr,
                )
                await ws_broadcast(msg)

        # 3) compute fragility index and push every snapshot
        # fi = compute_fi(...)
        # await ws_broadcast(build_fragility_update_message(...))
"""
