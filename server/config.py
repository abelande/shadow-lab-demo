"""Server configuration via Pydantic settings."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class ServerConfig(BaseModel):
    """Runtime configuration for the Staircase Terminal server."""

    # Server
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8420, description="Bind port")

    # Instrument
    instrument: str = Field(default="ES", description="Active instrument symbol")

    # WebSocket
    ws_max_clients: int = Field(default=50, description="Max concurrent WebSocket clients")
    frame_rate_limit: float = Field(default=2.0, description="Max frames/sec to browser (decoupled from processing rate)")

    # Webhook bridge
    webhook_url: Optional[str] = Field(default=None, description="TradingView webhook URL")
    webhook_confidence_threshold: float = Field(default=0.7, description="Min confidence to fire webhook")
    webhook_rate_limit_seconds: float = Field(default=30.0, description="Min seconds between webhooks")

    # Risk thresholds
    risk_max_position: float = Field(default=1.0, description="Max position size multiplier")
    risk_abstain_threshold: float = Field(default=0.3, description="Abstain below this confidence")

    # Engine mode
    mode: str = Field(default="paused", description="Engine mode: live, replay, paused")


# Singleton config instance
config = ServerConfig()
