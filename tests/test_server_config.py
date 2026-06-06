"""Tests for ServerConfig."""
from __future__ import annotations
import pytest
from p6.server.config import ServerConfig


def test_server_config_defaults():
    cfg = ServerConfig()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8420
    assert cfg.instrument == "ES"
    assert cfg.mode == "paused"
    assert cfg.frame_rate_limit == 10.0


def test_server_config_ws_max_clients_default():
    cfg = ServerConfig()
    assert cfg.ws_max_clients == 50


def test_server_config_webhook_url_none_by_default():
    cfg = ServerConfig()
    assert cfg.webhook_url is None


def test_server_config_risk_defaults():
    cfg = ServerConfig()
    assert cfg.risk_max_position == 1.0
    assert cfg.risk_abstain_threshold == 0.3


def test_server_config_override():
    cfg = ServerConfig(host="127.0.0.1", port=9000, instrument="NQ")
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9000
    assert cfg.instrument == "NQ"


def test_server_config_mode_update():
    cfg = ServerConfig()
    cfg.mode = "live"
    assert cfg.mode == "live"


def test_server_config_frame_rate_override():
    cfg = ServerConfig(frame_rate_limit=30.0)
    assert cfg.frame_rate_limit == 30.0
