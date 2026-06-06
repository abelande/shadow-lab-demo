"""
Structured-logging tests.

Verifies:
  - JsonFormatter produces valid JSON with all core fields
  - ContextFilter injects ContextVar state onto every record
  - with_context() threads IDs through nested log calls
  - ContextVar propagates across asyncio await points
  - contextvars.copy_context() lets a thread inherit the parent context
    (matches WebhookRenderer's pattern)
  - configure_logging() is idempotent (no duplicate emission)
  - Engine match() stamps a correlation_id that propagates to broker
    subscribers' logs
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import threading
import contextvars
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent.parent))   # for p6v2.*

from p6lab._logging import (
    ContextFilter, JsonFormatter,
    configure_logging, get_context, new_correlation_id,
    set_context, reset_context, with_context,
)


# ---------------------------------------------------------------------------
# Helper — capture stderr from the root logger into a buffer
# ---------------------------------------------------------------------------

def _install_capture(json_mode: bool = True) -> io.StringIO:
    root = logging.getLogger()
    buf = io.StringIO()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter() if json_mode else logging.Formatter("%(message)s"))
    handler.addFilter(ContextFilter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return buf


@pytest.fixture(autouse=True)
def _reset_root():
    """Snapshot root handlers + level, restore after each test."""
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers_before:
        root.addHandler(h)
    root.setLevel(level_before)


# ---------------------------------------------------------------------------
# Formatter + filter
# ---------------------------------------------------------------------------

def test_json_formatter_shape():
    buf = _install_capture(json_mode=True)
    logging.getLogger("test.module").info("hello world")
    line = buf.getvalue().strip().splitlines()[-1]
    obj = json.loads(line)
    assert obj["level"] == "INFO"
    assert obj["logger"] == "test.module"
    assert obj["message"] == "hello world"
    assert "ts" in obj


def test_context_filter_injects_correlation_id():
    buf = _install_capture(json_mode=True)
    with with_context(correlation_id="abc123", instrument="NQ"):
        logging.getLogger("test").warning("alert!")
    obj = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert obj["correlation_id"] == "abc123"
    assert obj["instrument"] == "NQ"


def test_empty_context_omits_keys():
    """No context set → keys should be absent (not empty strings) from output."""
    buf = _install_capture(json_mode=True)
    logging.getLogger("test").info("plain")
    obj = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert "correlation_id" not in obj


# ---------------------------------------------------------------------------
# ContextVar propagation
# ---------------------------------------------------------------------------

def test_with_context_nested():
    """Inner context overrides outer, resets on exit."""
    with with_context(correlation_id="outer"):
        assert get_context()["correlation_id"] == "outer"
        with with_context(correlation_id="inner"):
            assert get_context()["correlation_id"] == "inner"
        assert get_context()["correlation_id"] == "outer"
    assert get_context() == {}


def test_set_context_reset_roundtrip():
    assert get_context() == {}
    tok = set_context(foo="bar")
    assert get_context()["foo"] == "bar"
    reset_context(tok)
    assert get_context() == {}


def test_new_correlation_id_uniqueness():
    ids = {new_correlation_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_asyncio_context_propagation():
    """ContextVar naturally propagates across `await`."""
    async def child():
        return get_context().get("correlation_id")

    async def main():
        with with_context(correlation_id="async-test"):
            return await child()

    assert asyncio.run(main()) == "async-test"


def test_thread_copy_context_pattern():
    """``contextvars.copy_context()`` makes the thread inherit the parent context."""
    result: dict = {}

    def worker():
        result["cid"] = get_context().get("correlation_id")

    with with_context(correlation_id="thread-test"):
        ctx = contextvars.copy_context()
        t = threading.Thread(target=lambda: ctx.run(worker))
        t.start(); t.join()

    assert result["cid"] == "thread-test"


def test_thread_without_copy_context_loses_it():
    """Plain `threading.Thread(target=fn)` does NOT inherit the context."""
    result: dict = {}

    def worker():
        result["cid"] = get_context().get("correlation_id", "(lost)")

    with with_context(correlation_id="ignored"):
        t = threading.Thread(target=worker)
        t.start(); t.join()

    assert result["cid"] == "(lost)"


# ---------------------------------------------------------------------------
# configure_logging — end-to-end
# ---------------------------------------------------------------------------

def test_configure_logging_replaces_handlers_by_default():
    """Calling twice should NOT stack handlers."""
    configure_logging(level="INFO", json=True)
    configure_logging(level="INFO", json=True)
    n_stream = sum(1 for h in logging.getLogger().handlers
                   if isinstance(h, logging.StreamHandler))
    assert n_stream == 1


def test_configure_logging_writes_file(tmp_path):
    log_path = tmp_path / "app.log"
    configure_logging(level="INFO", json=True, log_file=log_path)
    with with_context(correlation_id="file-test"):
        logging.getLogger("test.file").info("logged to disk")
    text = log_path.read_text().strip()
    assert text, "log file should have content"
    line = text.splitlines()[-1]
    obj = json.loads(line)
    assert obj["correlation_id"] == "file-test"
    assert obj["message"] == "logged to disk"


# ---------------------------------------------------------------------------
# Broker / engine integration
# ---------------------------------------------------------------------------

def test_broker_subscribers_inherit_engine_context():
    """When engine.match() wraps its body in with_context(), any subscriber
    logs fired during dispatch carry the same correlation_id."""
    buf = _install_capture(json_mode=True)
    from p6lab.correlation.match_broker import MatchBroker

    captured_cids = []

    def log_and_capture(match):
        log = logging.getLogger("test.subscriber")
        log.info("subscriber saw match")
        captured_cids.append(get_context().get("correlation_id"))

    bus = MatchBroker()
    bus.subscribe(log_and_capture)

    cid = new_correlation_id()
    with with_context(correlation_id=cid):
        bus.emit({"pattern_id": "X"})

    # Subscriber itself observed the context
    assert captured_cids == [cid]
    # AND the log line carries it
    lines = [json.loads(ln) for ln in buf.getvalue().strip().splitlines() if ln]
    sub_lines = [ln for ln in lines if ln.get("logger") == "test.subscriber"]
    assert sub_lines and sub_lines[0]["correlation_id"] == cid
