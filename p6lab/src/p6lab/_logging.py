"""
p6lab._logging
==============

Structured-logging toolkit for p6lab. One import, two primitives:

    from p6lab._logging import configure_logging, with_context

    configure_logging(json=True)                 # app-startup: wire JSON formatter
    with with_context(correlation_id="abc123"):  # per-cycle: thread an ID
        logger.info("match found")
        # → {"timestamp": "...", "level": "INFO", "message": "match found",
        #    "correlation_id": "abc123", ...}

Design goals
------------
- **Zero external deps.** Stdlib ``logging`` + ``contextvars`` only. No
  ``structlog`` / ``python-json-logger`` so the module can live in the
  core ``p6lab`` runtime without bloating the dep tree.

- **Non-invasive.** Existing ``logger = logging.getLogger(__name__)`` call
  sites keep working. The context + JSON formatting is attached at the
  root logger; records propagate up and pick up the filter + formatter
  without each module being aware.

- **Thread-safe via ContextVar.** Context propagates naturally across
  ``await`` points (asyncio). For threads spawned via ``threading.Thread``
  the caller must explicitly ``contextvars.copy_context()`` — see
  ``WebhookRenderer`` for the reference pattern.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Context propagation
# ---------------------------------------------------------------------------

_CTX_KEYS = ("correlation_id", "instrument", "regime", "symbol", "request_id")
_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "p6lab_log_ctx", default={},
)


def get_context() -> dict:
    """Return a shallow copy of the current log context (read-only view)."""
    return dict(_ctx.get())


def set_context(**kwargs: Any) -> contextvars.Token:
    """Merge *kwargs* into the current context. Returns a reset token."""
    current = _ctx.get()
    return _ctx.set({**current, **kwargs})


def reset_context(token: contextvars.Token) -> None:
    _ctx.reset(token)


@contextlib.contextmanager
def with_context(**kwargs: Any) -> Iterator[None]:
    """Context-manager form of ``set_context``. Guaranteed reset on exit."""
    token = set_context(**kwargs)
    try:
        yield
    finally:
        reset_context(token)


def new_correlation_id() -> str:
    """12-hex-char unique-enough id for per-cycle correlation."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Filter — injects ContextVar state onto every LogRecord
# ---------------------------------------------------------------------------

class ContextFilter(logging.Filter):
    """Attach current ContextVar values to every LogRecord as attributes."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _ctx.get()
        for key in _CTX_KEYS:
            setattr(record, key, ctx.get(key, ""))
        return True


# ---------------------------------------------------------------------------
# JSON formatter — minimal, no external dep
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record. Context fields appear as top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        # %f isn't substituted by logging.Formatter.formatTime — append ms manually.
        ts = f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')}.{int(record.msecs):03d}Z"
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Only include context keys that are populated — keeps noise down
        for key in _CTX_KEYS:
            val = getattr(record, key, "")
            if val:
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Preserve any caller-supplied extras that aren't stock logging attrs
        for k, v in record.__dict__.items():
            if k.startswith("_") or k in _STOCK_ATTRS or k in _CTX_KEYS:
                continue
            if k in payload:
                continue
            try:
                json.dumps(v)        # only keep JSON-serializable extras
                payload[k] = v
            except (TypeError, ValueError):
                pass
        return json.dumps(payload, default=str)


_STOCK_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
})


# ---------------------------------------------------------------------------
# configure_logging — one-shot setup
# ---------------------------------------------------------------------------

def configure_logging(
    level: str | int = "INFO",
    *,
    json: bool = False,
    log_file: Path | str | None = None,
    replace_handlers: bool = True,
) -> None:
    """Install a ContextFilter + formatter on the root logger.

    Parameters
    ----------
    level
        Root logger level — ``"INFO"``, ``"DEBUG"``, etc.
    json
        When True, emit newline-delimited JSON (production). When False,
        emit human-readable ``%(levelname)s %(name)s: %(message)s``.
    log_file
        Optional path — if given, ALSO attach a FileHandler writing the
        same records (same formatter) to disk. Parent dir is created.
    replace_handlers
        When True (default), existing root handlers are removed first
        to prevent duplicate emission. Set False if the app already
        wires its own handlers and just wants the filter attached.
    """
    root = logging.getLogger()
    filter_ = ContextFilter()
    formatter: logging.Formatter = (
        JsonFormatter() if json
        else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )

    if replace_handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(formatter)
    stream.addFilter(filter_)
    root.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.addFilter(filter_)
        root.addHandler(fh)

    root.setLevel(level)
