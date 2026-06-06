"""
AuditLogRenderer — append every engine match to a JSONL file.

Append-only; one line per match; newline-delimited JSON so the file can be
tailed in real time and parsed by any JSONL-aware tool (jq, pandas
``read_json(lines=True)``, DuckDB, Spark, ...).

Thread-safe: a lock serializes writes so multiple broker threads can't
interleave lines.

Typical wiring:

    from p6lab.correlation.renderers import AuditLogRenderer
    audit = AuditLogRenderer(
        Path("artifacts/live/matches.jsonl"),
        include_run_meta=True,
    )
    broker.subscribe(audit)
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditLogRenderer:
    """Append a JSONL row per emitted match.

    Parameters
    ----------
    path
        File to write to. Parent directory is created if missing.
    include_run_meta
        If True, writes a single header line ``{"_type": "run_start", ...}``
        with package versions + git SHA on first write, so downstream
        consumers can recover provenance without a sibling file.
    fsync
        Call ``os.fsync`` after each write. Slower but survives OS crashes.
        Default False (fine for replay / smoketest; flip True for live).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        include_run_meta: bool = False,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fsync = fsync
        self._run_meta_written = not include_run_meta
        self.lines_written = 0

    def __call__(self, match: Any) -> None:
        """Broker subscriber entry point."""
        payload = self._to_jsonable(match)
        with self._lock:
            if not self._run_meta_written:
                self._write_header()
                self._run_meta_written = True
            self._append(payload)
            self.lines_written += 1

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _write_header(self) -> None:
        meta = {"_type": "run_start", **_collect_run_meta()}
        self._append(meta)

    def _append(self, obj: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, default=str) + "\n")
            if self._fsync:
                fh.flush()
                import os
                os.fsync(fh.fileno())

    @staticmethod
    def _to_jsonable(match: Any) -> dict[str, Any]:
        """Convert ``PatternMatch`` or any dataclass/dict to JSON-ready dict."""
        if isinstance(match, dict):
            return match
        if is_dataclass(match):
            return asdict(match)
        # Best-effort fallback: capture public attrs
        return {k: getattr(match, k) for k in dir(match)
                if not k.startswith("_") and not callable(getattr(match, k, None))}


def _collect_run_meta() -> dict[str, Any]:
    """Package versions + git SHA + timestamp — keep audit trails provenance-rich."""
    import importlib.metadata as md
    import platform
    import subprocess
    from datetime import datetime, timezone

    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
    }
    pkg_versions: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "lightgbm", "p6lab"):
        try:
            pkg_versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    meta["package_versions"] = pkg_versions

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        meta["git_sha"] = sha
    except Exception:
        pass
    return meta
