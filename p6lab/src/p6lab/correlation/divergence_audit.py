"""
DivergenceAuditor — side-by-side live ↔ replay drift detector.

Subscribes to two ``MatchBroker`` instances (one per feed) and pairs up
matches by ``(pattern_id, match_window_end_ms)`` within a small jitter
window. For every matched pair, records the absolute delta in
``ensemble_score`` and logs deltas above ``delta_threshold`` to a JSONL
file alongside the other audit artefacts.

Typical wiring in a live deployment:

    replay_broker = MatchBroker()
    live_broker   = MatchBroker()
    engine_replay = CorrelationEngine(..., broker=replay_broker)
    engine_live   = CorrelationEngine(..., broker=live_broker)

    auditor = DivergenceAuditor(
        output_path=Path("artifacts/live/divergence.jsonl"),
        delta_threshold=0.05,
    )
    auditor.attach(replay_broker, source="replay")
    auditor.attach(live_broker,   source="live")

    # Run both engines on the same symbol. After a while:
    print(auditor.snapshot())

The auditor itself is a **renderer peer** in the broker's subscriber
set — it does not interpose, cannot block either engine, and can be
attached/detached at runtime.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DivergenceAuditor:
    """Live ↔ replay drift detector.

    Parameters
    ----------
    output_path
        JSONL file to append divergence entries to. Created on first write.
    delta_threshold
        Absolute ``|score_A - score_B|`` above which an entry is logged.
        Set to 0 to log every paired match.
    pair_window_ms
        Maximum ms gap between ``match_window_end_ms`` of two candidates
        before they stop being considered a pair. Default 100ms — wider
        than snapshot cadence, tight enough that unrelated matches don't
        cross-pair.
    buffer_size
        Per-source deque length for unmatched matches. FIFO; oldest
        unmatched candidates are dropped first.
    """

    def __init__(
        self,
        output_path: Path | str,
        *,
        delta_threshold: float = 0.05,
        pair_window_ms: int = 100,
        buffer_size: int = 2_000,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.delta_threshold = float(delta_threshold)
        self.pair_window_ms = int(pair_window_ms)

        self._lock = threading.Lock()
        self._buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self._attached: list[tuple[Any, Callable, str]] = []

        # Running stats (available via snapshot())
        self._total_matches: dict[str, int] = defaultdict(int)
        self._pairs = 0
        self._divergences = 0
        self._delta_hist: deque[float] = deque(maxlen=2_000)
        self._started_at = time.time()

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def attach(self, broker, *, source: str) -> Callable:
        """Subscribe to ``broker`` tagging every event as coming from *source*.

        Returns the callable registered on the broker so it can be
        explicitly ``broker.unsubscribe(...)``-ed later if needed.
        """
        def _sub(match, _src=source):
            self._on_event(_src, match)
        broker.subscribe(_sub)
        self._attached.append((broker, _sub, source))
        return _sub

    def detach_all(self) -> None:
        for broker, sub, _src in self._attached:
            try:
                broker.unsubscribe(sub)
            except Exception:
                pass
        self._attached.clear()

    # ------------------------------------------------------------------
    # Core pairing
    # ------------------------------------------------------------------

    def _on_event(self, source: str, match: Any) -> None:
        end_ms = self._attr(match, "match_window_end_ms") or self._attr(match, "timestamp_ms") or 0
        pattern_id = self._attr(match, "pattern_id") or "?"
        score = float(self._attr(match, "ensemble_score") or 0.0)
        entry = (int(end_ms), str(pattern_id), score, match)

        with self._lock:
            self._total_matches[source] += 1
            # Try to pair against the *other* source
            other = self._other_source(source)
            if other:
                partner = self._pop_best_pair(other, end_ms, pattern_id)
                if partner is not None:
                    self._record_pair(source, entry, other, partner)
                    return
            # No pair yet — buffer for later
            self._buffers[source].append(entry)

    def _pop_best_pair(self, other: str, end_ms: int, pattern_id: str):
        """Find and remove the best candidate from *other*'s buffer.

        Best = same pattern_id AND |Δ end_ms| ≤ pair_window_ms.
        Prefer smallest timestamp delta.
        """
        best_i, best_delta = -1, None
        buf = self._buffers[other]
        for i, (e_ts, e_pid, _, _) in enumerate(buf):
            if e_pid != pattern_id:
                continue
            d = abs(e_ts - end_ms)
            if d > self.pair_window_ms:
                continue
            if best_delta is None or d < best_delta:
                best_delta, best_i = d, i
        if best_i < 0:
            return None
        # Remove by index from a deque: splice
        entry = buf[best_i]
        del buf[best_i]
        return entry

    def _record_pair(self, source_a: str, entry_a, source_b: str, entry_b) -> None:
        ts_a, pid, score_a, match_a = entry_a
        ts_b, _,  score_b, match_b = entry_b
        delta = abs(score_a - score_b)
        self._pairs += 1
        self._delta_hist.append(delta)

        if delta < self.delta_threshold:
            return
        self._divergences += 1
        payload = {
            "_type": "divergence",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pattern_id": pid,
            "source_a": source_a, "source_b": source_b,
            "score_a": score_a, "score_b": score_b,
            "delta": delta,
            "ts_a_ms": ts_a, "ts_b_ms": ts_b,
            "ts_delta_ms": ts_a - ts_b,
        }
        try:
            with open(self.output_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            logger.exception("divergence auditor: failed to write entry")

    def _other_source(self, source: str) -> str | None:
        known = set(self._buffers.keys()) | {s for _, _, s in self._attached}
        others = [s for s in known if s != source]
        return others[0] if len(others) == 1 else None

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            uptime = round(time.time() - self._started_at, 1)
            deltas = list(self._delta_hist)
            hist = {
                "n": len(deltas),
                "max":  max(deltas) if deltas else 0.0,
                "mean": (sum(deltas) / len(deltas)) if deltas else 0.0,
                "p50":  _pct(deltas, 50),
                "p95":  _pct(deltas, 95),
                "p99":  _pct(deltas, 99),
            }
            unmatched = {src: len(buf) for src, buf in self._buffers.items()}
            return {
                "uptime_seconds": uptime,
                "total_matches_per_source": dict(self._total_matches),
                "pairs_formed": self._pairs,
                "divergences_logged": self._divergences,
                "delta_threshold": self.delta_threshold,
                "delta_stats": hist,
                "unmatched_per_source": unmatched,
                "output_path": str(self.output_path),
            }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _attr(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)


def _pct(seq: list[float], p: int) -> float:
    if not seq:
        return 0.0
    s = sorted(seq)
    k = (len(s) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] * (hi - k) + s[hi] * (k - lo)
