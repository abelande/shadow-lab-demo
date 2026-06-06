#!/usr/bin/env python3
"""Wave 8.5-N — replay 30 days of outcomes through OutcomeTrackerRenderer."""
import glob, json
from pathlib import Path

from p6lab.correlation.renderers.outcome_tracker import (
    OutcomeTrackerRenderer, _PatternAggregate,
)
from p6lab.patterns.library import PatternLibrary

LIB_PATH = Path("artifacts/p6lab/pattern_library/library.yaml")
lib = PatternLibrary(LIB_PATH); lib.load()
pre = {n: p.status.value for n, p in lib._data.patterns.items()}

tracker = OutcomeTrackerRenderer(
    outcomes_path=Path("artifacts/p6lab/outcomes/wave85N_replay.jsonl"),
    library=lib,
    reaggregate_every_n=50,
    retire_below_hit_rate=0.50,
)

total = 0
for path in sorted(glob.glob("artifacts/p6lab/outcomes/shadow-*.jsonl")):
    for line in open(path):
        row = json.loads(line)
        pid = row["pattern_id"]
        agg = tracker._aggregates.setdefault(pid, _PatternAggregate())
        agg.add(row["entry_ts_ms"], float(row["realized_return"]), bool(row["hit"]))
        tracker.outcomes_closed += 1
        tracker._closes_since_reagg += 1
        total += 1
        if tracker._closes_since_reagg >= tracker.reaggregate_every_n:
            tracker.reaggregate()

tracker.reaggregate()

post = {n: p.status.value for n, p in lib._data.patterns.items()}
changed = [(n, pre[n], post[n]) for n in pre if pre[n] != post[n]]
print(f"replayed {total} outcomes")
print(f"patterns changed status: {len(changed)}")
for n, o, p in changed:
    print(f"  {n}: {o} → {p}")