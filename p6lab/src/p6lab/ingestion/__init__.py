"""
p6lab.ingestion — Data preparation infrastructure.

Spec: p6-notebook-lab-spec.md §3 Shared Infrastructure — p6lab.ingestion
  §3.1 triple_view.py    — time-aligned L3/L2/L1 triples emitter
  §3.2 event_windowing.py — pluggable window iterator
  §3.3 instrument_normalizer.py — cross-instrument normalization

Import from submodules directly to keep this package lazy — some
submodules depend on later phases:
    from p6lab.ingestion.triple_view import TripleViewEmitter
    from p6lab.ingestion.event_windowing import WindowIterator
"""
