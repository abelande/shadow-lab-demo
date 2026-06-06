"""
p6lab.persistence — Wave 8.5-E

Durable state for risk + correlation components. SQLite-backed,
single-file-per-process, WAL journal mode for concurrent readers.

Currently exports:
    StateStore          class
    StateStoreRecord    dataclass marker
"""
from p6lab.persistence.state_store import StateStore

__all__ = ["StateStore"]
