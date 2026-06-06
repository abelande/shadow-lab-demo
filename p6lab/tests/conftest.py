"""Pytest configuration for p6lab test suite."""
from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="include slow tests (e.g. NB04 mining notebook, ~3min)",
    )
