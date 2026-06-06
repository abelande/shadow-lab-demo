"""Root conftest — registers the project root as the 'p6' package.

The project directory is named 'p6 - staircase-model' (spaces + hyphen),
which is not a valid Python identifier. This conftest uses importlib to
register the package under the canonical name 'p6' before any tests run.
"""
from __future__ import annotations
import importlib.util
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if "p6" not in sys.modules:
    _init = os.path.join(_PROJECT_ROOT, "__init__.py")
    _spec = importlib.util.spec_from_file_location(
        "p6",
        _init,
        submodule_search_locations=[_PROJECT_ROOT],
    )
    _pkg = importlib.util.module_from_spec(_spec)
    _pkg.__path__ = [_PROJECT_ROOT]  # type: ignore[assignment]
    _pkg.__package__ = "p6"
    sys.modules["p6"] = _pkg
    _spec.loader.exec_module(_pkg)  # type: ignore[union-attr]
