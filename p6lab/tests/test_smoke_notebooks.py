"""
Smoke-test the lab notebooks end-to-end.

Each notebook is executed in-place via nbclient under the *current* installed
package set, against the file/window pinned in `notebooks/_common.py`.
Failures in any cell (assertion gate, import, computation) fail the test.

Run with:
    pytest tests/test_smoke_notebooks.py -v          # all 5 notebooks
    pytest tests/test_smoke_notebooks.py -v -k nb03  # one
    pytest tests/test_smoke_notebooks.py -v -m fast  # only the cheap ones

Or via the Makefile:
    make smoketest
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError as exc:
    pytest.skip(f"smoketest requires nbformat + nbclient: {exc}",
                allow_module_level=True)

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"

# (notebook_name, marker, per-cell timeout in seconds)
NOTEBOOKS = [
    ("WRAP_P6_L1_FEATURE_LAB.ipynb",  "fast", 300),  # NB03
    ("WRAP_P6_PATTERN_MINING.ipynb",  "slow", 600),  # NB04 (1.8M MBO events)
    ("WRAP_P6_EXECUTION_SIM.ipynb",   "fast", 300),  # NB05
    ("WRAP_P6_CORRELATION_LAB.ipynb", "fast", 400),  # NB06
    ("WRAP_P6_CASCADE_LAB.ipynb",     "fast", 300),  # NB07
    ("WRAP_P6_MULTIDAY_CPCV.ipynb",   "slow", 600),  # Wave 2 #2 (multi-day)
]

log = logging.getLogger(__name__)


def _execute(nb_path: Path, timeout: int) -> nbformat.NotebookNode:
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
        allow_errors=False,
    )
    client.execute()
    return nb


@pytest.mark.parametrize(
    "nb_name,marker,timeout",
    NOTEBOOKS,
    ids=[n for n, *_ in NOTEBOOKS],
)
def test_notebook_smoke(nb_name: str, marker: str, timeout: int,
                        request: pytest.FixtureRequest) -> None:
    """Execute the notebook end-to-end; any cell error fails the test."""
    if marker == "slow" and not request.config.getoption("--slow", default=False):
        pytest.skip("slow notebook — pass --slow to include")

    nb_path = NB_DIR / nb_name
    assert nb_path.exists(), f"missing notebook: {nb_path}"

    nb = _execute(nb_path, timeout)

    # Surface any cell-level error output (defence-in-depth; nbclient should
    # have raised, but `allow_errors=False` doesn't catch warnings).
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                pytest.fail(
                    f"{nb_name} cell {i} errored: "
                    f"{out['ename']}: {out['evalue']}"
                )


# --slow option registered in tests/conftest.py
