"""
Cross-instrument smoketest — run each notebook against ES/CL/GC/SI in addition
to NQ, catching instrument-specific assumptions (tick size, symbol filtering,
book density) before they bite a live run.

The data slice is swapped via env vars read by ``notebooks/_common.py``.
NQ itself is already covered by ``test_smoke_notebooks.py`` — this file
exercises the *other* instruments.

Run with:
    make smoketest-instruments
    pytest tests/test_smoke_instruments.py -v

Not marked slow — each instrument × notebook is comparable to the NQ
smoketest (~5s/cell); four instruments × three notebooks is ~60s total.
NB04 mining and NB07 cascade are skipped here (threshold-sensitive;
validated separately per-instrument after cascade-taxonomy lands).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError as exc:
    pytest.skip(f"requires nbformat + nbclient: {exc}",
                allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
DATA_DIR = ROOT.parent / "data"

# (instrument_id, data_file_name, tick_size)
INSTRUMENTS = [
    ("ES", "es-mbo-2026-03-27.dbn.zst", 0.25),
    ("CL", "cl-mbo-2026-03-27.dbn.zst", 0.01),
    ("GC", "gc-mbo-2026-03-27.dbn.zst", 0.10),
    ("SI", "si-mbo-2026-03-27.dbn.zst", 0.005),
]

# Only the three "portable" notebooks — NB04 (mining) and NB07 (cascade) have
# threshold params that are NQ-tuned and need cascade-taxonomy work before
# they're instrument-portable.
PORTABLE_NOTEBOOKS = [
    "WRAP_P6_L1_FEATURE_LAB.ipynb",   # NB03
    "WRAP_P6_EXECUTION_SIM.ipynb",    # NB05
    "WRAP_P6_CORRELATION_LAB.ipynb",  # NB06
]


def _execute(nb_path: Path, timeout: int, env: dict) -> nbformat.NotebookNode:
    """Run a notebook with the given env vars inherited by the kernel."""
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
        allow_errors=False,
    )
    # nbclient spawns the kernel as a subprocess; it inherits our env vars.
    old_env = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        client.execute()
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return nb


# Known NQ-tuning failures — tracked in roadmap item "cross-instrument
# robustness" (Wave 2). These xfail until the notebooks are instrument-
# agnostic. The test is about exposing the fragility, not hiding it.
# Re-run verified on 2026-04-15 @ 1000 snapshots; confirmed real failures.
KNOWN_NQ_ONLY: set[tuple[str, str]] = {
    ("WRAP_P6_L1_FEATURE_LAB.ipynb",   "ES"),
    ("WRAP_P6_L1_FEATURE_LAB.ipynb",   "CL"),
    ("WRAP_P6_L1_FEATURE_LAB.ipynb",   "GC"),
    ("WRAP_P6_L1_FEATURE_LAB.ipynb",   "SI"),
    ("WRAP_P6_EXECUTION_SIM.ipynb",    "GC"),
    ("WRAP_P6_EXECUTION_SIM.ipynb",    "SI"),
    ("WRAP_P6_CORRELATION_LAB.ipynb",  "GC"),
    ("WRAP_P6_CORRELATION_LAB.ipynb",  "SI"),
}


@pytest.mark.parametrize(
    "instrument,data_file,tick_size",
    INSTRUMENTS,
    ids=[i[0] for i in INSTRUMENTS],
)
@pytest.mark.parametrize("notebook", PORTABLE_NOTEBOOKS)
def test_notebook_instrument(instrument: str, data_file: str,
                             tick_size: float, notebook: str,
                             request: pytest.FixtureRequest) -> None:
    """Each portable notebook must execute cleanly on the given instrument."""
    if (notebook, instrument) in KNOWN_NQ_ONLY:
        request.node.add_marker(pytest.mark.xfail(
            reason="NQ-tuned notebook; cross-instrument robustness is Wave 2 work",
            strict=False,
        ))

    data_path = DATA_DIR / data_file
    if not data_path.exists():
        pytest.skip(f"data file missing: {data_path}")

    nb_path = NB_DIR / notebook
    assert nb_path.exists(), f"missing notebook: {nb_path}"

    env = {
        "P6LAB_DATA_FILE": str(data_path),
        "P6LAB_SYMBOL": instrument,
        "P6LAB_TICK_SIZE": str(tick_size),
        "P6LAB_MAX_SNAPSHOTS": "1000",
    }
    nb = _execute(nb_path, timeout=300, env=env)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                pytest.fail(
                    f"{notebook} on {instrument} cell {i} errored: "
                    f"{out['ename']}: {out['evalue']}"
                )
