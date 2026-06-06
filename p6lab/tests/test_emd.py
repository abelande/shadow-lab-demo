"""Tests for p6lab.features.emd (Wave 7 Phase 7D)."""
from __future__ import annotations

import numpy as np
import pytest

from p6lab.features.emd import (
    EMD_FEATURE_NAMES,
    aggregate_imfs,
    decompose,
    snapshot_emd_features,
)


def test_warmup_returns_empty_decomposition() -> None:
    assert decompose([1.0, 2.0]) == []


def test_decompose_composite_returns_multiple_imfs() -> None:
    """A 2-frequency sum should be decomposable into at least 2 IMFs."""
    n = 512
    t = np.linspace(0, 10, n)
    slow = np.sin(2 * np.pi * 0.2 * t)
    fast = np.sin(2 * np.pi * 3.0 * t)
    imfs = decompose(slow + fast, max_imfs=5)
    assert len(imfs) >= 2


def test_aggregate_imfs_zero_on_empty() -> None:
    snap = aggregate_imfs([])
    for k in EMD_FEATURE_NAMES:
        assert snap[k] == 0.0


def test_aggregate_imfs_energy_positive() -> None:
    imfs = [np.asarray([1.0, -1.0, 1.0, -1.0]), np.asarray([0.1, 0.2, 0.3, 0.4])]
    snap = aggregate_imfs(imfs)
    assert snap["emd_energy_imf_1"] > 0
    assert snap["emd_energy_imf_2"] > 0


def test_snapshot_runs_end_to_end() -> None:
    n = 512
    rng = np.random.default_rng(0)
    mid = 20_000.0 + np.cumsum(rng.normal(0, 0.25, n))
    snap = snapshot_emd_features(mid, max_imfs=3)
    for k in EMD_FEATURE_NAMES:
        assert k in snap
        assert np.isfinite(snap[k])


def test_trend_slope_captures_rising_trend() -> None:
    """Pure rising-trend input → residual trend slope should be positive."""
    x = np.linspace(0, 100, 256)   # strictly rising
    snap = snapshot_emd_features(x, max_imfs=3)
    assert snap["emd_residual_trend_slope"] >= 0


# Wave 8.5-C tests
def test_wave_85_c_classical_fallback_warns_once(caplog, monkeypatch) -> None:
    """When PyEMD isn't available, decompose() must log a single
    warning per process (via the module-level sentinel) and still
    return classical-sifted IMFs.

    We force the ImportError by making the PyEMD import fail, and we
    reset the sentinel in both directions (before + after) so other
    tests aren't affected.
    """
    import sys

    import p6lab.features.emd as emd_mod
    monkeypatch.setattr(emd_mod, "_WARNED_CLASSICAL_FALLBACK", False)
    # Block PyEMD import by injecting a sentinel into sys.modules
    monkeypatch.setitem(sys.modules, "PyEMD", None)

    caplog.clear()
    caplog.set_level("WARNING", logger="p6lab.features.emd")
    # First call triggers the warning
    emd_mod.decompose(np.arange(64, dtype=float))
    warnings = [r for r in caplog.records if "classical sifting" in r.getMessage()]
    assert len(warnings) == 1, f"expected exactly 1 warning on first call, got {len(warnings)}"
    # Second call: warning must NOT re-fire (sentinel already set)
    caplog.clear()
    emd_mod.decompose(np.arange(64, dtype=float))
    warnings_2 = [r for r in caplog.records if "classical sifting" in r.getMessage()]
    assert len(warnings_2) == 0, "warning fired twice — sentinel is broken"
    # Sentinel should now be True
    assert emd_mod._WARNED_CLASSICAL_FALLBACK is True


def test_wave_85_c_pyemd_path_does_not_warn(caplog) -> None:
    """When PyEMD IS installed, the warning must never fire. Exercises
    the happy path explicitly so a future refactor that moves the
    warning outside the ImportError handler is caught."""
    import p6lab.features.emd as emd_mod
    # Only run if PyEMD is actually installed in this environment —
    # skip otherwise so the test is portable.
    try:
        import PyEMD  # noqa: F401
    except ImportError:
        pytest.skip("PyEMD not installed; this test verifies the non-fallback path")
    caplog.clear()
    caplog.set_level("WARNING", logger="p6lab.features.emd")
    emd_mod.decompose(np.arange(64, dtype=float))
    warnings = [r for r in caplog.records if "classical sifting" in r.getMessage()]
    assert len(warnings) == 0, "warning fired on the PyEMD-happy path"
