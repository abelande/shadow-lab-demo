"""Integration test for the promote_pattern CLI.

Wave 4 Phase 1D gate: fixture parquet + CLI invocation → library.yaml
has the pattern with the expected status.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.fixture
def candidate_parquet(tmp_path: Path) -> Path:
    """Create a fixture mined_candidates.parquet with 2 clusters."""
    df = pd.DataFrame([
        {"cluster_id": 0, "member_count": 250,
         "hit_rate_5m": 0.62, "sharpe": 0.45, "mean_move_ticks": 2.0,
         "hit_rate_up": 0.62, "n": 250, "sharpe_proxy": 0.45},
        {"cluster_id": 1, "member_count": 180,
         "hit_rate_5m": 0.58, "sharpe": 0.35, "mean_move_ticks": -1.5,
         "hit_rate_up": 0.42, "n": 180, "sharpe_proxy": 0.35},
    ])
    path = tmp_path / "mined_candidates.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def library_path(tmp_path: Path) -> Path:
    return tmp_path / "library.yaml"


_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "promote_pattern.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True, text=True, check=False,
    )


class TestPromotePatternCLI:
    def test_promotes_cluster_to_mined_approved(self, candidate_parquet: Path,
                                                  library_path: Path) -> None:
        result = _run(
            "--library", str(library_path),
            "--candidate", str(candidate_parquet),
            "--cluster-id", "0",
            "--name", "bid_heavy_burst",
            "--status", "mined_approved",
        )
        assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"
        assert library_path.exists()
        data = yaml.safe_load(library_path.read_text())
        assert "bid_heavy_burst" in data["patterns"]
        assert data["patterns"]["bid_heavy_burst"]["status"] == "mined_approved"
        assert data["patterns"]["bid_heavy_burst"]["instruments"] == ["NQ"]

    def test_dry_run_does_not_write(self, candidate_parquet: Path,
                                      library_path: Path) -> None:
        result = _run(
            "--library", str(library_path),
            "--candidate", str(candidate_parquet),
            "--cluster-id", "0",
            "--name", "dry_run_check",
            "--dry-run",
        )
        assert result.returncode == 0
        assert not library_path.exists()

    def test_missing_cluster_id_errors(self, candidate_parquet: Path,
                                         library_path: Path) -> None:
        result = _run(
            "--library", str(library_path),
            "--candidate", str(candidate_parquet),
            "--cluster-id", "999",
            "--name", "anything",
        )
        assert result.returncode == 2
        assert "not found" in result.stderr.lower()

    def test_missing_parquet_errors(self, tmp_path: Path,
                                      library_path: Path) -> None:
        result = _run(
            "--library", str(library_path),
            "--candidate", str(tmp_path / "nonexistent.parquet"),
            "--cluster-id", "0",
            "--name", "x",
        )
        assert result.returncode == 2
