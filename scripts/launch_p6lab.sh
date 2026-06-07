#!/usr/bin/env bash
# Launch VS Code on .openclaw with p6lab notebooks pointed at a data file
# on a removable drive.
#
# Usage:
#   scripts/launch_p6lab.sh                       # uses DEFAULT_DATA_FILE below
#   scripts/launch_p6lab.sh /path/to/file.dbn.zst # one-off override
#   P6LAB_DATA_FILE=/path/... scripts/launch_p6lab.sh
#
# Env-var overrides are read by p6lab/notebooks/_common.py — see that file
# for the full list (P6LAB_DATA_FILE, P6LAB_SYMBOL, P6LAB_TICK_SIZE,
# P6LAB_MAX_SNAPSHOTS, P6LAB_MODE).

set -euo pipefail

# --- Edit this once for your usual drive path ---------------------------------
DEFAULT_DATA_FILE="/Volumes/BELTALK/p6data-0322-0421/nq/*dbn.zst"
# ------------------------------------------------------------------------------

DATA_FILE="${1:-${P6LAB_DATA_FILE:-$DEFAULT_DATA_FILE}}"

if [ ! -f "$DATA_FILE" ]; then
  echo "ERROR: data file not found: $DATA_FILE" >&2
  echo "  - Is the removable drive mounted?  ls /media/$USER" >&2
  echo "  - Pass a path:  $0 /path/to/file.dbn.zst" >&2
  echo "  - Or edit DEFAULT_DATA_FILE at the top of this script." >&2
  exit 1
fi

export P6LAB_DATA_FILE="$DATA_FILE"
export P6LAB_SYMBOL="${P6LAB_SYMBOL:-NQ}"
export P6LAB_TICK_SIZE="${P6LAB_TICK_SIZE:-0.25}"
export P6LAB_MAX_SNAPSHOTS="${P6LAB_MAX_SNAPSHOTS:-2000000}"
export P6LAB_MODE="${P6LAB_MODE:-replay}"

# Thread caps — tune to your CPU. Set to physical core count, not hyperthreads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "p6lab launch:"
echo "  data_file     = $P6LAB_DATA_FILE"
echo "  symbol        = $P6LAB_SYMBOL  (tick=$P6LAB_TICK_SIZE)"
echo "  max_snapshots = $P6LAB_MAX_SNAPSHOTS"
echo "  mode          = $P6LAB_MODE"
echo "  threads       = OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"
echo "  workspace     = $REPO_ROOT"

if pgrep -x code >/dev/null 2>&1; then
  echo
  echo "NOTE: VS Code is already running. New windows inherit env from THIS"
  echo "      shell, but existing windows keep the env they were launched"
  echo "      with. Restart any open notebook kernel after this opens, or"
  echo "      fully quit VS Code first if env vars don't take effect."
fi

cd "$REPO_ROOT"
exec code .
