#!/usr/bin/env bash
# p6lab_env.sh — switchable env profiles for NB06 / NB07 / live runner
#
# Usage:
#   source p6lab/scripts/p6lab_env.sh <profile>
#   source p6lab/scripts/p6lab_env.sh list
#
# Sets:
#   P6LAB_DATA_FILE        — single path or comma-separated list
#   P6LAB_SYMBOL           — instrument code (NQ / ES / CL / GC / SI)
#   OMP_NUM_THREADS        — also MKL / OpenBLAS / NumExpr
#
# Notes:
#   * Source — don't execute. Sourcing exports vars into the current shell so
#     a Jupyter / VS Code launched from THIS shell inherits them. Thread-count
#     vars are read at import time by numpy/lightgbm; changing them after a
#     kernel is already running is too late — restart the kernel first.
#   * Override the data dir via:
#       P6LAB_DATA_DIR=/some/path source p6lab/scripts/p6lab_env.sh nq-day

PROFILE="${1:-list}"

: "${P6LAB_DATA_DIR:=/Volumes/BELTALK/p6-data-0322-0421/nq}"

__p6_set_threads() {
    export OMP_NUM_THREADS="$1"
    export MKL_NUM_THREADS="$1"
    export OPENBLAS_NUM_THREADS="$1"
    export NUMEXPR_NUM_THREADS="$1"
}

__nq_day_file() {
    local d="$1"             # e.g., 20260323
    local next                # 20260324
    next=$(date -j -v+1d -f "%Y%m%d" "$d" "+%Y%m%d" 2>/dev/null \
        || date -d "$d +1 day" "+%Y%m%d")
    echo "${P6LAB_DATA_DIR}/nq-mbo-${d}T0000Z-${next}T0000Z.dbn.zst"
}

__nq_range_files() {
    # Comma-separated zulu-window filenames from start (incl) to end (incl).
    # Capture date increment into a temp var so a failed BSD attempt doesn't
    # wipe cur before the GNU fallback runs (else `date -d "" +1 day` ≈ today).
    local start="$1" end="$2" cur next files=""
    cur="$start"
    while [[ "$cur" -le "$end" ]]; do
        [[ -n "$files" ]] && files="${files},"
        files="${files}$(__nq_day_file "$cur")"
        if next=$(date -j -v+1d -f "%Y%m%d" "$cur" "+%Y%m%d" 2>/dev/null); then
            cur="$next"
        else
            cur=$(date -d "$cur +1 day" "+%Y%m%d")
        fi
    done
    echo "$files"
}

case "$PROFILE" in
    # ─── NQ — single-day profiles (zulu-window naming via __nq_day_file) ──
    # Pick any specific calendar date with: source ... nq-on YYYYMMDD
    nq-on)
        if [[ -z "${2:-}" ]]; then
            echo "Usage: source p6lab/scripts/p6lab_env.sh nq-on YYYYMMDD" >&2
            return 1 2>/dev/null || exit 1
        fi
        export P6LAB_DATA_FILE="$(__nq_day_file "$2")"
        export P6LAB_SYMBOL="NQ"
        __p6_set_threads 1
        ;;
    nq-mar23) export P6LAB_DATA_FILE="$(__nq_day_file 20260323)"; export P6LAB_SYMBOL="NQ"; __p6_set_threads 1 ;;
    nq-mar24) export P6LAB_DATA_FILE="$(__nq_day_file 20260324)"; export P6LAB_SYMBOL="NQ"; __p6_set_threads 1 ;;
    nq-mar25) export P6LAB_DATA_FILE="$(__nq_day_file 20260325)"; export P6LAB_SYMBOL="NQ"; __p6_set_threads 1 ;;
    nq-mar26) export P6LAB_DATA_FILE="$(__nq_day_file 20260326)"; export P6LAB_SYMBOL="NQ"; __p6_set_threads 1 ;;
    nq-mar27) export P6LAB_DATA_FILE="$(__nq_day_file 20260327)"; export P6LAB_SYMBOL="NQ"; __p6_set_threads 1 ;;
    nq-day)   export P6LAB_DATA_FILE="$(__nq_day_file 20260325)"; export P6LAB_SYMBOL="NQ"; __p6_set_threads 1 ;;
    nq-day-fast)
        export P6LAB_DATA_FILE="$(__nq_day_file 20260325)"
        export P6LAB_SYMBOL="NQ"
        __p6_set_threads 4
        ;;

    # ─── NQ — multi-day profiles for the diagnostic rerun ─────────────────
    nq-2day)
        export P6LAB_DATA_FILE="$(__nq_day_file 20260325),$(__nq_day_file 20260326)"
        export P6LAB_SYMBOL="NQ"
        __p6_set_threads 1
        ;;
    nq-2day-fast)
        export P6LAB_DATA_FILE="$(__nq_day_file 20260325),$(__nq_day_file 20260326)"
        export P6LAB_SYMBOL="NQ"
        __p6_set_threads 4
        ;;
    nq-3day)
        export P6LAB_DATA_FILE="$(__nq_day_file 20260324),$(__nq_day_file 20260325),$(__nq_day_file 20260326)"
        export P6LAB_SYMBOL="NQ"
        __p6_set_threads 1
        ;;
    nq-week-mar23)
        export P6LAB_DATA_FILE="$(__nq_day_file 20260323),$(__nq_day_file 20260324),$(__nq_day_file 20260325),$(__nq_day_file 20260326),$(__nq_day_file 20260327)"
        export P6LAB_SYMBOL="NQ"
        __p6_set_threads 1
        ;;
    # Range form (inclusive start..end). Usage: source ... nq-range 20260323 20260327
    nq-range)
        if [[ -z "${2:-}" || -z "${3:-}" ]]; then
            echo "Usage: source p6lab/scripts/p6lab_env.sh nq-range YYYYMMDD YYYYMMDD" >&2
            return 1 2>/dev/null || exit 1
        fi
        export P6LAB_DATA_FILE="$(__nq_range_files "$2" "$3")"
        export P6LAB_SYMBOL="NQ"
        __p6_set_threads 1
        ;;

    # ─── NQ — original sample / special files ─────────────────────────────
    nq-15min)
        export P6LAB_DATA_FILE="${P6LAB_DATA_DIR}/nq-mbo-sample-15min.dbn.zst"
        export P6LAB_SYMBOL="NQ"
        __p6_set_threads 1
        ;;
    nq-april)
        # Two-file April slice (irregular time windows; not standard 24h).
        export P6LAB_DATA_FILE="${P6LAB_DATA_DIR}/nq-mbo-20260412T2200Z-20260413T1613Z.dbn.zst,${P6LAB_DATA_DIR}/nq-mbo-20260413T0000Z-20260414T0000Z.dbn.zst"
        export P6LAB_SYMBOL="NQ"
        __p6_set_threads 1
        ;;

    # ─── Other instruments ────────────────────────────────────────────────
    es-day)
        export P6LAB_DATA_FILE="${P6LAB_DATA_DIR}/es-mbo-2026-03-27.dbn.zst"
        export P6LAB_SYMBOL="ES"
        __p6_set_threads 1
        ;;
    cl-day)
        export P6LAB_DATA_FILE="${P6LAB_DATA_DIR}/cl-mbo-2026-03-27.dbn.zst"
        export P6LAB_SYMBOL="CL"
        __p6_set_threads 1
        ;;
    gc-day)
        export P6LAB_DATA_FILE="${P6LAB_DATA_DIR}/gc-mbo-2026-03-27.dbn.zst"
        export P6LAB_SYMBOL="GC"
        __p6_set_threads 1
        ;;
    si-day)
        export P6LAB_DATA_FILE="${P6LAB_DATA_DIR}/si-mbo-2026-03-27.dbn.zst"
        export P6LAB_SYMBOL="SI"
        __p6_set_threads 1
        ;;

    # ─── Help / list ──────────────────────────────────────────────────────
    list|help|""|--help|-h)
        cat <<EOF
p6lab env profiles — source this script with a profile name

  Usage:  source p6lab/scripts/p6lab_env.sh <profile>

  NQ training profiles (zulu-window naming derived via __nq_day_file):
    nq-on YYYYMMDD     any single calendar day (1 thread)
    nq-mar23..27       single named days (1 thread each)
    nq-day             alias for nq-mar25 (default Mac-safe)
    nq-day-fast        nq-mar25 with 4 threads (Linux/VPS only)
    nq-2day            mar25 + mar26 (1 thread)
    nq-2day-fast       mar25 + mar26 (4 threads)
    nq-3day            mar24 + mar25 + mar26 (1 thread)
    nq-week-mar23      mar23..mar27 (5 days, 1 thread)
    nq-range S E       inclusive range — e.g. nq-range 20260323 20260331
    nq-15min           15-min smoke sample (1 thread)
    nq-april           irregular April slice (1 thread)

  Other instruments (single day each):
    es-day          ES 2026-03-27
    cl-day          CL 2026-03-27
    gc-day          GC 2026-03-27
    si-day          SI 2026-03-27

  list            show this help

  Override data dir (e.g. on Mac with BELTALK drive):
    P6LAB_DATA_DIR=/Volumes/BELTALK/p6-data-0322-0421/nq \\
      source p6lab/scripts/p6lab_env.sh nq-day

  After sourcing, launch jupyter / code from the SAME shell.
EOF
        unset -f __p6_set_threads 2>/dev/null
        return 0 2>/dev/null || exit 0
        ;;
    *)
        echo "Unknown profile: $PROFILE" >&2
        echo "Run: source p6lab/scripts/p6lab_env.sh list" >&2
        unset -f __p6_set_threads 2>/dev/null
        return 1 2>/dev/null || exit 1
        ;;
esac

# ─── Verify file(s) exist ────────────────────────────────────────────────
if [[ "${P6LAB_DATA_FILE:-}" == *","* ]]; then
    IFS=',' read -ra _P6_FILES <<< "$P6LAB_DATA_FILE"
    for _f in "${_P6_FILES[@]}"; do
        [[ -f "$_f" ]] || echo "WARN: missing data file: $_f" >&2
    done
    unset _P6_FILES _f
elif [[ -n "${P6LAB_DATA_FILE:-}" ]] && [[ ! -f "$P6LAB_DATA_FILE" ]]; then
    echo "WARN: missing data file: $P6LAB_DATA_FILE" >&2
fi

# ─── Print summary ───────────────────────────────────────────────────────
echo "P6LAB env set ($PROFILE):"
echo "  P6LAB_DATA_FILE      = $P6LAB_DATA_FILE"
echo "  P6LAB_SYMBOL         = $P6LAB_SYMBOL"
echo "  OMP_NUM_THREADS      = $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS      = $MKL_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS = $OPENBLAS_NUM_THREADS"

unset -f __p6_set_threads 2>/dev/null
