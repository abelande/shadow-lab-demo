#!/usr/bin/env python3
"""Stage 4 tier backfill + tier-aware retirement analysis.

Wave 8.5-N companion script. Two phases:

  1. BACKFILL — collect every match's `proba` from `tiers-*.jsonl`,
     compute global percentile thresholds across all days, re-tier each
     match using those global thresholds. Recovers warmup-period
     matches that had `tier_pct: null`.

  2. ANALYSIS — join backfilled tiers with `shadow-*.jsonl` outcomes
     by `(pattern_id, entry_ts_ms)`, compute hit-rate per
     `(tier, pattern)` cell. Outputs a markdown report.

Usage
-----

    # Default paths (~/p6-v2/p6lab/artifacts/p6lab/outcomes/)
    python3 scripts/stage4_tier_backfill_analysis.py

    # Custom outcomes directory
    python3 scripts/stage4_tier_backfill_analysis.py \\
        --outcomes-dir /custom/path

    # Skip backfill if already done; analysis only
    python3 scripts/stage4_tier_backfill_analysis.py --analysis-only

    # Custom percentile thresholds
    python3 scripts/stage4_tier_backfill_analysis.py \\
        --thresholds '{"A_strict":0.995,"A_relaxed":0.99,"B":0.975,"C":0.95}'

Output
------

  - {outcomes_dir}/tiers-*-backfilled.jsonl per source tier file
  - {outcomes_dir}/stage4_tier_analysis_{stamp}.md (markdown report)
  - Console summary (tier distribution, tier × pattern hit-rate table)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger("stage4_backfill")

DEFAULT_THRESHOLDS = {
    "A_strict": 0.995,
    "A_relaxed": 0.99,
    "B": 0.975,
}
DEFAULT_OUTCOMES_DIR = Path.home() / "p6-v2" / "p6lab" / "artifacts" / "p6lab" / "outcomes"
MIN_SAMPLES_PER_CELL = 5  # tier × pattern cells with fewer samples are dropped from the table


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outcomes-dir", type=Path, default=DEFAULT_OUTCOMES_DIR,
                    help=f"Directory containing tiers-*.jsonl + shadow-*.jsonl files "
                         f"(default: {DEFAULT_OUTCOMES_DIR})")
    ap.add_argument("--thresholds", type=str, default=None,
                    help="JSON dict of percentile thresholds, e.g. "
                         '\'{"A_strict":0.995,"A_relaxed":0.99,"B":0.975}\'. '
                         "Defaults to 99.5/99/97.5 percentiles.")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_CELL,
                    help=f"Drop tier × pattern cells with fewer than N samples from the "
                         f"hit-rate table (default: {MIN_SAMPLES_PER_CELL}).")
    ap.add_argument("--analysis-only", action="store_true",
                    help="Skip backfill phase; assume tiers-*-backfilled.jsonl already exist.")
    ap.add_argument("--report-dir", type=Path, default=None,
                    help="Where to write the markdown report. Default: same as outcomes-dir.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress info-level console output.")
    return ap.parse_args()


def _load_thresholds(arg: str | None) -> dict[str, float]:
    """Parse user-supplied thresholds JSON or return defaults."""
    if arg is None:
        return DEFAULT_THRESHOLDS.copy()
    try:
        parsed = json.loads(arg)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--thresholds is not valid JSON: {e}")
    if not isinstance(parsed, dict):
        raise SystemExit(f"--thresholds must be a JSON object, got {type(parsed).__name__}")
    for name, val in parsed.items():
        if not isinstance(val, (int, float)) or not 0.0 < val < 1.0:
            raise SystemExit(f"--thresholds[{name}]={val!r} must be a float in (0, 1)")
    return parsed


# ---------------------------------------------------------------------------
# Phase 1 — backfill
# ---------------------------------------------------------------------------

def collect_global_probas(tier_files: list[Path]) -> list[float]:
    """Read every match's `proba` across all source tier files."""
    probas: list[float] = []
    for tf in tier_files:
        try:
            with open(tf) as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning("%s:%d skipping malformed JSON: %s", tf, line_no, e)
                        continue
                    if "proba" in row:
                        probas.append(float(row["proba"]))
        except OSError as e:
            logger.warning("could not read %s: %s", tf, e)
    return probas


def compute_thresholds(probas: list[float],
                        percentiles: dict[str, float]) -> dict[str, float]:
    """Map each tier name to a probability threshold from the global distribution."""
    if not probas:
        raise RuntimeError("no proba values found — cannot compute thresholds")
    arr = np.asarray(probas, dtype=np.float64)
    return {name: float(np.quantile(arr, pct)) for name, pct in percentiles.items()}


def assign_tier(proba: float, thresholds: dict[str, float]) -> str | None:
    """Return the strictest tier name the proba clears, or None below all."""
    # Sort tiers by descending threshold (strictest first)
    for name, thr in sorted(thresholds.items(), key=lambda kv: -kv[1]):
        if proba >= thr:
            return name
    return None


def backfill_tier_files(tier_files: list[Path],
                          thresholds: dict[str, float]) -> tuple[int, Counter]:
    """Re-tier every match using global thresholds; write `-backfilled.jsonl` siblings."""
    total_rows = 0
    tier_counter: Counter = Counter()
    for tf in tier_files:
        out_path = tf.with_name(tf.stem + "-backfilled.jsonl")
        with open(tf) as fin, open(out_path, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = float(row.get("proba", 0.0))
                tier = assign_tier(p, thresholds)
                row["tier_pct"] = tier
                tier_counter[tier or "below_all"] += 1
                total_rows += 1
                fout.write(json.dumps(row) + "\n")
        logger.info("backfilled %s → %s (%d rows)",
                    tf.name, out_path.name,
                    sum(1 for _ in open(out_path)))
    return total_rows, tier_counter


# ---------------------------------------------------------------------------
# Phase 2 — tier × pattern hit-rate analysis
# ---------------------------------------------------------------------------

def build_tier_index(backfilled_files: list[Path]) -> dict[tuple[str, int], str | None]:
    """Map (pattern_id, entry_ts_ms) → tier_pct."""
    idx: dict[tuple[str, int], str | None] = {}
    for bf in backfilled_files:
        with open(bf) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (str(row["pattern_id"]), int(row["entry_ts_ms"]))
                idx[key] = row.get("tier_pct")
    return idx


def aggregate_hit_rates(
    shadow_files: list[Path],
    tier_idx: dict[tuple[str, int], str | None],
) -> dict[str, dict[str, list[bool]]]:
    """Group `hit` booleans by (tier, pattern_id)."""
    table: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for sf in shadow_files:
        with open(sf) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    oc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (str(oc.get("pattern_id", "")), int(oc.get("entry_ts_ms", 0)))
                tier = tier_idx.get(key)
                if tier is None:
                    tier = "untiered"
                table[tier][str(oc.get("pattern_id", "?"))].append(bool(oc.get("hit", False)))
    return table


def render_table(
    table: dict[str, dict[str, list[bool]]],
    min_samples: int,
) -> str:
    """Render the tier × pattern hit-rate as a markdown table."""
    rows: list[tuple[str, str, int, float]] = []
    tier_order = ["A_strict", "A_relaxed", "B", "C", "untiered"]
    seen_tiers = set(table.keys())
    ordered_tiers = [t for t in tier_order if t in seen_tiers] + \
                    [t for t in seen_tiers if t not in tier_order]
    for tier in ordered_tiers:
        for pat, hits in sorted(table[tier].items()):
            if len(hits) < min_samples:
                continue
            wr = sum(hits) / len(hits)
            rows.append((tier, pat, len(hits), wr))
    lines = [
        f"| {'tier':<12} | {'pattern':<30} | {'n':>7} | {'hit_rate':>10} |",
        f"|{'-' * 14}|{'-' * 32}|{'-' * 9}|{'-' * 12}|",
    ]
    for tier, pat, n, wr in rows:
        lines.append(f"| {tier:<12} | {pat:<30} | {n:>7d} | {wr:>10.4f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(
    report_path: Path,
    *,
    outcomes_dir: Path,
    thresholds: dict[str, float],
    tier_counter: Counter,
    total_rows: int,
    table_md: str,
    n_tier_files: int,
    n_shadow_files: int,
) -> None:
    """Write a markdown report summarising the backfill + analysis."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Stage 4 — Tier Backfill + Hit-Rate Analysis",
        "",
        f"*Generated: {stamp}*",
        "",
        f"**Outcomes directory:** `{outcomes_dir}`",
        f"**Source tier files:** {n_tier_files}",
        f"**Source shadow files:** {n_shadow_files}",
        f"**Total matches re-tiered:** {total_rows:,}",
        "",
        "## Global percentile thresholds",
        "",
        f"| {'Tier':<12} | {'Threshold':>12} |",
        f"|{'-' * 14}|{'-' * 14}|",
    ]
    for name in sorted(thresholds.keys(), key=lambda k: -thresholds[k]):
        lines.append(f"| {name:<12} | {thresholds[name]:>12.6f} |")
    lines += [
        "",
        "## Tier distribution after backfill",
        "",
        f"| {'Tier':<12} | {'Count':>10} | {'%':>7} |",
        f"|{'-' * 14}|{'-' * 12}|{'-' * 9}|",
    ]
    total = sum(tier_counter.values()) or 1
    for tier, count in sorted(tier_counter.items(), key=lambda kv: -kv[1]):
        pct = 100 * count / total
        lines.append(f"| {tier:<12} | {count:>10,d} | {pct:>6.2f}% |")
    lines += [
        "",
        "## Hit-rate by (tier, pattern)",
        "",
        table_md,
        "",
        "## Interpretation",
        "",
        "- Compare `A_strict` and `A_relaxed` rows against `untiered` rows.",
        "  If A_strict's `hit_rate` is materially higher (≥ +0.10) than untiered's,",
        "  the percentile filter is separating signal from noise as designed.",
        "- A_strict row with `n` < 50 is statistically thin — interpret cautiously.",
        "- Patterns missing from A_strict but present in untiered fired enough to",
        "  be evaluated but not at top-tier confidence — those are pattern-level",
        "  misses; tag for review.",
        "- A row where `untiered` hit-rate ≈ 0.50 is a strong baseline (random);",
        "  any tier hit-rate above that is real lift.",
        "",
        "## Decision rules",
        "",
        "- If A_strict + A_relaxed hit-rates ≥ 0.65: percentile-tier serving is",
        "  defensible for paper-trade evaluation.",
        "- If A_strict hit-rate < 0.55: the model's top-percentile claim",
        "  did not survive out-of-sample. File as a Wave 9 research item.",
        "- If multiple patterns retired AND tier hit-rates show lift: Wave 8.5",
        "  Tier-2 closes with full validation. Composite rating 7.5 → 9.0.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    logger.info("wrote %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    outcomes_dir = args.outcomes_dir.expanduser().resolve()
    if not outcomes_dir.is_dir():
        raise SystemExit(f"--outcomes-dir does not exist: {outcomes_dir}")

    tier_files = sorted(p for p in outcomes_dir.glob("tiers-*.jsonl")
                        if "-backfilled" not in p.stem)
    shadow_files = sorted(outcomes_dir.glob("shadow-*.jsonl"))

    if not tier_files:
        raise SystemExit(
            f"no tiers-*.jsonl files in {outcomes_dir}; "
            "run Stage 2 with --percentile-tier-filter first"
        )
    if not shadow_files:
        raise SystemExit(
            f"no shadow-*.jsonl files in {outcomes_dir}; "
            "run Stage 2 (Wave 8.5-J) first"
        )

    logger.info("%d tier files, %d shadow files in %s",
                len(tier_files), len(shadow_files), outcomes_dir)

    percentiles = _load_thresholds(args.thresholds)
    logger.info("using percentiles: %s", percentiles)

    # Phase 1 — backfill
    if args.analysis_only:
        logger.info("--analysis-only set; skipping backfill")
        thresholds = {}
        tier_counter = Counter()
        total_rows = 0
    else:
        logger.info("phase 1 — backfill")
        probas = collect_global_probas(tier_files)
        if not probas:
            raise SystemExit("no proba values found across tier files")
        thresholds = compute_thresholds(probas, percentiles)
        logger.info("global thresholds (from %d matches):", len(probas))
        for name, thr in sorted(thresholds.items(), key=lambda kv: -kv[1]):
            logger.info("  %-12s = %.6f", name, thr)
        total_rows, tier_counter = backfill_tier_files(tier_files, thresholds)
        logger.info("backfilled %d rows across %d files", total_rows, len(tier_files))

    # Phase 2 — analysis
    logger.info("phase 2 — tier × pattern hit-rate")
    backfilled_files = sorted(outcomes_dir.glob("tiers-*-backfilled.jsonl"))
    if not backfilled_files:
        raise SystemExit(
            "no tiers-*-backfilled.jsonl files; "
            "run without --analysis-only first"
        )
    tier_idx = build_tier_index(backfilled_files)
    logger.info("loaded %d tier classifications", len(tier_idx))

    table = aggregate_hit_rates(shadow_files, tier_idx)
    table_md = render_table(table, min_samples=args.min_samples)

    # Console summary
    print()
    print("=" * 70)
    print("Tier × Pattern Hit-Rate")
    print("=" * 70)
    print(table_md)
    print()
    print("Tier distribution:")
    for tier, count in sorted(tier_counter.items(), key=lambda kv: -kv[1]):
        total = sum(tier_counter.values()) or 1
        print(f"  {tier:<12s}: {count:>7,d}  ({100*count/total:>5.2f}%)")

    # Write markdown report
    report_dir = (args.report_dir or outcomes_dir).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = report_dir / f"stage4_tier_analysis_{stamp}.md"
    write_report(
        report_path,
        outcomes_dir=outcomes_dir,
        thresholds=thresholds,
        tier_counter=tier_counter,
        total_rows=total_rows,
        table_md=table_md,
        n_tier_files=len(tier_files),
        n_shadow_files=len(shadow_files),
    )
    print(f"\nFull report → {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
