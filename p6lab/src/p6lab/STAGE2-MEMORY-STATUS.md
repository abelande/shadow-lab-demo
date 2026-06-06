# Stage 2 Status Reference & Memory for Future Runs

*Quick-reference command catalogue for monitoring the overnight Stage 2 run
from any terminal session. Each command is self-contained — no shell vars
required. Commands assume the project root is `~/p6-v2/p6lab/`.*

---

## 1. Is the Stage 2 loop alive?

### Quickest check

```bash
PID=$(cat ~/p6-v2/p6lab/logs/stage2.pid 2>/dev/null) && \
  kill -0 "$PID" 2>/dev/null && \
  echo "RUNNING (PID $PID)" || echo "NOT RUNNING"
```

### Detailed process info

```bash
# Show the bash -c parent + any python3 children spawned by the loop
PID=$(cat ~/p6-v2/p6lab/logs/stage2.pid 2>/dev/null)
ps -p "$PID" -o pid,etime,pcpu,pmem,command 2>/dev/null
echo "--- python children ---"
pgrep -f "run_live.py" | xargs -I{} ps -p {} -o pid,etime,pcpu,pmem,command 2>/dev/null
```

### Search by command line (works even without PID file)

```bash
ps aux | grep -E "run_live|stage2-loop" | grep -v grep
```

---

## 2. Latest log file commands

### Tail the most recent Stage 2 log

```bash
LATEST_LOG=$(ls -t ~/p6-v2/p6lab/logs/stage2-*.log 2>/dev/null | head -1)
echo "Tailing: $LATEST_LOG"
tail -f "$LATEST_LOG"
# Press Ctrl+C to stop tailing — the script keeps running
```

### Just the last 30 lines (no follow)

```bash
LATEST_LOG=$(ls -t ~/p6-v2/p6lab/logs/stage2-*.log 2>/dev/null | head -1)
tail -30 "$LATEST_LOG"
```

### Search the latest log for specific events

```bash
LATEST_LOG=$(ls -t ~/p6-v2/p6lab/logs/stage2-*.log 2>/dev/null | head -1)

# Tier filter activations (one per day at startup)
grep "tier filter active" "$LATEST_LOG"

# Day boundaries
grep -E "START|END|SKIP|COMPLETE" "$LATEST_LOG"

# Errors / tracebacks
grep -iE "error|traceback|exception|warning" "$LATEST_LOG"

# Per-day stats summaries
grep -E "snapshots_ingested|matches_emitted" "$LATEST_LOG" | tail -10
```

### Check the most recent diagnostic test run

```bash
LATEST_DIAG=$(ls -t ~/p6-v2/p6lab/artifacts/p6lab/diagnostics/diag-*-runner.log 2>/dev/null | head -1)
echo "Latest diagnostic: $LATEST_DIAG"
grep "tier filter active" "$LATEST_DIAG"
grep -A 8 "live runner done" "$LATEST_DIAG"
```

---

## 3. Days completed and outcomes progress

### Day count

```bash
ls ~/p6-v2/p6lab/artifacts/p6lab/outcomes/shadow-*.jsonl 2>/dev/null | wc -l
# Expect: 30 when complete
```

### Total outcomes recorded across all days

```bash
wc -l ~/p6-v2/p6lab/artifacts/p6lab/outcomes/shadow-*.jsonl 2>/dev/null | tail -1
```

### Side-car tier files (Option A — for backfill)

```bash
ls -lh ~/p6-v2/p6lab/artifacts/p6lab/outcomes/tiers-*.jsonl 2>/dev/null
wc -l ~/p6-v2/p6lab/artifacts/p6lab/outcomes/tiers-*.jsonl 2>/dev/null | tail -1
```

### Per-day breakdown

```bash
for f in ~/p6-v2/p6lab/artifacts/p6lab/outcomes/shadow-*.jsonl; do
  day=$(basename "$f" .jsonl | sed 's/shadow-//')
  count=$(wc -l < "$f")
  size=$(ls -lh "$f" | awk '{print $5}')
  echo "$day: $count outcomes ($size)"
done
```

---

## 4. Memory and system health

### Current memory pressure

```bash
top -l 1 | grep PhysMem
memory_pressure -Q
```

### Top 10 memory consumers right now

```bash
top -l 1 -o mem -n 10
```

### Is BELTALK still mounted?

```bash
mount | grep -i beltalk
ls /Volumes/BELTALK/p6-data-0322-0421/nq/ 2>/dev/null | head -3
```

### CPU activity (caffeinate working?)

```bash
ps -ef | grep caffeinate | grep -v grep
pmset -g | grep -E "sleep|disksleep"
```

---

## 5. All-in-one status snapshot

Save this whole block as `~/p6-v2/p6lab/check-stage2.sh` for one-command checks:

```bash
cat > ~/p6-v2/p6lab/check-stage2.sh << 'EOF'
#!/bin/bash
echo "============================================="
echo "  p6lab Stage 2 status @ $(date)"
echo "============================================="
echo ""
echo "[Process]"
PID=$(cat ~/p6-v2/p6lab/logs/stage2.pid 2>/dev/null)
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  echo "  RUNNING (PID $PID)"
  ps -p "$PID" -o etime,pcpu,pmem,command 2>/dev/null | tail -1
else
  echo "  NOT RUNNING (last PID was: $PID)"
fi
echo ""
echo "[Python subprocess]"
pgrep -f "run_live.py" | head -3 | while read p; do
  ps -p "$p" -o pid,etime,pcpu,pmem,command 2>/dev/null | tail -1
done
echo ""
echo "[Latest log]"
LATEST_LOG=$(ls -t ~/p6-v2/p6lab/logs/stage2-*.log 2>/dev/null | head -1)
echo "  $LATEST_LOG"
tail -8 "$LATEST_LOG" 2>/dev/null
echo ""
echo "[Progress]"
SHADOW_COUNT=$(ls ~/p6-v2/p6lab/artifacts/p6lab/outcomes/shadow-*.jsonl 2>/dev/null | wc -l | tr -d ' ')
TIER_COUNT=$(ls ~/p6-v2/p6lab/artifacts/p6lab/outcomes/tiers-*.jsonl 2>/dev/null | wc -l | tr -d ' ')
TOTAL_OUTCOMES=$(wc -l ~/p6-v2/p6lab/artifacts/p6lab/outcomes/shadow-*.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
echo "  Shadow files:  $SHADOW_COUNT / 30"
echo "  Tier files:    $TIER_COUNT / 30"
echo "  Total outcomes: ${TOTAL_OUTCOMES:-0}"
echo ""
echo "[Memory]"
top -l 1 | grep PhysMem
memory_pressure -Q | tail -1
echo ""
echo "[Drive]"
mount | grep -i beltalk | head -1 || echo "  WARNING: BELTALK not mounted!"
echo ""
echo "============================================="
EOF
chmod +x ~/p6-v2/p6lab/check-stage2.sh
```

Then anytime, in any terminal:

```bash
bash ~/p6-v2/p6lab/check-stage2.sh
```

---

## 6. Python diagnostic snippets (deeper analysis)

### Tier distribution across all days (real-time as Stage 2 progresses)

```bash
python3 << 'EOF'
import json, glob
from collections import Counter
tiers = Counter()
total = 0
for p in sorted(glob.glob('/Users/abelkifle/p6-v2/p6lab/artifacts/p6lab/outcomes/tiers-*.jsonl')):
    for line in open(p):
        row = json.loads(line)
        t = row.get('tier_pct') or 'untiered_warmup'
        tiers[t] += 1
        total += 1
print(f"Total tagged matches: {total}")
for t, c in sorted(tiers.items(), key=lambda x: -x[1]):
    pct = 100 * c / total if total else 0
    print(f"  {t:30s}: {c:7d}  ({pct:5.1f}%)")
EOF
```

### Outcome win-rate by tier (only meaningful after backfill)

```bash
python3 << 'EOF'
import json, glob
from collections import defaultdict

# Build tier index from side-car files
tier_idx = {}
for p in sorted(glob.glob('/Users/abelkifle/p6-v2/p6lab/artifacts/p6lab/outcomes/tiers-*.jsonl')):
    for line in open(p):
        row = json.loads(line)
        key = (row['pattern_id'], row['entry_ts_ms'])
        tier_idx[key] = (row.get('tier_pct'), row.get('proba'))

# Iterate outcomes, join by (pattern_id, entry_ts_ms)
by_tier = defaultdict(list)
for p in sorted(glob.glob('/Users/abelkifle/p6-v2/p6lab/artifacts/p6lab/outcomes/shadow-*.jsonl')):
    for line in open(p):
        outcome = json.loads(line)
        key = (outcome['pattern_id'], outcome['entry_ts_ms'])
        tier, _ = tier_idx.get(key, (None, None))
        by_tier[tier or 'untiered'].append(outcome['hit'])

print(f"{'tier':30s} {'n':>8} {'hit_rate':>10}")
print("-" * 50)
for tier, hits in sorted(by_tier.items(), key=lambda x: -len(x[1])):
    n = len(hits)
    hit_rate = sum(hits) / n if n else 0
    print(f"{tier:30s} {n:>8d} {hit_rate:>10.4f}")
EOF
```

### Memory snapshot history (if you've been monitoring)

```bash
python3 << 'EOF'
import re
from pathlib import Path
log = sorted(Path('/Users/abelkifle/p6-v2/p6lab/logs/').glob('stage2-*.log'))[-1]
print(f"Reading: {log}")
last_unused = None
peaks = []
for line in open(log):
    m = re.search(r'PhysMem:\s+(\d+)G used.*?(\d+)M unused', line)
    if m:
        unused_mb = int(m.group(2))
        peaks.append(unused_mb)
print(f"Memory unused (MB) across {len(peaks)} samples:")
print(f"  min: {min(peaks) if peaks else 'n/a'} MB")
print(f"  max: {max(peaks) if peaks else 'n/a'} MB")
print(f"  current: {peaks[-1] if peaks else 'n/a'} MB")
EOF
```

---

## 7. Tomorrow morning post-Stage-2 checklist

```bash
# 1. Did it complete?
LATEST_LOG=$(ls -t ~/p6-v2/p6lab/logs/stage2-*.log | head -1)
grep "STAGE 2 COMPLETE" "$LATEST_LOG" && echo "DONE ✅" || echo "INCOMPLETE — re-run loop will resume"

# 2. Day count
echo "Days completed: $(ls ~/p6-v2/p6lab/artifacts/p6lab/outcomes/shadow-*.jsonl 2>/dev/null | wc -l) / 30"

# 3. Any subprocess crashes?
grep "exit=" "$LATEST_LOG" | grep -v "exit=0" | head

# 4. Any new native crashes?
ls -lt ~/Library/Logs/DiagnosticReports/Python* 2>/dev/null | head -3

# 5. Total tier-tagged matches
wc -l ~/p6-v2/p6lab/artifacts/p6lab/outcomes/tiers-*.jsonl 2>/dev/null | tail -1

# 6. Backfill warmup periods (Approach 1 from PercentileTierFilter docs)
python3 << 'EOF'
import json, glob, numpy as np
from pathlib import Path

all_probas = []
tier_files = sorted(glob.glob("/Users/abelkifle/p6-v2/p6lab/artifacts/p6lab/outcomes/tiers-*.jsonl"))
for tf in tier_files:
    for line in open(tf):
        row = json.loads(line)
        all_probas.append(row['proba'])

print(f"Total matches across 30 days: {len(all_probas)}")
thresholds = {
    'A_strict':  float(np.quantile(all_probas, 0.995)),
    'A_relaxed': float(np.quantile(all_probas, 0.99)),
    'B':         float(np.quantile(all_probas, 0.975)),
}
print("Global thresholds:", thresholds)

for tf in tier_files:
    out_path = Path(tf).with_name(Path(tf).stem + "-backfilled.jsonl")
    with open(out_path, "w") as out:
        for line in open(tf):
            row = json.loads(line)
            p = row['proba']
            if p >= thresholds['A_strict']:    row['tier_pct'] = 'A_strict'
            elif p >= thresholds['A_relaxed']: row['tier_pct'] = 'A_relaxed'
            elif p >= thresholds['B']:         row['tier_pct'] = 'B'
            else:                              row['tier_pct'] = None
            out.write(json.dumps(row) + "\n")
print(f"Wrote {len(tier_files)} backfilled tier files")
EOF

# 7. Reset macOS settings
sudo pmset -a disksleep 10
sudo pmset -a sleep 10
ps -ef | grep caffeinate | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null
```

---

## 8. Recovery — if something went wrong overnight

### Loop crashed but most days completed

The loop's resume logic skips days that already have a non-empty outcome file. Just relaunch:

```bash
cd ~/p6-v2/p6lab
nohup bash /tmp/stage2-loop.sh > /tmp/stage2-nohup.out 2>&1 &
echo $! > ~/p6-v2/p6lab/logs/stage2.pid
```

### Outcome files exist but tier side-car files are missing or partial

The Stage 4 backfill recovers from the proba values you have. If tiers files are completely missing for some days, those days must be re-run (delete shadow file and let resume logic re-process):

```bash
# Identify days with shadow but no tiers
for shadow in ~/p6-v2/p6lab/artifacts/p6lab/outcomes/shadow-*.jsonl; do
  day=$(basename "$shadow" .jsonl | sed 's/shadow-//')
  tier_file="${shadow/shadow-/tiers-}"
  if [[ ! -f "$tier_file" ]]; then
    echo "MISSING tiers for: $day"
    # Optionally remove shadow to trigger re-processing
    # rm "$shadow"
  fi
done
```

### BELTALK unmounted mid-run

Replug, then relaunch loop. Resume logic skips completed days.

```bash
# Verify drive is back
mount | grep -i beltalk

# Resume
nohup bash /tmp/stage2-loop.sh > /tmp/stage2-nohup.out 2>&1 &
echo $! > ~/p6-v2/p6lab/logs/stage2.pid
```

---

## Reference paths

| Path | Purpose |
|---|---|
| `~/p6-v2/p6lab/logs/stage2-*.log` | Per-Stage-2-launch consolidated log |
| `~/p6-v2/p6lab/logs/stage2.pid` | Current Stage 2 PID file |
| `~/p6-v2/p6lab/artifacts/p6lab/outcomes/shadow-*.jsonl` | Per-day closed outcomes (one row per match) |
| `~/p6-v2/p6lab/artifacts/p6lab/outcomes/tiers-*.jsonl` | Per-day side-car tier classifications |
| `~/p6-v2/p6lab/artifacts/p6lab/outcomes/audit-*.jsonl` | Per-day raw match audit |
| `~/p6-v2/p6lab/artifacts/p6lab/diagnostics/diag-*` | Pre-Stage-2 diagnostic test runs |
| `/tmp/stage2-loop.sh` | Cached loop script |
| `/tmp/stage2-nohup.out` | nohup wrapper stdout/stderr |
***
# Hardware & VPS Sizing for p6lab — Research Brief

## Context

Your machine (M4 16 GB) is hitting **8.3 GB swap**, **~60 MB free RAM**, and 38 Chrome processes — RAM-saturated. You want to know what it actually takes to run p6lab notebooks with `max_snapshots=None` on full-session Databento MBO files in a "live trading research" context, what it costs on Vultr (and cheaper alternatives), and which portable laptops can do it.

This is a research deliverable, not a code change. Findings below are grounded in your actual project profile (CPU-bound replay/backtest engine, optional GPU, large multi-month parquet artifacts) — not generic specs.

---

## What your project actually needs

From the codebase scan:
- **`max_snapshots`** caps the in-memory `OrderBookSnapshot` list. Default = 500K (~585 MB). Setting to `None` switches to a streaming/sliding-window read of the full `.dbn.zst`, peaking around **400–600 MB per session** — not the linear blow-up you might fear. The real memory pressure comes from **multi-month / multi-instrument workloads**, not a single overnight file.
- **CPU-bound** on pattern matching (HDBSCAN, template matcher, correlation engine <50 ms target) and ML training (LightGBM/XGBoost/CatBoost in NB06).
- **I/O-bound** on input — sequential zstd decompression at ~5–20 MB/s. NVMe matters.
- **GPU optional** — only the BSV CNN autoencoder uses PyTorch, guarded by `HAS_TORCH`. CPU is the default path.
- **Heavy artifacts**: triple_view parquets, execution sims, cascade logs, feature logs accumulate to ~1.1 GB+ per full backtest run.

---

## 1. Three compute tiers (laptop **or** VPS sizing)

| Tier | RAM | CPU | GPU | Storage | What you can do |
|---|---|---|---|---|---|
| 🟥 **Just cutting it** | **16 GB** | 4–6 perf cores | none | 512 GB NVMe | Single-day replay, one instrument, NB01–NB02. Will swap once you open Chrome + VS Code + a notebook. **This is what you have now.** |
| 🟨 **Comfortable** | **32 GB** | 8 perf cores (or 8P+4E) | none / iGPU | 1 TB NVMe | 1–3 month backtests, NB04 pattern mining on full data, single-instrument live correlation engine alongside browser/IDE without swap. **The sweet spot for p6lab.** |
| 🟩 **Full ease** | **64 GB** | 12–16 perf cores | optional 8 GB+ dGPU (only if you turn on the BSV CNN) | 2 TB NVMe | 12-month cascade backtests across 4+ instruments in parallel, multiple kernels open, model training without swap. Headroom for everything else you do. |

**Note on "unlimited max_snapshots":** in p6lab this is bounded by the file size, not your RAM, because the engine streams. 32 GB handles the largest single-session case with 90% headroom. The 64 GB tier is justified by *parallel* sessions and multi-month aggregation, not by raising `max_snapshots`.

---

## 2. VPS pricing — traffic-light tiers

### Vultr (your reference point)

| Light | Plan family | Spec | ~Monthly |
|---|---|---|---|
| 🟥 Cheapest viable | **Cloud Compute (regular)** | 4 vCPU / 16 GB / 320 GB NVMe | ~$80 |
| 🟨 Recommended | **High Frequency** (3 GHz+ Xeon, NVMe local) | 8 vCPU / 32 GB / 512 GB NVMe | ~$192 |
| 🟩 Power | **Optimized Cloud Compute** (dedicated AMD, no noisy neighbors) | 16 vCPU / 64 GB / 800 GB NVMe | ~$640 |

High Frequency is the right Vultr tier for p6lab because pattern matching is single-thread-sensitive. Avoid the cheap shared Cloud Compute for the correlation engine — variable clock speeds break the <50 ms latency target.

### Cheaper competitors (same memory tier)

| Provider | Comparable plan | Spec | ~Monthly | Notes |
|---|---|---|---|---|
| 🟢 **Hetzner Cloud** (CCX/CPX) | CCX33 dedicated AMD | 8 vCPU / 32 GB / 240 GB NVMe | **~$60–70** | Best price/spec on the market. EU + US (Ashburn/Hillsboro) regions. |
| 🟢 **OVHcloud** | VPS Elite | 8 vCPU / 32 GB / 640 GB NVMe | **~$70–90** | Solid NVMe, generous bandwidth, weaker UI. |
| 🟡 **Contabo** | Cloud VPS L | 8 vCPU / 32 GB / 800 GB NVMe | **~$25–35** | Cheapest by far, but **shared CPU** — clock speed dips will violate the engine's <50 ms target. Use only for batch backtests, not live correlation. |
| 🟡 **DigitalOcean / Linode** | CPU-Optimized 32 GB | 8 vCPU / 32 GB / 400 GB SSD | ~$240 | Premium-priced like Vultr's Optimized tier. Best docs/UX, marginal performance edge. |

**Bottom line:** for p6lab specifically, **Hetzner CCX33 (~$65/mo)** gives you Vultr-High-Frequency-class performance at ~⅓ the price. Contabo is a trap for the live correlation engine but fine for nightly batch backtests.

### Tiered VPS recommendation

| Tier | Provider/Plan | ~Monthly |
|---|---|---|
| 🟥 Just cutting it | Hetzner CCX23 (4 vCPU / 16 GB / 160 GB NVMe) | ~$30 |
| 🟨 Recommended | **Hetzner CCX33 (8 vCPU / 32 GB / 240 GB NVMe)** | **~$65** |
| 🟩 Full ease | Hetzner CCX53 (16 vCPU / 64 GB / 600 GB NVMe) | ~$240 |

---

## 3. Portable laptops (top 5 brands, 32 GB+, can sustain p6lab workloads)

**Market context:** DRAM is in shortage in 2026; laptop prices are rising. Configure now if you're going to.

| Tier | Brand | Model | Config | ~Price |
|---|---|---|---|---|
| 🟨 Comfortable (32 GB) | **Apple** | MacBook Pro 14" M4 Pro | 32 GB / 1 TB | ~$2,400 |
| 🟨 Comfortable (32 GB) | **Lenovo** | ThinkPad X1 Carbon Gen 13 | 32 GB / 1 TB | ~$2,000 |
| 🟨 Comfortable (32 GB) | **Lenovo** | Yoga Pro 9i 16 Gen 10 | 32 GB / 1 TB | ~$1,500 (cheapest credible 32 GB option) |
| 🟨 Comfortable (32 GB) | **Dell** | XPS 14 / 16 (2026 refresh) | 32 GB / 1 TB | ~$2,100 |
| 🟨 Comfortable (32 GB) | **ASUS** | Zenbook S 16 (Ryzen AI 9 HX 370) | 32 GB / 1 TB | ~$1,900 |
| 🟨 Comfortable (32 GB) | **HP** | OmniBook Ultra / EliteBook | 32 GB / 1 TB | ~$1,800 |
| 🟩 Full ease (48–64 GB) | **Apple** | MacBook Pro 16" M5 Pro | 48 GB / 1 TB | ~$2,900 |
| 🟩 Full ease (64 GB) | **Apple** | MacBook Pro 16" M5 Max | 64 GB / 1 TB | ~$3,700 |
| 🟩 Full ease (64 GB) | **Lenovo** | ThinkPad P1 Gen 8 (mobile workstation) | 64 GB / 2 TB | ~$3,200 |
| 🟩 Full ease (64 GB) | **Dell** | Precision 5690 | 64 GB / 2 TB | ~$3,400 |

**For p6lab specifically:**
- Apple Silicon's unified memory + fast NVMe makes a **MacBook Pro 14" M4 Pro / 32 GB** the best portable match — pandas/polars/zstd/pyarrow all run well on arm64 and the M-series sustains long compute without thermal throttling.
- A **ThinkPad P1 / Precision 5690 with 64 GB** is the best Windows/Linux option if you want a CUDA path open for the BSV CNN later.

---

## 4. Practical recommendation

Given p6lab's actual profile:

1. **If buying once for ~5 years:** MacBook Pro 16" M5 Pro 48 GB (~$2,900). 50% headroom over the comfortable tier, runs the full notebook suite + browser + Cursor without swap, doesn't lock you out of multi-month backtests later.
2. **If staying on 16 GB and offloading heavy work:** rent a **Hetzner CCX33 (~$65/mo)** for backtests/notebook kernels and SSH/VS Code Remote into it. ~$780/yr ≈ ¼ the cost of a 64 GB laptop, and you can scale up by the hour for cascade runs. **Best ROI by far if you're not constantly travelling.**
3. **Avoid:** Contabo for the live correlation engine, base 16 GB MacBook Air for any heavy notebook work, and base-spec Vultr Cloud Compute for low-latency code.

---

## Verification (how to validate the recommendation)

Before committing to a purchase or VPS plan:

1. **Measure peak RAM on your real workload** — run NB04 (`WRAP_P6_PATTERN_MINING`) with `max_snapshots=None` on a full-day file and watch `Activity Monitor → Memory → Memory Used` and `vm_stat` swap counters. If peak Memory Used ≤ 24 GB, the 32 GB tier is sufficient. If you hit 28 GB+, jump to 64 GB.
2. **Trial Hetzner for 1 month** before committing — the CCX33 is hourly-billed (~$0.09/hr), so a week of testing costs ~$15. Run NB04 + NB06 on it, compare wall-clock to your laptop, decide.
3. **For laptops, look at sustained perf, not peak** — Notebook suites are minutes-to-hours of compute. Check Notebookcheck/PCWorld sustained-load benchmarks for the specific configs above before buying; thin-and-light chassis throttle.

---

## Sources

- [Vultr Pricing](https://www.vultr.com/pricing/) · [Vultr High Frequency](https://www.vultr.com/products/high-frequency-compute/) · [Vultr Optimized](https://www.vultr.com/products/optimized-cloud-compute/) · [Vultr Review 2026 — Better Stack](https://betterstack.com/community/guides/web-servers/vultr-review/)
- [Hetzner alternatives roundup](https://dev.to/alakkadshaw/hetzner-alternatives-for-2025-digitalocean-linode-vultr-ovhcloud-5936) · [DigitalOcean vs Hetzner 2026](https://betterstack.com/community/guides/web-servers/digitalocean-vs-hetzner/) · [Top low-cost VPS 2026](https://www.nucamp.co/blog/top-10-low-cost-vps-providers-in-2026-affordable-alternatives-to-aws-azure-gcp-and-vercel)
- [Best laptops 2026 — Tom's Hardware](https://www.tomshardware.com/laptops/best-laptops) · [Best 32 GB RAM laptops — Windows Central](https://www.windowscentral.com/best-laptop-32gb-ram) · [PCWorld best laptops](https://www.pcworld.com/article/436674/best-pc-laptops.html)
