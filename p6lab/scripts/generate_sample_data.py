"""
Generate the tiny `data/nq-mbo-sample-*.dbn.zst` fixture used by CI.

Strategy — direct byte-level slice of the source DBN file:

  1. Stream-decompress the full overnight .dbn.zst
  2. Parse the DBN header to find where records begin (v1 format: 8-byte
     fixed prefix + ``metadata_length`` bytes of metadata)
  3. Assume MBO schema (56-byte fixed records); walk records until the
     first ``ts_event`` that exceeds ``start_ts + duration_ns``
  4. Write header + kept records back through zstd

This works purely from the DBN bytewise layout — no databento-level
filtering APIs involved — so it produces a valid, self-contained file
that ``databento.DBNStore.from_file`` round-trips cleanly.

Run:

    python3 scripts/generate_sample_data.py                   # default 5 min
    python3 scripts/generate_sample_data.py --duration 10     # 10 min
    python3 scripts/generate_sample_data.py --source FILE     # alt source
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT.parent / "data" / "nq-mbo-overnight-2026-03-26.dbn.zst"
DEFAULT_OUT = ROOT.parent / "data" / "nq-mbo-sample-15min.dbn.zst"
MBO_RECORD_BYTES = 56


def subset_dbn(source: Path, out: Path, duration_minutes: float, *,
               zstd_level: int = 6) -> dict:
    import zstandard as zstd
    if not source.is_file():
        raise FileNotFoundError(f"source missing: {source}")

    # 1. Stream-decompress the source .dbn.zst
    dctx = zstd.ZstdDecompressor()
    with open(source, "rb") as f, dctx.stream_reader(f) as reader:
        raw = reader.read()

    # 2. Parse DBN header
    if raw[:3] != b"DBN":
        raise ValueError(f"{source} is not a DBN file (magic {raw[:3]!r})")
    metadata_len = struct.unpack("<I", raw[4:8])[0]
    records_start = 8 + metadata_len
    records_len = len(raw) - records_start
    if records_len % MBO_RECORD_BYTES:
        raise ValueError(
            f"records section ({records_len} bytes) not divisible by "
            f"{MBO_RECORD_BYTES} — source is not pure-MBO schema"
        )
    n_records = records_len // MBO_RECORD_BYTES

    # 3. Find the first record past (start_ts + duration)
    start_ts = struct.unpack(
        "<Q", raw[records_start + 8 : records_start + 16]
    )[0]
    duration_ns = int(duration_minutes * 60 * 1_000_000_000)
    cutoff_ts = start_ts + duration_ns

    cutoff_idx = n_records   # fall through: take the whole file
    for i in range(n_records):
        off = records_start + i * MBO_RECORD_BYTES
        ts = struct.unpack("<Q", raw[off + 8 : off + 16])[0]
        if ts > cutoff_ts:
            cutoff_idx = i
            break

    cutoff_bytes = records_start + cutoff_idx * MBO_RECORD_BYTES
    filtered = raw[:cutoff_bytes]

    # 4. Re-compress
    cctx = zstd.ZstdCompressor(level=zstd_level)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(cctx.compress(filtered))

    stats = {
        "source": str(source),
        "output": str(out),
        "source_records": n_records,
        "kept_records": cutoff_idx,
        "duration_minutes": duration_minutes,
        "compressed_bytes": out.stat().st_size,
        "decompressed_bytes": len(filtered),
    }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE),
                    help="input .dbn.zst file")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output .dbn.zst file")
    ap.add_argument("--duration", type=float, default=15.0,
                    help="minutes of tape to keep (default: 15)")
    ap.add_argument("--zstd-level", type=int, default=6,
                    help="zstd compression level 1-22 (default: 6)")
    args = ap.parse_args()

    stats = subset_dbn(
        Path(args.source), Path(args.out),
        duration_minutes=args.duration,
        zstd_level=args.zstd_level,
    )

    print(f"source        : {stats['source']}")
    print(f"output        : {stats['output']}")
    print(f"source records: {stats['source_records']:,}")
    print(f"kept records  : {stats['kept_records']:,}")
    print(f"duration      : {stats['duration_minutes']:.1f} minutes")
    print(f"compressed    : {stats['compressed_bytes']:,} bytes "
          f"({stats['compressed_bytes'] / 1024:.1f} KB)")
    print(f"decompressed  : {stats['decompressed_bytes']:,} bytes")

    # Round-trip validate the output
    try:
        import databento as db
        store = db.DBNStore.from_file(args.out)
        n = sum(1 for _ in store)
        assert n == stats["kept_records"], f"round-trip mismatch: {n} vs {stats['kept_records']}"
        print(f"round-trip    : OK ({n:,} records parsed by databento)")
    except ImportError:
        print("round-trip    : skipped (databento not installed)")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
