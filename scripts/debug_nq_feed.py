#!/usr/bin/env python3
"""
Debug script: inspect raw Databento MBO data to determine how prices,
sides, and actions are encoded in to_df() output.

Usage:
    python debug_nq_feed.py data/nq-mbo-2026-03-20.dbn.zst
"""
from __future__ import annotations

import sys
import os

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/nq-mbo-2026-03-20.dbn.zst"

    print(f"Loading: {path}")
    import databento as db

    store = db.DBNStore.from_file(path)
    df = store.to_df()

    print(f"\nTotal rows: {len(df):,}")
    print(f"\nColumn dtypes:")
    print(df.dtypes)

    print(f"\nFirst 20 rows (key columns):")
    cols = [c for c in ["price", "side", "action", "order_id", "size"] if c in df.columns]
    print(df[cols].head(20).to_string())

    print(f"\nPrice stats (raw):")
    p = df["price"] if "price" in df.columns else None
    if p is not None:
        print(f"  min  : {p.min()}")
        print(f"  max  : {p.max()}")
        print(f"  mean : {p.mean():.2f}")
        print(f"  nan  : {p.isna().sum()}")
        print(f"  dtype: {p.dtype}")
        # Detect if values look like fixed-point (>>1e6) or real prices (~20000 for NQ)
        non_nan = p.dropna()
        if len(non_nan) > 0:
            sample = non_nan.iloc[0]
            print(f"  first non-nan: {sample}")
            if abs(sample) > 1_000_000:
                print("  ⚠️  Looks like fixed-point (needs /1e9)")
            else:
                print("  ✅ Looks like real price already")

    print(f"\nSide value_counts:")
    if "side" in df.columns:
        print(df["side"].value_counts())
        print(f"  dtype: {df['side'].dtype}")

    print(f"\nAction value_counts:")
    if "action" in df.columns:
        print(df["action"].value_counts())
        print(f"  dtype: {df['action'].dtype}")

    # Check for rows where price is 0 or NaN after potential conversion
    if p is not None:
        zero_rows = (p == 0).sum()
        print(f"\nRows where price == 0: {zero_rows}")

    # Show a few ADD events specifically
    if "action" in df.columns:
        adds = df[df["action"] == "A"] if "A" in df["action"].values else df[df["action"] == 1]
        print(f"\nFirst 5 ADD events:")
        print(adds[cols].head(5).to_string())

    # Check index type
    print(f"\nIndex type: {type(df.index)}")
    print(f"Index dtype: {df.index.dtype}")
    if hasattr(df.index, 'tzinfo'):
        print(f"Index tzinfo: {df.index.tzinfo}")
    print(f"First index value: {df.index[0]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
