"""Train the BSV 1D CNN autoencoder and save checkpoint.

Wave 4 Phase 3. Collects BSV rows by running
``p6lab.features.l2_features.compute_book_shape_vector`` over the
sample tape(s), trains a small conv autoencoder, saves to
``artifacts/p6lab/bsv_cnn/encoder_v1.pt``.

Usage
-----
Train on the committed smoketest sample::

    python scripts/train_bsv_cnn.py

Train on a specific file with a different lookback::

    python scripts/train_bsv_cnn.py --file ../data/nq-mbo-fullday-2026-03-25.dbn.zst \
        --epochs 30 --batch 128
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="train_bsv_cnn")
    p.add_argument("--file", type=Path, default=None,
                   help="Override data file (default: smoketest sample)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/p6lab/bsv_cnn/encoder_v1.pt"))
    p.add_argument("--max-snapshots", type=int, default=2000)
    return p.parse_args(argv)


async def _collect_bsvs(data_file: Path | None,
                         max_snapshots: int) -> np.ndarray:
    """Returns (N, 40) BSV array via the notebook helpers."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
    from _common import collect_snapshots, NOTEBOOK_DATA_SLICE
    from p6lab.features.l2_features import L2Snapshot, compute_book_shape_vector

    slice_ = dict(NOTEBOOK_DATA_SLICE)
    if data_file:
        slice_["data_file"] = str(data_file)
    slice_["max_snapshots"] = max_snapshots

    snaps = await collect_snapshots(slice_)
    bsvs = []
    for s in snaps:
        if not (s.bids and s.asks):
            continue
        bid_map = {lvl.price: lvl.volume for lvl in s.bids[:20]}
        ask_map = {lvl.price: lvl.volume for lvl in s.asks[:20]}
        prices = sorted(set(bid_map) | set(ask_map), reverse=True)
        book_levels = [(p, bid_map.get(p, 0.0), ask_map.get(p, 0.0)) for p in prices]
        bp, ap = s.bids[0].price, s.asks[0].price
        l2_snap = L2Snapshot(
            timestamp_ms=s.timestamp_ms, symbol="NQ",
            mid_price=(bp + ap) / 2, book_levels=book_levels,
        )
        bsvs.append(compute_book_shape_vector(l2_snap))
    return np.stack(bsvs).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print(f"Collecting BSV rows (max_snapshots={args.max_snapshots})...")
    bsvs = asyncio.run(_collect_bsvs(args.file, args.max_snapshots))
    print(f"  collected {bsvs.shape[0]} rows × {bsvs.shape[1]}-dim")

    from p6lab.features.bsv_latent import train_autoencoder, save_encoder

    print(f"Training autoencoder (epochs={args.epochs}, batch={args.batch})...")
    model, history = train_autoencoder(
        bsvs,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        verbose=True,
    )

    save_encoder(model, args.output)
    print(f"\n✓ saved encoder to {args.output}")
    print(f"  final train_loss = {history['train_loss'][-1]:.6f}")
    print(f"  final val_loss   = {history['val_loss'][-1]:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
