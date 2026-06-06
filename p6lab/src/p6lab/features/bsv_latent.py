"""Wave 4 Phase 3 — 1D CNN autoencoder on BSV time-series.

The 40-dim ``book_shape_vector`` is a depth pyramid (20 bid levels + 20 ask
levels, self-normalized). Hand-crafted features aggregate it into scalars
(norm, top-5 sum, etc.) but a conv autoencoder can learn non-linear shapes
— "inverted T", "ladder", "hockey stick" — that hand-crafted reductions
miss.

The encoder takes a (40, 20) window (40 price levels × 20-snap lookback)
and outputs a 16-dim bottleneck. Those 16 values become new feature
columns ``bsv_latent_00..bsv_latent_15`` in NB06 §03.

Architecture
------------
Input: (batch, 40, 20)
  Conv1d(40 → 32, kernel=3, stride=1, padding=1) + ReLU + BatchNorm
  Conv1d(32 → 16, kernel=3, stride=2, padding=1) + ReLU + BatchNorm
  Conv1d(16 → 8,  kernel=3, stride=2, padding=1) + ReLU
  Flatten → Linear(8*5 → 16)  # bottleneck
  Linear(16 → 8*5) → Unflatten(8, 5)
  ConvTranspose1d(8  → 16, kernel=3, stride=2, padding=1, output_padding=1)
  ConvTranspose1d(16 → 32, kernel=3, stride=2, padding=1, output_padding=1)
  ConvTranspose1d(32 → 40, kernel=3, stride=1, padding=1)
  → (batch, 40, 20) reconstruction
Loss: MSE

Inference
---------
At match time, maintain a rolling (40, 20) BSV buffer per instrument;
when full, run encoder, append 16 latent floats to feature row.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


BSV_DIM = 40               # 20 bid + 20 ask
BSV_LOOKBACK = 20          # 20 snapshots ≈ 2s at 100ms cadence
LATENT_DIM = 16

LATENT_FEATURE_NAMES: list[str] = [
    f"bsv_latent_{i:02d}" for i in range(LATENT_DIM)
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


if HAS_TORCH:
    class BSVEncoder(nn.Module):
        """Conv1d encoder (40, 20) → 16."""
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(BSV_DIM, 32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1)
            self.bn2 = nn.BatchNorm1d(16)
            self.conv3 = nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(8 * 5, LATENT_DIM)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = torch.relu(self.conv3(x))
            x = self.flatten(x)
            return self.fc(x)

    class BSVDecoder(nn.Module):
        """Mirror decoder 16 → (40, 20)."""
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(LATENT_DIM, 8 * 5)
            self.deconv1 = nn.ConvTranspose1d(
                8, 16, kernel_size=3, stride=2, padding=1, output_padding=1,
            )
            self.deconv2 = nn.ConvTranspose1d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1,
            )
            self.deconv3 = nn.ConvTranspose1d(
                32, BSV_DIM, kernel_size=3, stride=1, padding=1,
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            x = self.fc(z).view(-1, 8, 5)
            x = torch.relu(self.deconv1(x))
            x = torch.relu(self.deconv2(x))
            return self.deconv3(x)

    class BSVAutoencoder(nn.Module):
        """Conv autoencoder for BSV time-series."""
        def __init__(self) -> None:
            super().__init__()
            self.encoder = BSVEncoder()
            self.decoder = BSVDecoder()

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon
else:
    class BSVEncoder: ...      # noqa: E701
    class BSVDecoder: ...      # noqa: E701
    class BSVAutoencoder: ...  # noqa: E701


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@dataclass
class BSVLatentState:
    """Rolling buffer of the last ``BSV_LOOKBACK`` BSV vectors."""
    buffer: list[np.ndarray]

    @classmethod
    def new(cls) -> "BSVLatentState":
        return cls(buffer=[])

    def push(self, bsv: np.ndarray) -> None:
        if bsv.shape != (BSV_DIM,):
            return   # skip malformed
        self.buffer.append(bsv.astype(np.float32))
        if len(self.buffer) > BSV_LOOKBACK:
            self.buffer.pop(0)

    @property
    def ready(self) -> bool:
        return len(self.buffer) == BSV_LOOKBACK

    def window(self) -> np.ndarray:
        """Return (40, 20) stacked window. Caller must check ready first."""
        if not self.ready:
            raise RuntimeError("BSVLatentState not ready — need 20 snapshots")
        # Shape (20, 40) after stacking → transpose to (40, 20)
        return np.stack(self.buffer, axis=1)


def extract_latent(
    model: "BSVAutoencoder",
    state: BSVLatentState,
    device: str = "cpu",
) -> np.ndarray:
    """Return a 16-dim latent vector for the current state, or zeros on warmup."""
    if not HAS_TORCH:
        return np.zeros(LATENT_DIM, dtype=np.float32)
    if not state.ready:
        return np.zeros(LATENT_DIM, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(state.window()).unsqueeze(0).to(device)
        z = model.encoder(x)
    return z.cpu().numpy().reshape(-1).astype(np.float32)


def extract_latent_batch(
    model: "BSVAutoencoder",
    bsvs: np.ndarray,   # (N, 40)
    lookback: int = BSV_LOOKBACK,
    device: str = "cpu",
) -> np.ndarray:
    """Batch extraction for NB06 §03. Returns (N, 16) array, zeros in warmup.

    Used by the notebook to produce the full ``bsv_latent_*`` columns
    when a checkpoint is available.
    """
    if not HAS_TORCH:
        return np.zeros((len(bsvs), LATENT_DIM), dtype=np.float32)
    n = len(bsvs)
    out = np.zeros((n, LATENT_DIM), dtype=np.float32)
    if n < lookback:
        return out
    # Build (n - lookback + 1, 40, 20) batch
    windows = np.stack(
        [bsvs[i - lookback + 1:i + 1].T for i in range(lookback - 1, n)],
        axis=0,
    ).astype(np.float32)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(windows).to(device)
        z = model.encoder(x).cpu().numpy()
    out[lookback - 1:] = z
    return out


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_encoder(model: "BSVAutoencoder", path: Path) -> None:
    if not HAS_TORCH:
        raise RuntimeError("torch not installed — cannot save encoder")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_encoder(path: Path) -> "BSVAutoencoder | None":
    """Load a saved autoencoder. Returns None if torch missing or file absent."""
    if not HAS_TORCH:
        return None
    if not path.exists():
        return None
    model = BSVAutoencoder()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_autoencoder(
    bsv_sequences: np.ndarray,    # (N, 40) rows of BSV
    *,
    lookback: int = BSV_LOOKBACK,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple["BSVAutoencoder", dict[str, list[float]]]:
    """Train a BSVAutoencoder on the supplied sequence.

    Returns (trained_model, {'train_loss': [...], 'val_loss': [...]}).
    Raises if torch is not installed.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch not installed — `pip install torch`")

    # Build (n_windows, 40, 20) dataset
    n = len(bsv_sequences)
    if n < lookback + 10:
        raise ValueError(f"need ≥{lookback + 10} BSV rows; got {n}")
    windows = np.stack(
        [bsv_sequences[i - lookback + 1:i + 1].T for i in range(lookback - 1, n)],
        axis=0,
    ).astype(np.float32)

    # Train/val split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(windows))
    split = int(len(windows) * (1.0 - val_split))
    train_idx, val_idx = idx[:split], idx[split:]
    x_train = torch.from_numpy(windows[train_idx]).to(device)
    x_val = torch.from_numpy(windows[val_idx]).to(device)

    model = BSVAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        train_loss_sum = 0.0
        n_batches = 0
        for i in range(0, len(x_train), batch_size):
            batch_idx = perm[i:i + batch_size]
            xb = x_train[batch_idx]
            _, recon = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            _, recon_val = model(x_val)
            val_loss = loss_fn(recon_val, x_val).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    return model, history
