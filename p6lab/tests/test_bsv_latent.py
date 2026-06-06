"""Wave 4 Phase 3 — unit tests for the 1D CNN BSV autoencoder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from p6lab.features.bsv_latent import (
    BSV_DIM, BSV_LOOKBACK, LATENT_DIM, LATENT_FEATURE_NAMES,
    BSVAutoencoder, BSVEncoder, BSVLatentState,
    extract_latent, extract_latent_batch,
    load_encoder, save_encoder, train_autoencoder,
)


# ---------------------------------------------------------------------------
# Architecture + shape tests
# ---------------------------------------------------------------------------


class TestArchitecture:
    def test_constants(self):
        assert BSV_DIM == 40
        assert BSV_LOOKBACK == 20
        assert LATENT_DIM == 16
        assert len(LATENT_FEATURE_NAMES) == LATENT_DIM

    def test_encoder_output_shape(self):
        m = BSVEncoder()
        m.eval()
        x = torch.randn(4, BSV_DIM, BSV_LOOKBACK)
        z = m(x)
        assert z.shape == (4, LATENT_DIM)

    def test_autoencoder_reconstruction_shape(self):
        m = BSVAutoencoder()
        m.eval()
        x = torch.randn(2, BSV_DIM, BSV_LOOKBACK)
        z, recon = m(x)
        assert z.shape == (2, LATENT_DIM)
        assert recon.shape == x.shape


# ---------------------------------------------------------------------------
# BSVLatentState rolling buffer
# ---------------------------------------------------------------------------


class TestBSVLatentState:
    def test_warmup_not_ready(self):
        s = BSVLatentState.new()
        assert not s.ready
        for _ in range(BSV_LOOKBACK - 1):
            s.push(np.ones(BSV_DIM))
        assert not s.ready

    def test_ready_after_fill(self):
        s = BSVLatentState.new()
        for _ in range(BSV_LOOKBACK):
            s.push(np.ones(BSV_DIM))
        assert s.ready
        w = s.window()
        assert w.shape == (BSV_DIM, BSV_LOOKBACK)

    def test_rolling_drops_oldest(self):
        s = BSVLatentState.new()
        for i in range(BSV_LOOKBACK + 5):
            s.push(np.full(BSV_DIM, float(i)))
        w = s.window()
        # First col should now be (5, 5, ..., 5) because i=5 was the first kept
        assert w[0, 0] == 5.0
        assert w[0, -1] == BSV_LOOKBACK + 4

    def test_malformed_bsv_ignored(self):
        s = BSVLatentState.new()
        s.push(np.ones(BSV_DIM))
        s.push(np.ones(10))   # wrong size — should be skipped
        assert len(s.buffer) == 1


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


class TestInference:
    def test_extract_latent_warmup_zeros(self):
        model = BSVAutoencoder()
        state = BSVLatentState.new()
        for _ in range(5):
            state.push(np.ones(BSV_DIM))
        z = extract_latent(model, state)
        assert z.shape == (LATENT_DIM,)
        assert np.all(z == 0)   # warmup → zeros

    def test_extract_latent_after_warmup(self):
        model = BSVAutoencoder()
        state = BSVLatentState.new()
        rng = np.random.default_rng(0)
        for _ in range(BSV_LOOKBACK):
            state.push(rng.random(BSV_DIM).astype(np.float32))
        z = extract_latent(model, state)
        assert z.shape == (LATENT_DIM,)
        # untrained model — just check no NaN
        assert np.all(np.isfinite(z))

    def test_batch_shape(self):
        model = BSVAutoencoder()
        rng = np.random.default_rng(0)
        bsvs = rng.random((100, BSV_DIM)).astype(np.float32)
        out = extract_latent_batch(model, bsvs)
        assert out.shape == (100, LATENT_DIM)
        # Warmup rows are zero
        assert np.all(out[:BSV_LOOKBACK - 1] == 0)
        # Post-warmup rows are non-zero (untrained model outputs noise)
        assert np.any(out[BSV_LOOKBACK - 1:] != 0)


# ---------------------------------------------------------------------------
# Training + serialization
# ---------------------------------------------------------------------------


class TestTraining:
    def test_trains_and_loss_decreases(self):
        rng = np.random.default_rng(0)
        # Generate a structured (not random) BSV stream: bid-heavy ramps
        t = np.linspace(0, 2 * np.pi, 500)
        base = np.stack([
            np.clip(np.sin(t + i * 0.05), 0, None) for i in range(BSV_DIM)
        ], axis=1).astype(np.float32)
        # Add small noise
        data = base + 0.05 * rng.standard_normal(base.shape).astype(np.float32)

        model, history = train_autoencoder(
            data, epochs=10, batch_size=32, verbose=False,
        )
        assert len(history["train_loss"]) == 10
        assert history["train_loss"][-1] < history["train_loss"][0]
        assert history["val_loss"][-1] < history["val_loss"][0]

    def test_save_load_roundtrip(self, tmp_path: Path):
        model = BSVAutoencoder()
        model.eval()
        path = tmp_path / "encoder.pt"
        save_encoder(model, path)
        loaded = load_encoder(path)
        assert loaded is not None

        # Forward passes match
        x = torch.randn(1, BSV_DIM, BSV_LOOKBACK)
        with torch.no_grad():
            z1, _ = model(x)
            z2, _ = loaded(x)
        assert torch.allclose(z1, z2, atol=1e-6)

    def test_load_missing_returns_none(self, tmp_path: Path):
        assert load_encoder(tmp_path / "nonexistent.pt") is None
