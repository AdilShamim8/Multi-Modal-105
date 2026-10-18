"""VQ-VAE codebook: quantization, straight-through gradients, EMA updates."""

from __future__ import annotations

import pytest
import torch

from multimodal.models.autoregressive.vqvae import VectorQuantizer


@pytest.fixture()
def quantizer():
    torch.manual_seed(0)
    q = VectorQuantizer(num_embeddings=64, embedding_dim=8, commitment_cost=0.25)
    q.eval()  # freeze EMA unless a test opts into training mode
    return q


class TestQuantization:
    def test_output_shape_matches_input(self, quantizer):
        z = torch.randn(2, 8, 6, 6)
        out, loss, indices = quantizer(z)
        assert out.shape == z.shape
        assert loss.dim() == 0
        assert indices.shape == (2, 36)  # (B, H*W)

    def test_indices_within_vocab(self, quantizer):
        z = torch.randn(4, 8, 6, 6)
        _, _, indices = quantizer(z)
        assert indices.min() >= 0 and indices.max() < 64

    def test_quantized_outputs_are_codebook_entries(self, quantizer):
        z = torch.randn(2, 8, 4, 4)
        with torch.no_grad():
            _, _, indices = quantizer(z)
        selected = quantizer.embedding(indices.reshape(-1))  # (N, D)
        # Re-run to get the quantized tensor (eval mode: no EMA drift).
        out, _, _ = quantizer(z)
        expected = selected.reshape(2, 4, 4, 8).permute(0, 3, 1, 2)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_deterministic_in_eval_mode(self, quantizer):
        z = torch.randn(2, 8, 4, 4)
        out1, _, idx1 = quantizer(z)
        out2, _, idx2 = quantizer(z)
        assert torch.equal(idx1, idx2)
        assert torch.equal(out1, out2)


class TestStraightThroughGradients:
    def test_gradient_flows_to_encoder(self, quantizer):
        z = torch.randn(2, 8, 4, 4, requires_grad=True)
        out, loss, _ = quantizer(z)
        (out.sum() + loss).backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_st_gradient_is_identity(self, quantizer):
        """Straight-through: value is the codebook entry, but d(out)/dz ≡ 1."""
        z = torch.randn(2, 8, 4, 4, requires_grad=True)
        out, _, _ = quantizer(z)
        out.sum().backward()
        # Gradient flowed as if out == z (identity Jacobian).
        assert torch.allclose(z.grad, torch.ones_like(z), atol=1e-6)


class TestLossDirection:
    def test_zero_loss_for_exact_codebook_hits(self, quantizer):
        with torch.no_grad():
            codes = quantizer.embedding.weight[:4].clone()  # 4 exact entries
        z = codes.reshape(4, 8, 1, 1)
        _, loss, _ = quantizer(z)
        assert float(loss) < 1e-6

    def test_commitment_grows_with_distance(self, quantizer):
        torch.manual_seed(1)
        near = torch.randn(2, 8, 4, 4) * 0.01
        far = torch.randn(2, 8, 4, 4) * 10.0
        _, loss_near, _ = quantizer(near)
        _, loss_far, _ = quantizer(far)
        assert float(loss_far) > float(loss_near)


class TestEMAUpdates:
    def test_training_mode_moves_codebook(self):
        torch.manual_seed(0)
        q = VectorQuantizer(num_embeddings=16, embedding_dim=8)
        q.train()
        before = q.embedding.weight.clone()
        z = torch.randn(8, 8, 4, 4)
        q(z)
        after = q.embedding.weight
        assert not torch.allclose(before, after), "EMA must move codebook entries"

    def test_cluster_sizes_accumulate(self):
        torch.manual_seed(0)
        q = VectorQuantizer(num_embeddings=16, embedding_dim=8)
        q.train()
        assert float(q.ema_cluster_size.sum()) == 0.0
        q(torch.randn(4, 8, 2, 2))
        assert float(q.ema_cluster_size.sum()) > 0.0
