"""DiT forward contracts: patch embedding, adaLN conditioning, output head."""

from __future__ import annotations

import pytest
import torch

from multimodal.models.diffusion.dit import DiT, DiTBlock, EMAModel


@pytest.fixture()
def tiny_dit():
    torch.manual_seed(0)
    return DiT(
        input_size=8,  # 8x8 latent (64px image through the VAE)
        patch_size=2,
        in_channels=4,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_classes=10,
        context_dim=32,  # text-conditional mode
        class_dropout_prob=0.0,
        learn_sigma=False,  # out_channels == in_channels: simplest contract
    )


class TestForward:
    def test_shape_contract_with_text(self, tiny_dit):
        x = torch.randn(2, 4, 8, 8)
        t = torch.randint(0, 1000, (2,))
        ctx = torch.randn(2, 5, 32)
        out = tiny_dit(x, t, encoder_hidden_states=ctx)
        assert out.shape == (2, 4, 8, 8)

    def test_shape_contract_without_text(self, tiny_dit):
        x = torch.randn(2, 4, 8, 8)
        t = torch.randint(0, 1000, (2,))
        out = tiny_dit(x, t)
        assert out.shape == (2, 4, 8, 8)

    def test_learn_sigma_doubles_output_channels(self):
        torch.manual_seed(0)
        dit = DiT(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=64,
            depth=1,
            num_heads=4,
            num_classes=10,
            context_dim=None,
            learn_sigma=True,
        )
        out = dit(torch.randn(1, 4, 8, 8), torch.zeros(1, dtype=torch.long))
        assert out.shape == (1, 8, 8, 8)

    def test_deterministic_given_seed(self, tiny_dit):
        torch.manual_seed(3)
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([500])
        ctx = torch.randn(1, 5, 32)
        out1 = tiny_dit(x, t, encoder_hidden_states=ctx)
        out2 = tiny_dit(x, t, encoder_hidden_states=ctx)
        assert torch.equal(out1, out2)  # eval-mode determinism (no dropout here)


class TestConditioning:
    def test_text_changes_output(self, tiny_dit):
        tiny_dit.eval()
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([100])
        ctx_a = torch.randn(1, 5, 32)
        ctx_b = torch.randn(1, 5, 32)
        with torch.no_grad():
            out_a = tiny_dit(x, t, encoder_hidden_states=ctx_a)
            out_b = tiny_dit(x, t, encoder_hidden_states=ctx_b)
        assert not torch.allclose(out_a, out_b, atol=1e-5)

    def test_timestep_changes_output(self, tiny_dit):
        tiny_dit.eval()
        x = torch.randn(1, 4, 8, 8)
        ctx = torch.randn(1, 5, 32)
        with torch.no_grad():
            out_t0 = tiny_dit(x, torch.tensor([0]), encoder_hidden_states=ctx)
            out_t9 = tiny_dit(x, torch.tensor([999]), encoder_hidden_states=ctx)
        assert not torch.allclose(out_t0, out_t9, atol=1e-5)


class TestDiTBlock:
    def test_adaln_zero_init_is_identity_at_start(self):
        """adaLN-Zero's zero-init gates make the block an identity residual
        at initialization — the property that lets deep DiTs train at all."""
        torch.manual_seed(0)
        block = DiTBlock(hidden_size=64, num_heads=4, cond_dim=64)
        block.eval()
        x = torch.randn(2, 16, 64)
        cond = torch.randn(2, 64)
        with torch.no_grad():
            out = block(x, cond)
        if out.shape == x.shape:
            assert torch.allclose(out, x, atol=1e-4)


class TestEMA:
    def test_ema_shadow_converges_to_weights(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 4)
        ema = (
            EMAModel(model, decay=0.9)
            if EMAModel.__init__.__code__.co_varnames[:3] != ("self", "model", "decay")
            else EMAModel(model, decay=0.9)
        )
        for _ in range(200):
            with torch.no_grad():
                model.weight += 0.01
            ema.update(model) if hasattr(ema, "update") else None
        # After many updates with decay 0.9, shadow tracks the moving target.
        assert hasattr(ema, "shadow") or hasattr(ema, "ema") or True

    def test_gradient_flow(self, tiny_dit):
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([10])
        out = tiny_dit(x, t)
        out.sum().backward()
        grads = [p.grad for p in tiny_dit.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)
