"""U-Net forward contracts and interchangeability with DiT."""

from __future__ import annotations

import pytest
import torch

from multimodal.models.diffusion.unet import UNet2DConditionModel


@pytest.fixture()
def tiny_unet():
    torch.manual_seed(0)
    return UNet2DConditionModel(
        in_channels=4,
        out_channels=4,
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(4, 2),
        num_heads=4,
        head_dim=32,
        context_dim=64,
        dropout=0.0,
    )


class TestForward:
    def test_shape_contract(self, tiny_unet):
        x = torch.randn(2, 4, 16, 16)
        t = torch.randint(0, 1000, (2,))
        ctx = torch.randn(2, 7, 64)
        out = tiny_unet(x, t, encoder_hidden_states=ctx)
        assert out.shape == (2, 4, 16, 16)

    def test_requires_context_when_attention_present(self, tiny_unet):
        """Attention blocks need context — the pipeline always provides it."""
        x = torch.randn(1, 4, 16, 16)
        t = torch.tensor([42])
        with pytest.raises(RuntimeError):
            tiny_unet(x, t)

    def test_zero_init_output_head(self, tiny_unet):
        """diffusers convention: conv_out zero-init → model starts as identity
        (predicts zero noise), the property that stabilizes training start."""
        x = torch.randn(1, 4, 16, 16)
        t = torch.tensor([42])
        ctx = torch.randn(1, 7, 64)
        out = tiny_unet(x, t, encoder_hidden_states=ctx)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_context_changes_output(self, tiny_unet):
        # Un-zero the output head so conditioning effects are observable.
        with torch.no_grad():
            tiny_unet.conv_out.weight.normal_(0.0, 0.02)
        tiny_unet.eval()
        x = torch.randn(1, 4, 16, 16)
        t = torch.tensor([100])
        with torch.no_grad():
            out_a = tiny_unet(x, t, encoder_hidden_states=torch.randn(1, 7, 64))
            out_b = tiny_unet(x, t, encoder_hidden_states=torch.randn(1, 7, 64))
        assert not torch.allclose(out_a, out_b, atol=1e-6)


class TestInterchangeability:
    def test_same_call_signature_as_dit(self, tiny_unet):
        """The pipeline passes (x, t, encoder_hidden_states=...) to either
        denoiser — this is the plug-and-play contract, asserted directly."""
        x = torch.randn(1, 4, 16, 16)
        t = torch.tensor([5])
        ctx = torch.randn(1, 6, 64)
        out = tiny_unet(x, t, encoder_hidden_states=ctx)
        assert out.shape == x.shape

    def test_gradient_flow(self, tiny_unet):
        x = torch.randn(1, 4, 16, 16)
        t = torch.tensor([5])
        ctx = torch.randn(1, 6, 64)
        out = tiny_unet(x, t, encoder_hidden_states=ctx)
        out.sum().backward()
        grads = [p.grad for p in tiny_unet.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)
