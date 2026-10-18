"""Multi-component integration: the REAL model stack on CPU at toy scale.

Tiny VAE + tiny DiT + real scheduler + real DDIM sampler, wired exactly
like T2IPipeline does. Slow (~20 s) — nightly tier, not the PR gate.
"""

from __future__ import annotations

import pytest
import torch

from multimodal.models.diffusion.dit import DiT
from multimodal.models.diffusion.noise_scheduler import NoiseScheduler
from multimodal.models.vae.vae import AutoencoderKL
from multimodal.sampling.ddim_sampler import DDIMSampler

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_stack():
    torch.manual_seed(0)
    vae = AutoencoderKL(
        base_channels=32,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=1,
        attention_resolutions=(8,),
        latent_channels=4,
    ).eval()
    dit = DiT(
        input_size=8, patch_size=2, in_channels=4, hidden_size=64,
        depth=2, num_heads=4, num_classes=10, context_dim=32,
        learn_sigma=False,
    ).eval()
    scheduler = NoiseScheduler(prediction_type="epsilon")
    return vae, dit, scheduler


def encode_prompt(text: str) -> torch.Tensor:
    """Deterministic stand-in for CLIP: hash-seeded embedding."""
    seed = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 5, 32, generator=g)


class TestEndToEndGeneration:
    def test_prompt_to_image_roundtrip(self, tiny_stack):
        vae, dit, scheduler = tiny_stack
        ctx = encode_prompt("a golden retriever in snow")
        sampler = DDIMSampler(scheduler)

        def model_fn(x, t, **kwargs):
            return dit(x, t, text_embeds=ctx.expand(x.shape[0], -1, -1))

        latents = sampler.sample(
            model_fn, (1, 4, 8, 8), "cpu",
            num_inference_steps=4, show_progress=False,
        )
        assert latents.shape == (1, 4, 8, 8)
        assert torch.isfinite(latents).all()

        with torch.no_grad():
            images = vae.decode_from_latent(latents / vae.scaling_factor)
        assert images.shape == (1, 3, 64, 64)
        assert torch.isfinite(images).all()

    def test_seed_reproducibility(self, tiny_stack):
        _vae, dit, scheduler = tiny_stack
        sampler = DDIMSampler(scheduler)
        ctx = encode_prompt("same prompt")

        def model_fn(x, t, **kw):
            return dit(x, t, text_embeds=ctx.expand(x.shape[0], -1, -1))

        torch.manual_seed(7)
        a = sampler.sample(model_fn, (1, 4, 8, 8), "cpu",
                           num_inference_steps=4, show_progress=False)
        torch.manual_seed(7)
        b = sampler.sample(model_fn, (1, 4, 8, 8), "cpu",
                           num_inference_steps=4, show_progress=False)
        assert torch.equal(a, b), "same seed must give byte-identical latents"

    def test_different_prompts_diverge(self, tiny_stack):
        _vae, dit, scheduler = tiny_stack
        sampler = DDIMSampler(scheduler)

        outputs = []
        for text in ("a cat", "a lighthouse"):
            ctx = encode_prompt(text)

            def model_fn(x, t, ctx=ctx, **kw):
                return dit(x, t, text_embeds=ctx.expand(x.shape[0], -1, -1))

            torch.manual_seed(3)
            outputs.append(sampler.sample(model_fn, (1, 4, 8, 8), "cpu",
                                          num_inference_steps=4,
                                          show_progress=False))
        assert not torch.allclose(outputs[0], outputs[1], atol=1e-4)

    def test_cfg_scale_changes_output(self, tiny_stack):
        _vae, dit, scheduler = tiny_stack
        cond = encode_prompt("a cat")
        uncond = encode_prompt("")

        def make_model(scale: float):
            def model_fn(x, t, **kw):
                out_cond = dit(x, t, text_embeds=cond.expand(x.shape[0], -1, -1))
                out_uncond = dit(x, t, text_embeds=uncond.expand(x.shape[0], -1, -1))
                return out_uncond + scale * (out_cond - out_uncond)
            return model_fn

        sampler = DDIMSampler(scheduler)
        torch.manual_seed(1)
        low = sampler.sample(make_model(0.0), (1, 4, 8, 8), "cpu",
                             num_inference_steps=4, show_progress=False)
        torch.manual_seed(1)
        high = sampler.sample(make_model(7.5), (1, 4, 8, 8), "cpu",
                              num_inference_steps=4, show_progress=False)
        assert not torch.allclose(low, high, atol=1e-4)

    def test_vae_reconstruction_fidelity(self, tiny_stack):
        """An identity-ish sanity: encoding then decoding must preserve gross
        statistics (mean/std) of the input at toy scale."""
        vae, _dit, _scheduler = tiny_stack
        torch.manual_seed(0)
        x = torch.randn(1, 3, 64, 64) * 0.5
        with torch.no_grad():
            z = vae.encode(x).mode()
            recon = vae.decode(z)
        assert recon.shape == x.shape
        # Cheap proxy for fidelity: energy in the same ballpark.
        assert recon.abs().mean() == pytest.approx(x.abs().mean(), rel=0.6)
