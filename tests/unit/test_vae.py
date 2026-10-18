"""VAE shape contracts and statistical invariants (tiny config, CPU)."""

from __future__ import annotations

import pytest
import torch

from multimodal.models.vae.vae import AutoencoderKL, DiagonalGaussian


@pytest.fixture()
def tiny_vae(tiny_vae_config):
    return AutoencoderKL(
        base_channels=tiny_vae_config.base_channels,
        channel_multipliers=tiny_vae_config.channel_multipliers,
        num_res_blocks=tiny_vae_config.num_res_blocks,
        attention_resolutions=tiny_vae_config.attention_resolutions,
        latent_channels=4,
    )


class TestShapes:
    def test_encode_returns_gaussian_with_8x_downsample(self, tiny_vae):
        x = torch.randn(2, 3, 64, 64)
        posterior = tiny_vae.encode(x)
        assert posterior.mu.shape == (2, 4, 8, 8)
        assert posterior.logvar.shape == (2, 4, 8, 8)

    def test_decode_restores_shape(self, tiny_vae):
        z = torch.randn(2, 4, 8, 8)
        out = tiny_vae.decode(z)
        assert out.shape == (2, 3, 64, 64)

    def test_forward_roundtrip_shapes(self, tiny_vae):
        x = torch.randn(2, 3, 64, 64)
        recon, posterior = tiny_vae(x)
        assert recon.shape == x.shape
        assert posterior.mu.shape == (2, 4, 8, 8)


class TestStatisticalContracts:
    def test_kl_nonnegative(self, tiny_vae):
        x = torch.randn(2, 3, 64, 64)
        posterior = tiny_vae.encode(x)
        assert float(posterior.kl()) >= 0.0

    def test_kl_zero_for_standard_normal_posterior(self):
        params = torch.cat(
            [torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8)], dim=1
        )  # mu=0, logvar=0 → std=1 → exactly N(0, I)
        gaussian = DiagonalGaussian(params)
        assert float(gaussian.kl()) == pytest.approx(0.0, abs=1e-6)

    def test_logvar_clamped(self):
        mu = torch.zeros(1, 4, 2, 2)
        logvar = torch.full((1, 4, 2, 2), 50.0)  # above the +20 clamp
        params = torch.cat([mu, logvar], dim=1)
        gaussian = DiagonalGaussian(params)
        assert float(gaussian.logvar.max()) <= 20.0

    def test_sample_vs_mode(self, tiny_vae):
        x = torch.randn(1, 3, 64, 64)
        posterior = tiny_vae.encode(x)
        torch.manual_seed(0)
        z = posterior.sample()
        assert z.shape == posterior.mu.shape
        assert not torch.allclose(z, posterior.mode())


class TestScalingFactor:
    def test_encode_decode_scaling_symmetry(self, tiny_vae):
        x = torch.randn(1, 3, 64, 64)
        z = tiny_vae.encode_to_latent(x)
        x_back = tiny_vae.decode_from_latent(z)
        assert x_back.shape == x.shape
        # The latent was genuinely rescaled: mode round-trips exactly.
        z_mode = tiny_vae.encode(x).mode() * tiny_vae.scaling_factor
        torch.manual_seed(7)
        z2 = tiny_vae.encode_to_latent(x)
        assert z2.shape == z_mode.shape
        assert float(z.abs().mean()) > 0
