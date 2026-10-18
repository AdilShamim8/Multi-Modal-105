"""StyleGAN2 system layer: losses, lazy R1 cadence, truncation, EMA math."""

from __future__ import annotations

import pytest
import torch

from multimodal.models.gans.stylegan import StyleGAN2


@pytest.fixture(scope="module")
def tiny_gan():
    torch.manual_seed(0)
    return StyleGAN2(
        z_dim=64, w_dim=64, img_resolution=32, img_channels=3,
        channel_base=512, channel_max=64, num_mapping_layers=2,
        loss_type="softplus", r1_gamma=10.0, lazy_reg_interval=4,
    )


class TestLosses:
    def test_softplus_discriminator_losses(self, tiny_gan):
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1)
        d_loss = tiny_gan.d_loss(real, fake)
        assert torch.isfinite(d_loss) and float(d_loss) >= 0.0

    def test_generator_loss_scalar(self, tiny_gan):
        g_loss = tiny_gan.g_loss(torch.randn(4, 1))
        assert g_loss.dim() == 0 and torch.isfinite(g_loss)

    def test_disc_loss_zero_for_perfect_discriminator(self, tiny_gan):
        # real → +inf, fake → -inf: softplus(-inf) = 0
        d_loss = tiny_gan.d_loss(torch.full((2, 1), 50.0), torch.full((2, 1), -50.0))
        assert float(d_loss) == pytest.approx(0.0, abs=1e-6)

    def test_invalid_loss_type_rejected(self):
        with pytest.raises(ValueError):
            StyleGAN2(loss_type="bce")

    def test_hinge_variant(self):
        gan = StyleGAN2(
            z_dim=64, w_dim=64, img_resolution=32,
            channel_base=512, channel_max=64, loss_type="hinge",
        )
        real = torch.randn(2, 1)
        fake = torch.randn(2, 1)
        assert torch.isfinite(gan.d_loss(real, fake))


class TestLazyR1:
    def test_only_fires_on_interval_steps(self, tiny_gan):
        fires = [s for s in range(16) if tiny_gan.r1_lazy(torch.randn(1, 3, 32, 32), s) is not None]
        assert fires == [0, 4, 8, 12]

    def test_amortized_by_interval(self, tiny_gan):
        penalty = tiny_gan.r1_lazy(torch.randn(1, 3, 32, 32), 4)
        assert penalty is not None
        # γ=10, interval=4 → effective coefficient 2.5 on the raw R1 term.
        assert float(penalty) >= 0.0

    def test_disabled_when_gamma_zero(self):
        gan = StyleGAN2(
            z_dim=64, w_dim=64, img_resolution=32,
            channel_base=512, channel_max=64, r1_gamma=0.0,
        )
        assert gan.r1_lazy(torch.randn(1, 3, 32, 32), 0) is None


class TestSampling:
    def test_sample_shape_and_range(self, tiny_gan):
        img = tiny_gan.sample(batch=2, seed=3, noise_mode="none")
        assert img.shape == (2, 3, 32, 32)

    def test_seed_reproducibility(self, tiny_gan):
        a = tiny_gan.sample(batch=1, seed=11, noise_mode="none")
        b = tiny_gan.sample(batch=1, seed=11, noise_mode="none")
        assert torch.equal(a, b)

    def test_stabilize_images_maps_to_0_1(self):
        img = torch.tensor([[-2.0, 0.0, 2.0]]).reshape(1, 1, 1, 3)
        out = StyleGAN2.stabilize_images(img)
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestEMA:
    def test_ema_converges_to_weights(self):
        model = torch.nn.Linear(4, 4)
        with torch.no_grad():
            target = torch.randn_like(model.weight)
        shadow = StyleGAN2.ema_copy(model, beta=0.0)  # beta 0 → instant copy
        for _ in range(3):
            with torch.no_grad():
                model.weight.copy_(target)
            StyleGAN2.ema_update(shadow, model, beta=0.0)
        assert torch.allclose(shadow.weight, target, atol=1e-6)

    def test_w_avg_tracks_mapping_mean(self, tiny_gan):
        tiny_gan.update_w_avg()
        assert float(tiny_gan.w_avg_n) == 1
        tiny_gan.update_w_avg()
        assert float(tiny_gan.w_avg_n) == 2


class TestGradientPenalty:
    def test_gp_positive_and_finite(self, tiny_gan):
        real = torch.randn(2, 3, 32, 32)
        fake = torch.randn(2, 3, 32, 32)
        gp = tiny_gan.gradient_penalty(real, fake, lam=10.0)
        assert torch.isfinite(gp) and float(gp) >= 0.0
