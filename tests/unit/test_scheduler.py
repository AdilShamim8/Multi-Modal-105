"""Noise scheduler math — the invariants every sampler depends on."""

from __future__ import annotations

import pytest
import torch

from multimodal.models.diffusion.noise_scheduler import NoiseScheduler

SCHEDULES = ["linear", "scaled_linear", "cosine"]


class TestScheduleInvariants:
    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_alphas_cumprod_monotonic_decreasing(self, schedule):
        s = NoiseScheduler(beta_schedule=schedule)
        ac = s.alphas_cumprod
        assert torch.all(ac[:-1] >= ac[1:]), f"{schedule}: ᾱ must decrease"

    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_alphas_cumprod_bounded(self, schedule):
        s = NoiseScheduler(beta_schedule=schedule)
        ac = s.alphas_cumprod
        assert torch.all(ac > 0) and torch.all(ac <= 1)

    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_betas_positive(self, schedule):
        s = NoiseScheduler(beta_schedule=schedule)
        assert torch.all(s.betas > 0)
        assert torch.all(s.betas < 1)

    def test_cosine_ends_near_zero(self):
        s = NoiseScheduler(beta_schedule="cosine")
        assert s.alphas_cumprod[-1] < 0.01  # terminal SNR ≈ 0 for cosine


class TestForwardProcess:
    def test_add_noise_at_t0_is_identityish(self, tiny_scheduler):
        x0 = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(x0)
        t = torch.zeros(2, dtype=torch.long)
        xt = tiny_scheduler.add_noise(x0, noise, t)
        # atol reflects sqrt(1 - ᾱ_0) = sqrt(β_0) ≈ 0.01 with |noise| ≤ ~4.
        assert torch.allclose(xt, x0, atol=0.05)

    def test_add_noise_terminal_dominated_by_noise(self):
        """On the default 1000-step schedule, x_T is ~pure noise: the
        correlation with the injected noise must dominate the signal."""
        s = NoiseScheduler(beta_schedule="scaled_linear")
        torch.manual_seed(0)
        x0 = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(x0)
        t = torch.full((2,), s.num_train_timesteps - 1, dtype=torch.long)
        xt = s.add_noise(x0, noise, t)

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.flatten(), b.flatten(), dim=0
            ).item()

        assert cos(xt, noise) > 0.98
        assert cos(xt, x0) < 0.15

    def test_variance_scaling_identity(self, tiny_scheduler):
        """Var[q(x_t|x_0=0)] = 1 - ᾱ_t exactly — the schedule's core promise."""
        torch.manual_seed(0)
        x0 = torch.zeros(2048, 4, 4, 4)
        noise = torch.randn_like(x0)
        for t in [0, 25, 50, 75, 99]:
            xt = tiny_scheduler.add_noise(
                x0, noise, torch.full((2048,), t, dtype=torch.long)
            )
            expected = float(tiny_scheduler.sqrt_one_minus_alphas_cumprod[t])
            assert abs(xt.std().item() - expected) < 0.03, f"t={t}"


class TestPredictionConversions:
    def test_velocity_roundtrip(self):
        s = NoiseScheduler(prediction_type="v_prediction")
        torch.manual_seed(0)
        x0 = torch.randn(2, 4, 8, 8)
        eps = torch.randn_like(x0)
        t = torch.randint(0, 1000, (2,))
        v = s.get_velocity(x0, eps, t)
        xt = s.add_noise(x0, eps, t)
        x0_back = s.predict_x0_from_velocity(xt, t, v)
        assert torch.allclose(x0_back, x0, atol=1e-3)

    def test_x0_from_noise_roundtrip(self):
        s = NoiseScheduler(prediction_type="epsilon")
        torch.manual_seed(0)
        x0 = torch.randn(2, 4, 8, 8)
        eps = torch.randn_like(x0)
        t = torch.randint(0, 1000, (2,))
        xt = s.add_noise(x0, eps, t)
        x0_back = s.predict_x0_from_noise(xt, t, eps)
        assert torch.allclose(x0_back, x0, atol=1e-3)

    def test_noise_from_x0_roundtrip(self):
        s = NoiseScheduler(prediction_type="epsilon")
        torch.manual_seed(0)
        x0 = torch.randn(2, 4, 8, 8)
        eps = torch.randn_like(x0)
        t = torch.randint(0, 1000, (2,))
        xt = s.add_noise(x0, eps, t)
        eps_back = s.predict_noise_from_x0(xt, t, x0)
        assert torch.allclose(eps_back, eps, atol=1e-3)


class TestPosterior:
    def test_variance_positive(self, tiny_scheduler):
        v = tiny_scheduler.posterior_variance[1:]  # t=0 posterior is degenerate
        assert torch.all(v > 0)

    def test_set_timesteps_spaced(self, tiny_scheduler):
        tiny_scheduler.set_timesteps(10)
        ts = tiny_scheduler.timesteps
        assert len(ts) == 10
        assert ts[0] > ts[-1]  # descending: sampling walks T → 0
        assert ts.max() < tiny_scheduler.num_train_timesteps
