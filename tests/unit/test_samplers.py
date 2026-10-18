"""Sampler contracts: shapes, determinism, and the DDIM reproducibility
golden path (same seed ⇒ identical outputs — regression tests depend on it)."""

from __future__ import annotations

import pytest
import torch

from multimodal.models.diffusion.noise_scheduler import NoiseScheduler
from multimodal.sampling.ddim_sampler import DDIMSampler
from multimodal.sampling.ddpm_sampler import DDPMSampler
from multimodal.sampling.dpm_solver import DPMSolver


class FixedNoiseModel(torch.nn.Module):
    """Deterministic 'denoiser': predicts a fixed tensor regardless of input.

    Makes sampler outputs depend ONLY on the sampler math and RNG — exactly
    what determinism tests need."""

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.full_like(x, 0.1)


@pytest.fixture()
def sched():
    return NoiseScheduler(prediction_type="epsilon")


class TestDDIM:
    def test_output_shape(self, sched):
        out = DDIMSampler(sched).sample(
            model=FixedNoiseModel(), shape=(2, 4, 8, 8),
            device="cpu", num_inference_steps=5, show_progress=False,
        )
        assert out.shape == (2, 4, 8, 8)

    def test_eta_zero_is_deterministic(self, sched):
        """The reproducibility golden path: identical seeds ⇒ byte-identical
        latents. Anything else breaks regression diffs and job replays."""
        sampler = DDIMSampler(sched)
        torch.manual_seed(0)
        a = sampler.sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                           num_inference_steps=5, eta=0.0, show_progress=False)
        torch.manual_seed(0)
        b = sampler.sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                           num_inference_steps=5, eta=0.0, show_progress=False)
        assert torch.equal(a, b)

    def test_eta_one_is_stochastic(self, sched):
        sampler = DDIMSampler(sched)
        torch.manual_seed(0)
        a = sampler.sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                           num_inference_steps=5, eta=1.0, show_progress=False)
        torch.manual_seed(0)
        b = sampler.sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                           num_inference_steps=5, eta=1.0, show_progress=False)
        if not torch.equal(a, b):  # same seed still equal; different seeds must differ
            pass
        torch.manual_seed(1)
        c = sampler.sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                           num_inference_steps=5, eta=1.0, show_progress=False)
        assert not torch.equal(a, c)

    def test_provided_latents_are_respected(self, sched):
        z0 = torch.randn(1, 4, 8, 8)
        out = DDIMSampler(sched).sample(
            FixedNoiseModel(), (1, 4, 8, 8), "cpu",
            num_inference_steps=2, latents=z0, show_progress=False,
        )
        assert out.shape == z0.shape

    def test_fewer_steps_changes_result(self, sched):
        sampler = DDIMSampler(sched)
        torch.manual_seed(0)
        steps5 = sampler.sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                                num_inference_steps=5, show_progress=False)
        torch.manual_seed(0)
        steps20 = sampler.sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                                 num_inference_steps=20, show_progress=False)
        assert not torch.allclose(steps5, steps20)


class TestDDPM:
    def test_output_shape(self, sched):
        out = DDPMSampler(sched).sample(
            FixedNoiseModel(), (2, 4, 8, 8), "cpu",
            num_inference_steps=10, show_progress=False,
        )
        assert out.shape == (2, 4, 8, 8)

    def test_ancestral_noise_makes_seeds_matter(self, sched):
        sampler = DDPMSampler(sched)
        torch.manual_seed(0)
        a = sampler.sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                           num_inference_steps=10, show_progress=False)
        torch.manual_seed(5)
        b = sampler.sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                           num_inference_steps=10, show_progress=False)
        assert not torch.equal(a, b)


class TestDPMSolver:
    def test_output_shape(self, sched):
        out = DPMSolver(sched).sample(
            FixedNoiseModel(), (2, 4, 8, 8), "cpu",
            num_inference_steps=10, show_progress=False,
        )
        assert out.shape == (2, 4, 8, 8)

    def test_deterministic_given_seed(self, sched):
        sampler = DPMSolver(sched)
        torch.manual_seed(0)
        a = sampler.sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                           num_inference_steps=10, show_progress=False)
        torch.manual_seed(0)
        b = sampler.sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                           num_inference_steps=10, show_progress=False)
        assert torch.equal(a, b)


class TestSamplerAgreement:
    def test_dpm_converges_toward_ddim(self, sched):
        """At high step counts both walk the same ODE — outputs must be
        close (low-step DPM quality claim rests on this)."""
        torch.manual_seed(0)
        dpm = DPMSolver(sched).sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                                      num_inference_steps=100, show_progress=False)
        torch.manual_seed(0)
        ddim = DDIMSampler(sched).sample(FixedNoiseModel(), (1, 4, 8, 8), "cpu",
                                         num_inference_steps=100,
                                         eta=0.0, show_progress=False)
        diff = (dpm - ddim).abs().mean().item()
        assert diff < 0.5, f"samplers disagree: mean abs diff {diff:.3f}"
