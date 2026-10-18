"""Config dataclass validation: constructor-level guards fire before training."""

from __future__ import annotations

import pytest

from multimodal.config.gan_config import GANConfig
from multimodal.config.t2i_config import T2IConfig
from multimodal.config.t2v_config import T2VConfig
from multimodal.config.vae_config import VAEConfig


class TestGANConfig:
    def test_defaults_are_stylegan2_canonical(self):
        cfg = GANConfig()
        assert (cfg.beta1, cfg.beta2) == (0.0, 0.99)
        assert cfg.r1_gamma == 10.0 and cfg.lazy_reg_interval == 16

    def test_invalid_loss_type_rejected(self):
        with pytest.raises(ValueError, match="loss_type"):
            GANConfig(loss_type="bce")

    def test_non_power_of_two_resolution_rejected(self):
        with pytest.raises(ValueError, match="power of two"):
            GANConfig(img_resolution=100)

    @pytest.mark.parametrize("resolution", [64, 256, 1024])
    def test_power_of_two_resolutions_accepted(self, resolution):
        assert GANConfig(img_resolution=resolution).img_resolution == resolution

    def test_optimizer_defaults_wired(self):
        cfg = GANConfig()
        assert cfg.optimizer.learning_rate == cfg.gen_lr


class TestVAEConfig:
    def test_ldm_kl_weight_default(self):
        # The LDM finding baked into defaults (see docs/models/vae.md).
        assert VAEConfig().kl_weight == 1e-6

    def test_optimizer_lr_matches_paper(self):
        assert VAEConfig().optimizer.learning_rate == 4.5e-6


class TestT2IConfig:
    def test_latent_geometry_consistent(self):
        cfg = T2IConfig()
        assert cfg.latent_size == cfg.image_size // 8

    def test_caption_dropout_for_cfg(self):
        assert 0 < T2IConfig().caption_dropout_prob < 1


class TestT2VConfig:
    def test_temporal_compression(self):
        cfg = T2VConfig()
        assert cfg.num_frames % cfg.temporal_compression == 0

    def test_memory_regime_defaults(self):
        cfg = T2VConfig()
        assert cfg.gradient_checkpointing is True
        assert cfg.latent_caching is True

    def test_effective_batch(self):
        cfg = T2VConfig()
        assert cfg.batch_size * cfg.gradient_accumulation_steps == 8
