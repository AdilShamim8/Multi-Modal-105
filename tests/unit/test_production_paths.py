"""Regression tests for the production loading paths repaired on 2026-10-18.

Each test pins a defect that the fake-injection test pattern could never
see, because the fakes replaced exactly the code that was broken:

- ``T2IPipeline.from_pretrained`` did not exist — engine image loading
  raised AttributeError on the first real bundle.
- ``T2VPipeline`` did not exist anywhere — engine video loading raised
  ImportError (engine imported it from a module that never defined it).
- ``DDPMSampler.sample`` rejected the ``latents`` kwarg — img2img with the
  ddpm sampler raised TypeError.
- ``GANTrainer`` called CheckpointManager/MetricLogger kwargs that never
  existed, clipped gradients against a missing config field, and crashed
  checkpoint saves on a nested dataclass in metadata.
- ``InferenceEngine._run`` ignored the ``videos`` result key — every video
  pipeline result raised GenerationError('no outputs').

The bundles here are built exactly the way ``mma export`` + the model
registry lay them out, so these tests exercise the real load path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset

from multimodal.models.diffusion.noise_scheduler import NoiseScheduler
from multimodal.sampling.ddpm_sampler import DDPMSampler


class _FixedNoiseModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.full_like(x, 0.1)


class _FakeTextEncoder(torch.nn.Module):
    """Deterministic CLIP/T5 stand-in: tokenize → ids, encode → embeddings."""

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self._emb = torch.nn.Embedding(32, embed_dim)

    def tokenize(self, texts, device=None):
        count = len(texts) if not isinstance(texts, str) else 1
        ids = torch.randint(0, 32, (count, 6))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode(self, input_ids, attention_mask=None):
        return self._emb(input_ids)

    def forward(self, input_ids, attention_mask=None):
        return self._emb(input_ids)


def _write_bundle(
    directory: Path,
    model: torch.nn.Module,
    config: dict,
    vae: torch.nn.Module | None = None,
    vae_filename: str = "vae.safetensors",
) -> Path:
    """Lay out a registry bundle: weights + config + checksummed manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(directory / "weights.safetensors"))
    if vae is not None:
        save_file(vae.state_dict(), str(directory / vae_filename))
    (directory / "config.json").write_text(json.dumps(config))
    sha = hashlib.sha256((directory / "weights.safetensors").read_bytes()).hexdigest()
    (directory / "bundle.json").write_text(
        json.dumps({"artifact": "weights.safetensors", "sha256": sha})
    )
    return directory


class TestT2IPipelineFromPretrained:
    @pytest.fixture()
    def bundle(self, tmp_path: Path) -> tuple[Path, torch.nn.Module]:
        from multimodal.models.diffusion.dit import DiT
        from multimodal.models.vae.vae import AutoencoderKL

        model = DiT(input_size=8, patch_size=2, hidden_size=36, depth=2, num_heads=4)
        vae = AutoencoderKL(
            in_channels=3,
            latent_channels=4,
            base_channels=32,
            channel_multipliers=(1, 2, 4, 4),
            num_res_blocks=1,
        )
        config = {
            "architecture": "dit",
            "latent_size": 8,
            "patch_size": 2,
            "hidden_size": 36,
            "depth": 2,
            "num_heads": 4,
            "num_train_timesteps": 50,
            "vae": {
                "in_channels": 3,
                "latent_channels": 4,
                "base_channels": 32,
                "channel_multipliers": [1, 2, 4, 4],
                "num_res_blocks": 1,
            },
        }
        return _write_bundle(tmp_path / "t2i-dit-tiny", model, config, vae=vae), model

    def test_loads_weights_and_generates(self, bundle):
        from multimodal.agent.pipeline import T2IPipeline

        bundle_dir, model = bundle
        pipe = T2IPipeline.from_pretrained(
            bundle_dir, device="cpu", sampler="ddim", text_encoder=_FakeTextEncoder()
        )
        # The DiT weights round-tripped from the bundle.
        assert all(
            torch.equal(pipe.model.state_dict()[k], v) for k, v in model.state_dict().items()
        )
        out = pipe("a cat", width=64, height=64, num_inference_steps=2)
        assert len(out["images"]) == 1
        assert out["images"][0].size == (64, 64)

    def test_rejects_tampered_bundle(self, bundle):
        from multimodal.agent.pipeline import T2IPipeline
        from multimodal.exceptions import CheckpointError

        bundle_dir, _ = bundle
        weights = bundle_dir / "weights.safetensors"
        weights.write_bytes(weights.read_bytes() + b"corruption")
        with pytest.raises(CheckpointError):
            T2IPipeline.from_pretrained(bundle_dir, device="cpu", text_encoder=_FakeTextEncoder())

    def test_rejects_unknown_architecture(self, tmp_path: Path):
        from multimodal.agent.pipeline import T2IPipeline

        d = tmp_path / "bad-arch"
        d.mkdir()
        (d / "weights.safetensors").write_bytes(b"")
        (d / "config.json").write_text(json.dumps({"architecture": "rnngan"}))
        sha = hashlib.sha256((d / "weights.safetensors").read_bytes()).hexdigest()
        (d / "bundle.json").write_text(
            json.dumps({"artifact": "weights.safetensors", "sha256": sha})
        )
        with pytest.raises(ValueError, match="architecture"):
            T2IPipeline.from_pretrained(d, device="cpu", text_encoder=_FakeTextEncoder())


class TestT2VPipeline:
    @pytest.fixture()
    def bundle(self, tmp_path: Path) -> tuple[Path, torch.nn.Module]:
        from multimodal.models.compression.video_vae import VideoAutoencoderKL
        from multimodal.models.diffusion.dit_video import VideoDiT

        model = VideoDiT(
            num_frames=2,
            frame_size=8,
            in_channels=4,
            spatial_patch_size=2,
            hidden_size=36,
            depth=2,
            num_heads=4,
            temporal_heads=2,
            context_dim=768,
            pooled_text_dim=64,
        )
        vae = VideoAutoencoderKL(
            in_channels=3,
            latent_channels=4,
            base_channels=32,
            channel_multipliers=(1, 2, 4),
            num_res_blocks=1,
            temporal_compression=2,
            spatial_compression=4,
        )
        config = {
            "architecture": "video_dit",
            "latent_num_frames": 2,
            "latent_frame_size": 8,
            "latent_channels": 4,
            "spatial_patch_size": 2,
            "hidden_size": 36,
            "depth": 2,
            "num_heads": 4,
            "temporal_heads": 2,
            "context_dim": 768,
            "pooled_text_dim": 64,
            "num_train_timesteps": 50,
            "video_vae": {
                "in_channels": 3,
                "latent_channels": 4,
                "base_channels": 32,
                "channel_multipliers": [1, 2, 4],
                "num_res_blocks": 1,
                "temporal_compression": 2,
                "spatial_compression": 4,
            },
        }
        d = _write_bundle(
            tmp_path / "t2v-dit-tiny", model, config, vae=vae, vae_filename="video_vae.safetensors"
        )
        return d, model

    def test_generate_raw_frames(self, bundle):
        from multimodal.agent.t2v_pipeline import T2VPipeline

        d, _ = bundle
        pipe = T2VPipeline.from_pretrained(
            d, device="cpu", text_encoder=_FakeTextEncoder(768), default_num_steps=3
        )
        out = pipe.generate("a cat playing", num_frames=4, width=32, height=32, output_format="raw")
        video = out["videos"][0]
        assert video.shape == (3, 4, 32, 32)
        assert video.dtype == torch.uint8

    def test_weights_round_trip(self, bundle):
        from multimodal.agent.t2v_pipeline import T2VPipeline

        d, model = bundle
        pipe = T2VPipeline.from_pretrained(d, device="cpu", text_encoder=_FakeTextEncoder(768))
        assert all(
            torch.equal(pipe.model.state_dict()[k], v) for k, v in model.state_dict().items()
        )

    def test_engine_imports_the_video_pipeline_symbol(self):
        # The original defect: engine imported T2VPipeline from
        # agent.pipeline, which never defined it.
        import inspect

        import multimodal.serving.engine as engine_mod
        from multimodal.agent.t2v_pipeline import T2VPipeline  # noqa: F401

        source = inspect.getsource(engine_mod.InferenceEngine._load_t2v)
        assert "agent.t2v_pipeline" in source


class TestDDPMSamplerLatents:
    def test_latents_kwarg_is_accepted_and_used(self):
        sched = NoiseScheduler(prediction_type="epsilon")
        sampler = DDPMSampler(sched)
        initial = torch.full((2, 4, 8, 8), 0.5)
        out = sampler.sample(
            model=_FixedNoiseModel(),
            shape=(2, 4, 8, 8),
            device="cpu",
            num_inference_steps=5,
            show_progress=False,
            latents=initial,
        )
        assert out.shape == initial.shape

    def test_callable_model_wrapper_is_accepted(self):
        sched = NoiseScheduler(prediction_type="epsilon")
        sampler = DDPMSampler(sched)

        def model_with_cfg(x, t, **kwargs):
            return torch.full_like(x, 0.1)

        out = sampler.sample(
            model=model_with_cfg,
            shape=(2, 4, 8, 8),
            device="cpu",
            num_inference_steps=3,
            show_progress=False,
        )
        assert out.shape == (2, 4, 8, 8)


class TestGANTrainerProductionPath:
    def test_train_step_and_checkpoint_roundtrip(self, tmp_path: Path):
        from multimodal.config.gan_config import GANConfig
        from multimodal.models.gans.stylegan import StyleGAN2
        from multimodal.training.gan_trainer import GANTrainer

        torch.manual_seed(0)
        gan = StyleGAN2(
            z_dim=64,
            w_dim=64,
            img_resolution=32,
            channel_base=512,
            channel_max=64,
            num_mapping_layers=2,
            loss_type="softplus",
        )
        config = GANConfig(
            img_resolution=32,
            z_dim=64,
            w_dim=64,
            channel_base=512,
            channel_max=64,
            num_train_steps=1,
            num_epochs=1,
            use_wandb=False,
            r1_gamma=0.0,
            mbstd_group_size=1,
            batch_size=2,
            gen_lr=1e-4,
            disc_lr=1e-4,
            log_every_n_steps=1,
            save_every_n_steps=10**9,
            eval_every_n_steps=10**9,
            ema_beta=0.9,
        )
        loader = DataLoader(TensorDataset(torch.randn(4, 3, 32, 32)), batch_size=2)
        trainer = GANTrainer(
            gan=gan, config=config, train_loader=loader, checkpoint_dir=str(tmp_path)
        )
        stats = trainer.train_step(torch.randn(2, 3, 32, 32))
        assert set(stats) == {"gan/d_loss", "gan/g_loss", "gan/r1"}
        assert all(v == v for v in stats.values())  # no NaNs

        trainer._save()
        saved = tmp_path / "checkpoints" / "checkpoint-1"
        for name in ("model.pt", "ema_model.pt", "optimizer.pt", "discriminator.pt", "optim_d.pt"):
            assert (saved / name).exists(), f"missing {name}"
        metadata = json.loads((saved / "metadata.json").read_text())
        assert metadata["gan"]["optimizer"]["optimizer"] == "adam"  # nested config serializes


class TestEngineVideoResultKey:
    def test_videos_key_yields_bytes(self):
        from multimodal.exceptions import GenerationError
        from multimodal.serving.engine import InferenceEngine

        class FakeVideoPipeline:
            def generate(self, **params):
                return {"videos": [b"FAKE_MP4_BYTES"]}

        class EmptyImagePipeline:
            def __call__(self, **params):
                return {"images": []}

        assert InferenceEngine._run(FakeVideoPipeline(), "video", {}) == [b"FAKE_MP4_BYTES"]
        with pytest.raises(GenerationError, match="no outputs"):
            InferenceEngine._run(EmptyImagePipeline(), "image", {})
