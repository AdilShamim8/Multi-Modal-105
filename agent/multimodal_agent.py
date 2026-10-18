import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Literal
import logging
import json
import time

from multimodal.models.vae.vae import AutoencoderKL
from multimodal.models.compression.video_vae import VideoAutoencoderKL
from multimodal.models.diffusion.dit import DiT, EMAModel
from multimodal.models.diffusion.dit_video import VideoDiT
from multimodal.models.diffusion.noise_scheduler import NoiseScheduler
from multimodal.models.text_encoder import CLIPTextEncoder, T5TextEncoder
from sampling.ddim_sampler import DDIMSampler
from sampling.dpm_solver import DPMSolver
from sampling.ddpm_sampler import DDPMSampler
from agent.pipeline import T2IPipeline

logger = logging.getLogger(__name__)


class MultimodalGenerationAgent:
    """
    Production Multi-modal Generation Agent.
    
    Unified interface for:
    - Text-to-Image (T2I) generation
    - Text-to-Video (T2V) generation
    - Image-to-Image (I2I) editing
    - Image-to-Video (I2V) animation
    - Video-to-Video (V2V) editing (future)
    
    System overview:
    ┌─────────────────────────────────────────────┐
    │           MultimodalGenerationAgent         │
    │                                             │
    │  ┌──────────┐   ┌──────────┐   ┌────────┐  │
    │  │ T2I      │   │ T2V      │   │ I2V    │  │
    │  │ Pipeline │   │ Pipeline │   │ Future │  │
    │  └────┬─────┘   └────┬─────┘   └───┬────┘  │
    │       │              │              │       │
    │  ┌────┴──────────────┴──────────────┴────┐  │
    │  │           Shared Components            │  │
    │  │  VAE │ VideoVAE │ TextEncoder │ Sched  │  │
    │  └───────────────────────────────────────┘  │
    └─────────────────────────────────────────────┘
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        # Model paths
        t2i_model_path: Optional[str] = None,
        t2v_model_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        video_vae_path: Optional[str] = None,
        text_encoder_path: str = "openai/clip-vit-large-patch14",
        t5_encoder_path: Optional[str] = None,
        # Defaults
        default_t2i_steps: int = 30,
        default_t2v_steps: int = 50,
        default_guidance_scale: float = 7.5,
        default_sampler: str = "dpm++",
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.default_t2i_steps = default_t2i_steps
        self.default_t2v_steps = default_t2v_steps
        self.default_guidance_scale = default_guidance_scale

        logger.info(f"Initializing MultimodalGenerationAgent on {device}")

        # ── Shared noise scheduler ────────────────────────────────────────
        self.scheduler = NoiseScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="v_prediction",
        )

        # ── Text encoders ─────────────────────────────────────────────────
        self.clip_encoder = CLIPTextEncoder(
            model_name=text_encoder_path,
            max_length=77,
            freeze=True,
        ).to(self.device)
        self.clip_encoder.eval()

        self.t5_encoder = None
        if t5_encoder_path:
            self.t5_encoder = T5TextEncoder(
                model_name=t5_encoder_path,
                max_length=256,
                freeze=True,
            ).to(self.device)
            self.t5_encoder.eval()

        # ── Image VAE ─────────────────────────────────────────────────────
        self.image_vae = AutoencoderKL(
            in_channels=3,
            latent_channels=4,
            base_channels=128,
            channel_multipliers=(1, 2, 4, 4),
            num_res_blocks=2,
        ).to(self.device)
        if vae_path and Path(vae_path).exists():
            self._load_model_weights(self.image_vae, vae_path)
        self.image_vae.eval()

        # ── Video VAE ─────────────────────────────────────────────────────
        self.video_vae = VideoAutoencoderKL(
            in_channels=3,
            latent_channels=16,
            temporal_compression=4,
            spatial_compression=8,
        ).to(self.device)
        if video_vae_path and Path(video_vae_path).exists():
            self._load_model_weights(self.video_vae, video_vae_path)
        self.video_vae.eval()

        # ── T2I Model ─────────────────────────────────────────────────────
        self.t2i_model = None
        if t2i_model_path:
            self.t2i_model = self._load_t2i_model(t2i_model_path)

        # ── T2V Model ─────────────────────────────────────────────────────
        self.t2v_model = None
        if t2v_model_path:
            self.t2v_model = self._load_t2v_model(t2v_model_path)

        # ── Samplers ─────────────────────────────────────────────────────
        self.ddim_sampler = DDIMSampler(self.scheduler)
        self.dpm_sampler = DPMSolver(self.scheduler)
        self.ddpm_sampler = DDPMSampler(self.scheduler)
        self.default_sampler = default_sampler

        # ── T2I Pipeline ──────────────────────────────────────────────────
        self.t2i_pipeline = None
        if self.t2i_model is not None:
            self.t2i_pipeline = T2IPipeline(
                model=self.t2i_model,
                vae=self.image_vae,
                text_encoder=self.clip_encoder,
                scheduler=self.scheduler,
                device=str(self.device),
                sampler=default_sampler,
                default_num_steps=default_t2i_steps,
                default_guidance_scale=default_guidance_scale,
            )

        logger.info("MultimodalGenerationAgent ready")

    def _load_model_weights(self, model: torch.nn.Module, path: str):
        """Load model weights with fallback strategies."""
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.debug(f"Missing keys: {len(missing)}")
            logger.info(f"Loaded weights from {path}")
        except Exception as e:
            logger.warning(f"Failed to load weights from {path}: {e}")

    def _load_t2i_model(self, path: str) -> DiT:
        """Load T2I DiT model."""
        model = DiT(
            input_size=64,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            context_dim=768,
        ).to(self.device)
        self._load_model_weights(model, path)
        model.eval()
        return model

    def _load_t2v_model(self, path: str) -> VideoDiT:
        """Load T2V Video DiT model."""
        model = VideoDiT(
            num_frames=16,
            frame_size=32,
            in_channels=16,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            context_dim=768,
        ).to(self.device)
        self._load_model_weights(model, path)
        model.eval()
        return model

    # ─────────────────────────────────────────────────────────────────────────
    # Text-to-Image
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_image(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        sampler: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate image(s) from text prompt.
        
        Args:
            prompt:              Text description of desired image
            negative_prompt:     Text to avoid in generation
            height, width:       Output image resolution
            num_inference_steps: Denoising steps (more = higher quality)
            guidance_scale:      CFG scale (higher = more prompt adherence)
            num_images_per_prompt: Batch size per prompt
            seed:                Random seed for reproducibility
            sampler:             Override default sampler
        Returns:
            dict with 'images' (List[PIL.Image]), 'metadata'
        """
        if self.t2i_pipeline is None:
            raise RuntimeError("T2I model not loaded. Provide t2i_model_path.")

        start_time = time.time()

        if sampler and sampler != self.default_sampler:
            self.t2i_pipeline.sampler = self._get_sampler(sampler)

        result = self.t2i_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps or self.default_t2i_steps,
            guidance_scale=guidance_scale or self.default_guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            seed=seed,
        )

        elapsed = time.time() - start_time
        result["metadata"] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "steps": num_inference_steps or self.default_t2i_steps,
            "guidance_scale": guidance_scale or self.default_guidance_scale,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "sampler": sampler or self.default_sampler,
        }
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Text-to-Video
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_video(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_frames: int = 16,
        frame_size: int = 256,
        fps: int = 8,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        output_format: str = "gif",        # "gif" | "mp4" | "frames"
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt.
        
        Args:
            prompt:              Text description of desired video
            num_frames:          Number of frames to generate
            frame_size:          Spatial resolution (square)
            fps:                 Output video frame rate
            num_inference_steps: Denoising steps
            guidance_scale:      CFG scale
            seed:                Reproducibility seed
            output_format:       Output format: "gif", "mp4", or "frames"
            output_path:         Where to save output
        Returns:
            dict with 'video_path', 'frames' (numpy array), 'metadata'
        """
        if self.t2v_model is None:
            raise RuntimeError("T2V model not loaded. Provide t2v_model_path.")

        if seed is not None:
            torch.manual_seed(seed)

        if isinstance(prompt, str):
            prompt = [prompt]
        if negative_prompt is None:
            negative_prompt = [""] * len(prompt)

        B = len(prompt)
        start_time = time.time()

        # ── Encode text ───────────────────────────────────────────────────
        text_encoder = self.t5_encoder or self.clip_encoder
        if self.t5_encoder:
            pos_tokens = self.t5_encoder.tokenize(prompt)
            neg_tokens = self.t5_encoder.tokenize(negative_prompt)
            pos_embeds = self.t5_encoder(
                pos_tokens["input_ids"].to(self.device),
                pos_tokens.get("attention_mask", None),
            )
            neg_embeds = self.t5_encoder(
                neg_tokens["input_ids"].to(self.device),
                neg_tokens.get("attention_mask", None),
            )
        else:
            pos_tokens = self.clip_encoder.tokenize(prompt, device=self.device)
            neg_tokens = self.clip_encoder.tokenize(negative_prompt, device=self.device)
            pos_embeds = self.clip_encoder.encode(pos_tokens["input_ids"])
            neg_embeds = self.clip_encoder.encode(neg_tokens["input_ids"])

        pos_pool = pos_embeds.mean(dim=1)
        neg_pool = neg_embeds.mean(dim=1)

        # ── Latent shape ──────────────────────────────────────────────────
        T_lat = num_frames // self.video_vae.temporal_compression
        H_lat = frame_size // self.video_vae.spatial_compression
        W_lat = frame_size // self.video_vae.spatial_compression
        shape = (B, 16, T_lat, H_lat, W_lat)

        latents = torch.randn(*shape, device=self.device)

        # ── DDIM denoising loop ───────────────────────────────────────────
        n_steps = num_inference_steps or self.default_t2v_steps
        cfg_scale = guidance_scale or self.default_guidance_scale
        self.scheduler.set_timesteps(n_steps, str(self.device))

        for i, t_cur in enumerate(self.scheduler.timesteps):
            t_batch = t_cur.expand(B)

            # CFG: run cond + uncond
            model_out_cond = self.t2v_model(
                latents, t_batch,
                text_embeds=pos_embeds,
                text_pool=pos_pool,
            )
            model_out_uncond = self.t2v_model(
                latents, t_batch,
                text_embeds=neg_embeds,
                text_pool=neg_pool,
            )

            # Handle learned sigma
            if model_out_cond.shape[1] == 16 * 2:
                model_out_cond, _ = model_out_cond.chunk(2, dim=1)
                model_out_uncond, _ = model_out_uncond.chunk(2, dim=1)

            # Guidance
            model_out = model_out_uncond + cfg_scale * (
                model_out_cond - model_out_uncond
            )

            # DDIM step
            x0_pred = self.scheduler.get_x0_prediction(model_out, latents, t_batch)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            t_prev_idx = i + 1 if i < len(self.scheduler.timesteps) - 1 \
                         else len(self.scheduler.timesteps) - 1
            t_prev = self.scheduler.timesteps[t_prev_idx] if i < len(
                self.scheduler.timesteps) - 1 else 0

            t_prev_val = t_prev.item() if hasattr(t_prev, "item") else int(t_prev)
            t_cur_val = t_cur.item()

            alpha_t = self.scheduler.alphas_cumprod[t_cur_val].to(self.device)
            alpha_prev = self.scheduler.alphas_cumprod[max(t_prev_val, 0)].to(self.device)

            eps = self.scheduler.predict_noise_from_x0(latents, t_batch, x0_pred)
            dir_xt = (1 - alpha_prev).clamp(min=0).sqrt() * eps
            latents = alpha_prev.sqrt() * x0_pred + dir_xt

        # ── Decode video latents ──────────────────────────────────────────
        videos = self.video_vae.decode_from_latent(latents)   # (B, 3, T, H, W)
        videos = ((videos.clamp(-1, 1) + 1) / 2 * 255).byte()

        # ── Save output ───────────────────────────────────────────────────
        elapsed = time.time() - start_time
        saved_paths = []
        all_frames = []

        for b in range(B):
            frames_np = videos[b].permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
            all_frames.append(frames_np)

            if output_path:
                out_p = Path(output_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                if B > 1:
                    out_p = out_p.with_stem(f"{out_p.stem}_{b}")
            else:
                out_p = Path(f"generated_video_{b}_{int(time.time())}.{output_format}")

            if output_format == "gif":
                out_file = str(out_p.with_suffix(".gif"))
                imageio.mimwrite(out_file, frames_np, fps=fps, loop=0)
            elif output_format == "mp4":
                out_file = str(out_p.with_suffix(".mp4"))
                imageio.mimwrite(
                    out_file, frames_np,
                    fps=fps,
                    quality=8,
                    macro_block_size=1,
                )
            elif output_format == "frames":
                frames_dir = str(out_p.with_suffix(""))
                Path(frames_dir).mkdir(exist_ok=True)
                for fi, frame in enumerate(frames_np):
                    Image.fromarray(frame).save(
                        f"{frames_dir}/frame_{fi:04d}.png"
                    )
                out_file = frames_dir

            saved_paths.append(out_file)

        return {
            "video_paths": saved_paths,
            "frames": all_frames,        # List of (T, H, W, 3) numpy arrays
            "metadata": {
                "prompts": prompt,
                "num_frames": num_frames,
                "frame_size": frame_size,
                "fps": fps,
                "steps": n_steps,
                "guidance_scale": cfg_scale,
                "seed": seed,
                "elapsed_seconds": elapsed,
                "format": output_format,
            }
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Image-to-Image
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def edit_image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Image-to-image editing via DDIM inversion.
        strength: 0 = preserve input, 1 = full generation
        """
        if self.t2i_pipeline is None:
            raise RuntimeError("T2I model not loaded.")

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        edited = self.t2i_pipeline.img2img(
            prompt=prompt,
            init_image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        return {
            "images": edited,
            "metadata": {
                "prompt": prompt,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "steps": num_inference_steps,
                "seed": seed,
            }
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Utility methods
    # ─────────────────────────────────────────────────────────────────────────

    def _get_sampler(self, name: str):
        if name == "ddim":
            return self.ddim_sampler
        elif name in ("dpm++", "dpm_solver"):
            return self.dpm_sampler
        elif name == "ddpm":
            return self.ddpm_sampler
        else:
            raise ValueError(f"Unknown sampler: {name}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model parameter counts and configuration."""
        info = {}
        if self.t2i_model:
            info["t2i_params"] = sum(
                p.numel() for p in self.t2i_model.parameters()
            ) / 1e6
        if self.t2v_model:
            info["t2v_params"] = sum(
                p.numel() for p in self.t2v_model.parameters()
            ) / 1e6
        info["image_vae_params"] = sum(
            p.numel() for p in self.image_vae.parameters()
        ) / 1e6
        info["video_vae_params"] = sum(
            p.numel() for p in self.video_vae.parameters()
        ) / 1e6
        return info

    def save_config(self, path: str):
        """Save agent configuration to JSON."""
        config = {
            "device": str(self.device),
            "default_t2i_steps": self.default_t2i_steps,
            "default_t2v_steps": self.default_t2v_steps,
            "default_guidance_scale": self.default_guidance_scale,
            "default_sampler": self.default_sampler,
            "model_info": self.get_model_info(),
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Agent config saved to {path}")

    @classmethod
    def from_config(cls, config_path: str, **kwargs) -> "MultimodalGenerationAgent":
        """Load agent from JSON config file."""
        with open(config_path) as f:
            config = json.load(f)
        config.update(kwargs)
        return cls(**config)


def main():
    """Entry point: demonstrate the full agent."""
    import argparse

    parser = argparse.ArgumentParser("MultimodalGenerationAgent Demo")
    parser.add_argument("--mode", choices=["t2i", "t2v", "i2i"], default="t2i")
    parser.add_argument("--prompt", type=str, default="a beautiful sunset over the ocean")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t2i_model_path", type=str, default=None)
    parser.add_argument("--t2v_model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_size", type=int, default=256)
    args = parser.parse_args()

    # Initialize agent
    agent = MultimodalGenerationAgent(
        device=args.device,
        t2i_model_path=args.t2i_model_path,
        t2v_model_path=args.t2v_model_path,
        default_t2i_steps=args.steps,
        default_t2v_steps=args.steps,
        default_guidance_scale=args.guidance_scale,
    )

    print(f"Agent initialized. Model info: {agent.get_model_info()}")
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.mode == "t2i":
        result = agent.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=512, width=512,
            seed=args.seed,
        )
        for i, img in enumerate(result["images"]):
            out_path = str(Path(args.output) / f"image_{i:04d}.png")
            img.save(out_path)
            print(f"Saved: {out_path}")
        print(f"Metadata: {result['metadata']}")

    elif args.mode == "t2v":
        result = agent.generate_video(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_frames=args.num_frames,
            frame_size=args.frame_size,
            fps=8,
            seed=args.seed,
            output_format="gif",
            output_path=str(Path(args.output) / "video.gif"),
        )
        print(f"Generated videos: {result['video_paths']}")
        print(f"Metadata: {result['metadata']}")

    elif args.mode == "i2i":
        print("i2i requires --image argument (not shown for brevity)")

    agent.save_config(str(Path(args.output) / "agent_config.json"))


if __name__ == "__main__":
    main()