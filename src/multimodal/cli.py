"""Command-line interface: `mma` / `python -m multimodal.cli`.

Subcommands delegate to lazily-imported implementations so `mma --help`
stays instant even on cold containers (no torch import for help text).

Commands:
    train      launch a training plane (t2i | t2v | vae | gan | diffusion)
    generate   one-off generation from a prompt
    evaluate   run the evaluation suite (IS / FID / CLIP / diversity)
    serve      start the FastAPI serving plane
    worker     start a queue worker (GPU consumer)
    export     export a checkpoint for deployment (torchscript/onnx)
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mma",
        description="Multi-Modal Generation Platform control plane",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="multimodal-agent 1.0.0 (released 2026-10-18)",
        help="print version and exit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="launch a training plane")
    p_train.add_argument(
        "--pipeline", required=True,
        choices=["t2i", "t2v", "vae", "gan", "diffusion"],
    )
    p_train.add_argument("--config", required=True, help="YAML config path")
    p_train.add_argument("--resume", default=None, help="checkpoint to resume from")
    p_train.add_argument("--override", nargs="*", default=[],
                         help="key=value config overrides")

    # ── generate ───────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate", help="one-off generation")
    p_gen.add_argument("prompt")
    p_gen.add_argument("--negative", default=None)
    p_gen.add_argument("--modality", default="image", choices=["image", "video"])
    p_gen.add_argument("--width", type=int, default=512)
    p_gen.add_argument("--height", type=int, default=512)
    p_gen.add_argument("--steps", type=int, default=30)
    p_gen.add_argument("--guidance", type=float, default=7.5)
    p_gen.add_argument("--sampler", default="dpm++", choices=["ddpm", "ddim", "dpm++"])
    p_gen.add_argument("--outputs", type=int, default=1)
    p_gen.add_argument("--frames", type=int, default=16, help="video modality only")
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.add_argument("--model-dir", default=None, help="model registry directory")
    p_gen.add_argument("--out", default="./outputs", help="output directory")

    # ── evaluate ───────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="run the evaluation suite")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--samples-dir", required=True)
    p_eval.add_argument("--reference-dir", default=None)
    p_eval.add_argument("--metrics", nargs="*", default=["is", "fid", "clip"])

    # ── serve ──────────────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="start the API server")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--workers", type=int, default=1)
    p_serve.add_argument("--reload", action="store_true", help="dev hot reload")

    # ── worker ─────────────────────────────────────────────────────────────
    p_worker = sub.add_parser("worker", help="start a generation worker")
    p_worker.add_argument("--modality", default="image", choices=["image", "video", "all"])
    p_worker.add_argument("--max-batch", type=int, default=None)

    # ── export ─────────────────────────────────────────────────────────────
    p_export = sub.add_parser("export", help="export a checkpoint for deployment")
    p_export.add_argument("--model-dir", required=True)
    p_export.add_argument("--format", default="torchscript", choices=["torchscript", "onnx"])
    p_export.add_argument("--out", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from .version import RELEASE_DATE, __version__

        print(f"multimodal-agent {__version__} (released {RELEASE_DATE})")
        return 0

    from .logging import configure_logging

    configure_logging()

    if args.command == "serve":
        return _serve(args)
    if args.command == "worker":
        return _worker(args)
    if args.command == "train":
        return _train(args)
    if args.command == "generate":
        return _generate(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "export":
        return _export(args)
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2


# ── Command implementations (lazy imports keep help path light) ────────────
def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .serving.app import create_app

    app = create_app(load_models_at_startup=True)
    uvicorn.run(
        app,
        host=args.host or "0.0.0.0",
        port=args.port or 8000,
        reload=args.reload,  # dev only; ignored when workers > 1
        access_log=False,    # platform middleware already logs requests
    )
    return 0


def _worker(args: argparse.Namespace) -> int:
    from .serving.artifacts import build_artifact_store
    from .serving.config import ServingConfig
    from .serving.engine import InferenceEngine
    from .serving.queue import JobQueue, connect_redis
    from .serving.worker import GenerationWorker

    config = ServingConfig.from_env()
    queue = JobQueue(connect_redis(config.redis_url), config)
    engine = InferenceEngine(config)
    store = build_artifact_store(config)
    worker = GenerationWorker(
        queue=queue, engine=engine, store=store, config=config,
        max_batch=args.max_batch,
    )
    if args.modality in ("video", "all"):
        engine.load("video")
    worker.run()
    return 0


def _train(args: argparse.Namespace) -> int:
    from .config.loader import load_config, to_dataclass
    from .utils.seeds import seed_everything

    mapping = {
        "t2i": ("multimodal.config.t2i_config", "T2IConfig", "multimodal.training.t2i_trainer", "T2ITrainer"),
        "t2v": ("multimodal.config.t2v_config", "T2VConfig", "multimodal.training.t2v_trainer", "T2VTrainer"),
        "vae": ("multimodal.config.vae_config", "VAEConfig", "multimodal.training.vae_trainer", "VAETrainer"),
        "gan": ("multimodal.config.gan_config", "GANConfig", "multimodal.training.gan_trainer", "GANTrainer"),
        "diffusion": ("multimodal.config.diffusion_config", "DiffusionConfig", "multimodal.training.diffusion_trainer", "DiffusionTrainer"),
    }
    cfg_mod, cfg_cls, trainer_mod, trainer_cls = mapping[args.pipeline]
    import importlib

    cfg_type = getattr(importlib.import_module(cfg_mod), cfg_cls)
    trainer_type = getattr(importlib.import_module(trainer_mod), trainer_cls)

    payload = load_config(args.config)
    config = to_dataclass(payload, cfg_type)
    seed_everything(config.seed)
    trainer = trainer_type(config=config) if args.pipeline in ("vae",) else \
        trainer_type(config=config)
    trainer.train()
    return 0


def _generate(args: argparse.Namespace) -> int:
    from .serving.config import ServingConfig
    from .serving.engine import InferenceEngine
    from .serving.jobs import GenerationJob
    from .serving.schemas import new_job_id

    config = ServingConfig.from_env()
    if args.model_dir:
        config = config.model_copy(update={"model_dir": args.model_dir})
    engine = InferenceEngine(config)
    engine.load(args.modality)
    job = GenerationJob(
        job_id=new_job_id(),
        prompt=args.prompt,
        negative_prompt=args.negative or "",
        modality=args.modality,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        sampler=args.sampler,
        num_outputs=args.outputs,
        seed=args.seed,
    )
    output = engine.execute(job)
    from pathlib import Path

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (media, ext) in enumerate(zip(output.artifacts, output.extensions)):
        path = out_dir / f"{job.job_id}-out-{i}.{ext}"
        path.write_bytes(media)
        print(f"wrote {path} ({len(media)} bytes, seed={output.seed_used})")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluation.evaluator import T2IEvaluator

    evaluator = T2IEvaluator.from_folder(
        args.samples_dir, reference_dir=args.reference_dir
    )
    report = evaluator.evaluate_all(metrics=args.metrics)
    print(report)
    return 0


def _export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .agent.export import export_checkpoint

    out = export_checkpoint(
        model_dir=Path(args.model_dir), fmt=args.format, out_path=Path(args.out)
    )
    print(f"exported {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
