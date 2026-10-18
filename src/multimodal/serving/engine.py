"""Inference engine: loads generation pipelines and executes jobs.

Responsibilities (and nothing else — routing belongs to the API, persistence
to the queue):
* Lazy, timed model loading from the model registry directory.
* Warmup after load (first-token latency is a user-visible SLO).
* OOM-aware execution with automatic batch degradation and bounded retry.
* Deterministic per-job seeding so results are reproducible from the job id.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

from ..exceptions import ModelNotFoundError, OutOfMemoryError
from ..logging import get_logger
from ..utils.devices import free_device_memory_gb, resolve_device
from ..utils.seeds import derive_seed
from . import metrics
from .config import ServingConfig
from .jobs import GenerationJob

log = get_logger(__name__)


@dataclass
class GenerationOutput:
    artifacts: list[bytes]  # encoded media (PNG bytes / MP4 bytes)
    extensions: list[str]  # matching artifact extensions
    seed_used: int
    duration_s: float


class InferenceEngine:
    """Executes generation jobs against loaded T2I/T2V pipelines.

    The engine is deliberately pipeline-agnostic: `self.t2i` / `self.t2v`
    are opaque callables accepting a dict of job parameters and returning
    a list of PIL images (T2I) or video frames (T2V). Tests inject fakes;
    production injects `multimodal.agent.pipeline.T2IPipeline`.
    """

    def __init__(self, config: ServingConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device or resolve_device()
        self.t2i: Any | None = None
        self.t2v: Any | None = None
        self._loaded = {"t2i": False, "t2v": False}

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def load(self, modality: str = "image") -> None:
        """Load (once) the pipeline for a modality. Timed for /metrics."""
        key = self.config.t2i_model if modality == "image" else self.config.t2v_model
        started = time.perf_counter()
        path = f"{self.config.model_dir}/{key}"
        if modality == "image" and not self._loaded["t2i"]:
            self.t2i = self._load_t2i(path)
            self._loaded["t2i"] = True
        elif modality == "video" and not self._loaded["t2v"]:
            self.t2v = self._load_t2v(path)
            self._loaded["t2v"] = True
        metrics.set_model_load_seconds(time.perf_counter() - started)
        log.info("pipeline loaded", extra={"modality": modality, "key": key})

    def _load_t2i(self, path: str) -> Any:
        if not __import__("os").path.isdir(path):
            raise ModelNotFoundError(
                "T2I model directory missing",
                details={"path": path, "hint": "run scripts/preload_models.py or set MM_MODEL_DIR"},
            )
        from ..agent.pipeline import T2IPipeline  # heavy import, deferred

        return T2IPipeline.from_pretrained(path, device=self.device)

    def _load_t2v(self, path: str) -> Any:
        if not __import__("os").path.isdir(path):
            raise ModelNotFoundError("T2V model directory missing", details={"path": path})
        from ..agent.pipeline import T2VPipeline

        return T2VPipeline.from_pretrained(path, device=self.device)

    def warmup(self) -> None:
        """One cheap generation per loaded modality to trigger CUDA graphs,
        allocator warm pools, and lazy kernel compilation."""
        for modality, pipeline in (("image", self.t2i), ("video", self.t2v)):
            if pipeline is None:
                continue
            started = time.perf_counter()
            try:
                self._run(
                    pipeline,
                    modality,
                    {
                        "prompt": "warmup",
                        "width": 64,
                        "height": 64,
                        "num_steps": 2,
                        "num_outputs": 1,
                        "guidance_scale": 1.0,
                        "sampler": self.config.default_sampler,
                        "num_frames": 4,
                        "seed": 0,
                    },
                )
                log.info(
                    "warmup complete",
                    extra={
                        "modality": modality,
                        "seconds": round(time.perf_counter() - started, 2),
                    },
                )
            except Exception as exc:  # warmup failures are warnings, not fatal
                log.warning("warmup failed", extra={"modality": modality, "error": str(exc)})

    def ready(self) -> bool:
        return self._loaded["t2i"] or self._loaded["t2v"]

    # ── Execution ──────────────────────────────────────────────────────────
    def execute(self, job: GenerationJob) -> GenerationOutput:
        """Run one job with deterministic seeding and OOM degradation."""
        if job.modality == "image":
            if self.t2i is None:
                self.load("image")
            pipeline = self.t2i
        else:
            if self.t2v is None:
                self.load("video")
            pipeline = self.t2v

        seed = job.seed if job.seed is not None else derive_seed(42, job.job_id)
        params = {
            "prompt": job.prompt,
            "negative_prompt": job.negative_prompt or None,
            "width": job.width,
            "height": job.height,
            "num_steps": min(job.num_steps, self.config.max_steps),
            "guidance_scale": job.guidance_scale,
            "sampler": job.sampler,
            "num_outputs": job.num_outputs,
            "seed": seed,
            "num_frames": job.num_frames,
        }
        started = time.perf_counter()
        outputs = self._execute_with_oom_retry(pipeline, job.modality, params)
        duration = time.perf_counter() - started
        metrics.record_generation(job.modality, "succeeded", duration)
        return GenerationOutput(
            artifacts=outputs,
            extensions=["png"] * len(outputs) if job.modality == "image" else ["mp4"],
            seed_used=seed,
            duration_s=round(duration, 3),
        )

    def _execute_with_oom_retry(
        self, pipeline: Any, modality: str, params: dict[str, Any]
    ) -> list[bytes]:
        last_error: Exception | None = None
        degraded = dict(params)
        for _ in range(2):
            try:
                return self._run(pipeline, modality, degraded)
            except MemoryError as exc:  # torch raises RuntimeError for OOM; both handled
                last_error = exc
                self._degrade(degraded)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                last_error = exc
                self._degrade(degraded)
        raise OutOfMemoryError(
            "generation exhausted GPU memory after degradation",
            details={"free_gb": free_device_memory_gb(), "last_error": str(last_error)},
        )

    @staticmethod
    def _degrade(params: dict[str, Any]) -> None:
        """Halve the batch and cut steps — quality drops before availability does."""
        params["num_outputs"] = 1
        params["num_steps"] = max(4, params.get("num_steps", 30) // 2)

    @staticmethod
    def _run(pipeline: Any, modality: str, params: dict[str, Any]) -> list[bytes]:
        result = (
            pipeline(**params) if not hasattr(pipeline, "generate") else pipeline.generate(**params)
        )
        media = result.get("images", []) if isinstance(result, dict) else result
        encoded: list[bytes] = []
        for item in media:
            if hasattr(item, "save"):  # PIL image
                buf = io.BytesIO()
                item.save(buf, format="PNG")
                encoded.append(buf.getvalue())
            elif isinstance(item, bytes | bytearray):  # pre-encoded video
                encoded.append(bytes(item))
        if not encoded:
            from ..exceptions import GenerationError

            raise GenerationError("pipeline produced no outputs", details={"modality": modality})
        return encoded
