"""Generation submission and job status routes.

The submission path is deliberately thin: validate → moderate → admit →
enqueue. All GPU work happens in workers. `sync=true` short-circuits the
queue ONLY within a bounded timeout, for low-latency interactive use —
operators can disable it (MM_SYNC_TIMEOUT_S=0) to force queue-only flow.
"""

from __future__ import annotations

import time

from typing import Union

from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import JSONResponse

from ...exceptions import JobNotFoundError, MultimodalError, RateLimitError
from ...logging import get_logger
from .. import metrics
from ..jobs import GenerationJob, JobStatus
from ..ratelimit import rate_limit_response_headers
from ..schemas import (
    ArtifactOut,
    GenerationRequest,
    JobEnvelope,
    JobStatusOut,
    SyncGenerationOut,
    new_job_id,
)

log = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["generation"])


@router.post(
    "/generations",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=Union[JobEnvelope, SyncGenerationOut],
    summary="Submit a generation job",
    responses={200: {"model": SyncGenerationOut, "description": "Synchronous result"}},
)
async def submit_generation(
    body: GenerationRequest,
    request: Request,
    response: Response,
) -> JobEnvelope | JSONResponse:
    app = request.app

    # 1. Rate limit — per key, protects the queue, not just the process.
    identity = getattr(request.state, "key_hint", "anonymous")
    try:
        app.state.rate_limiter.check(identity)
    except RateLimitError as exc:
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_dict(),
            headers=rate_limit_response_headers(exc),
        )

    # 2. Moderation — before any GPU capacity is committed.
    app.state.moderation.enforce(body.prompt)

    # 3. Build the job record.
    job = GenerationJob(
        job_id=new_job_id(),
        prompt=body.prompt,
        negative_prompt=body.negative_prompt or "",
        modality=body.modality.value,
        width=body.width,
        height=body.height,
        num_frames=body.num_frames,
        num_steps=body.num_steps,
        guidance_scale=body.guidance_scale,
        sampler=body.sampler.value,
        num_outputs=body.num_outputs,
        seed=body.seed,
        api_key_hint=identity,
    )

    # 4. Sync path (bounded) or async admission into the queue.
    if body.sync and app.state.config.sync_timeout_s > 0:
        response.status_code = status.HTTP_200_OK
        return _sync_generate(app, job)

    app.state.queue.enqueue(job)
    base = str(request.base_url).rstrip("/")
    return JobEnvelope(
        job_id=job.job_id,
        status=job.status.value,
        created_at=job.created_at,
        accepted_at=time.time(),
        links={"status": f"{base}/v1/generations/{job.job_id}"},
    )


def _sync_generate(app, job: GenerationJob):  # noqa: ANN001, ANN202
    """Execute inline under the sync timeout; fall back to the queue on overrun."""
    import threading

    from ..engine import GenerationOutput

    result: dict[str, object] = {}
    done = threading.Event()

    def run() -> None:
        try:
            job.transition(JobStatus.RUNNING)
            output: GenerationOutput = app.state.engine.execute(job)
            keys = [
                app.state.store.save(job.job_id, f"out-{i}", media, ext)
                for i, (media, ext) in enumerate(zip(output.artifacts, output.extensions))
            ]
            job.artifact_keys = keys
            job.seed = output.seed_used
            job.transition(JobStatus.SUCCEEDED)
            result["output"] = output
        except MultimodalError as exc:
            job.error = exc.to_dict()
            job.transition(JobStatus.FAILED)
            result["error"] = exc
        except Exception as exc:  # noqa: BLE001
            job.error = {"code": "internal_error", "message": str(exc)}
            job.transition(JobStatus.FAILED)
            result["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    if not done.wait(timeout=app.state.config.sync_timeout_s):
        # Timed out: hand the job to the queue instead of failing it.
        job.status = JobStatus.QUEUED
        app.state.queue.enqueue(job)
        base = "/v1/generations"
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job.job_id, "status": "queued",
                "created_at": job.created_at, "accepted_at": time.time(),
                "detail": "sync timeout exceeded — job queued",
                "links": {"status": f"{base}/{job.job_id}"},
            },
        )
    if "error" in result:
        exc = result["error"]
        code = getattr(exc, "http_status", 500)
        return JSONResponse(
            status_code=code, content=getattr(exc, "to_dict", lambda: str(exc))()
        )
    output: GenerationOutput = result["output"]  # type: ignore[assignment]
    return SyncGenerationOut(
        job_id=job.job_id,
        status=job.status.value,
        artifacts=[
            ArtifactOut(kind="image", url=app.state.store.resolve(key))
            for key in job.artifact_keys
        ],
        seed_used=output.seed_used,
        duration_s=output.duration_s,
    )


@router.get(
    "/generations/{job_id}",
    response_model=JobStatusOut,
    summary="Get job status and artifacts",
)
async def get_generation(job_id: str, request: Request) -> JobStatusOut:
    job = request.app.state.queue.get(job_id)  # raises JobNotFoundError
    return JobStatusOut(
        job_id=job.job_id,
        status=job.status.value,
        modality=job.modality,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        duration_s=job.duration_s(),
        queue_wait_s=job.queue_wait_s(),
        artifacts=[
            ArtifactOut(
                kind="image" if job.modality == "image" else "video",
                url=request.app.state.store.resolve(key),
            )
            for key in job.artifact_keys
        ],
        error=job.error,
        attempts=job.attempts,
    )


@router.delete(
    "/generations/{job_id}",
    summary="Cancel a queued job",
)
async def cancel_generation(job_id: str, request: Request) -> JSONResponse:
    app = request.app
    try:
        job = app.state.queue.get(job_id)
    except JobNotFoundError as exc:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())
    if job.status is JobStatus.QUEUED:
        job.transition(JobStatus.CANCELLED)
        app.state.queue.update(job)
        metrics.record_generation(job.modality, "cancelled", 0.0)
        return JSONResponse(content={"job_id": job.job_id, "status": job.status.value})
    return JSONResponse(
        status_code=409,
        content={"code": "not_cancellable",
                 "message": f"job is {job.status.value}; only queued jobs can be cancelled",
                 "details": {}},
    )


@router.get("/queue/stats", summary="Queue telemetry", include_in_schema=False)
async def queue_stats(request: Request, depth_limit: int = Query(default=1, ge=0, le=100)) -> dict:
    metrics.set_queue_depth(request.app.state.queue.depth())
    return {
        "depth": request.app.state.queue.depth(),
        "processing": request.app.state.queue.processing_depth(),
        "rate_limiter": request.app.state.rate_limiter.snapshot(),
    }
