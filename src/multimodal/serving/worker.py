"""Generation worker: consumes the queue, executes, persists artifacts.

Operational contract:
* `run()` blocks forever; SIGTERM/SIGINT triggers a graceful drain bounded
  by WORKER_GRACE_PERIOD_S (in-flight jobs finish, queued jobs stay).
* Startup calls `recover_stuck()` — jobs orphaned by a previous crash are
  requeued, which is what makes rolling deploys safe.
* Every state change is written back to the job hash before side effects,
  so the API's view is never ahead of reality.
"""

from __future__ import annotations

import signal
import threading
import time
from typing import Optional

from ..exceptions import GenerationError, MultimodalError, OutOfMemoryError
from ..logging import get_logger, set_correlation_id
from . import metrics
from .artifacts import ArtifactStore
from .batching import BatchScheduler
from .config import ServingConfig
from .engine import InferenceEngine
from .jobs import GenerationJob, JobStatus
from .queue import JobQueue

log = get_logger(__name__)


class GenerationWorker:
    """Single-process queue consumer (scale horizontally via KEDA)."""

    def __init__(
        self,
        queue: JobQueue,
        engine: InferenceEngine,
        store: ArtifactStore,
        config: ServingConfig,
        max_batch: Optional[int] = None,
        max_wait_s: float = 0.25,
    ) -> None:
        self.queue = queue
        self.engine = engine
        self.store = store
        self.config = config
        self.scheduler = BatchScheduler(
            max_batch=max_batch or config.max_batch_size,
            max_wait_s=max_wait_s,
        )
        self._stop = threading.Event()
        self._active = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def run(self) -> None:
        self._install_signal_handlers()
        recovered = self.queue.recover_stuck()
        if recovered:
            log.info("startup recovery complete", extra={"recovered_jobs": recovered})
        self.engine.load("image")
        self.engine.warmup()

        log.info("worker ready", extra={
            "queue": self.config.queue_name,
            "device": self.engine.device,
        })
        while not self._stop.is_set():
            self._poll_once()
        self._drain()
        log.info("worker stopped cleanly")

    def stop(self) -> None:
        self._stop.set()

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, lambda *_: self.stop())
            except ValueError:  # not on main thread (tests)
                pass

    # ── Main loop ──────────────────────────────────────────────────────────
    def _poll_once(self) -> None:
        job = self.queue.dequeue(timeout_s=2)
        if job is not None:
            set_correlation_id(job.correlation_id)
            ready = self.scheduler.add(job)
            if ready is not None:
                self._execute_batch(ready)
        for batch in self.scheduler.flush_expired():
            self._execute_batch(batch)
        metrics.set_queue_depth(self.queue.depth())
        metrics.set_active_jobs(self._active + self.scheduler.pending())

    def _drain(self) -> None:
        deadline = time.monotonic() + 30.0
        for batch in self.scheduler.flush_all():
            self._execute_batch(batch)
        while self._active > 0 and time.monotonic() < deadline:
            time.sleep(0.1)

    # ── Execution ──────────────────────────────────────────────────────────
    def _execute_batch(self, batch) -> None:  # noqa: ANN001 - Batch
        for job in batch.jobs:
            self._begin(job)
        try:
            for job in batch.jobs:
                self._process(job)
        finally:
            for job in batch.jobs:
                self.queue.ack(job)

    def _begin(self, job: GenerationJob) -> None:
        self._active += 1
        job.transition(JobStatus.RUNNING)
        self.queue.update(job)
        log.info("job started", extra={
            "job_id": job.job_id, "modality": job.modality,
            "attempts": job.attempts, "signature": str(self.scheduler.stats()),
        })

    def _process(self, job: GenerationJob) -> None:
        try:
            output = self.engine.execute(job)
            keys = [
                self.store.save(job.job_id, f"out-{i}", media, ext)
                for i, (media, ext) in enumerate(zip(output.artifacts, output.extensions, strict=True))
            ]
            job.artifact_keys = keys
            job.seed = output.seed_used
            job.transition(JobStatus.SUCCEEDED)
            self.queue.update(job)
            log.info("job succeeded", extra={
                "job_id": job.job_id, "duration_s": output.duration_s,
                "artifacts": len(keys),
            })
        except OutOfMemoryError as exc:
            self._fail_or_retry(job, exc)
        except MultimodalError as exc:
            self._fail_or_retry(job, exc)
        except Exception as exc:  # noqa: BLE001 - worker must never die on one job
            self._fail_or_retry(job, GenerationError(
                f"unhandled worker error: {exc}", details={"type": type(exc).__name__}
            ))
        finally:
            self._active -= 1

    def _fail_or_retry(self, job: GenerationJob, exc: MultimodalError) -> None:
        job.error = exc.to_dict()
        metrics.record_generation(job.modality, "failed", 0.0)
        if job.attempts < 2:
            log.warning("job failed — requeueing for retry", extra={
                "job_id": job.job_id, "attempt": job.attempts, "error": exc.message,
            })
            # State machine: record the failure, THEN requeue (FAILED → QUEUED).
            job.transition(JobStatus.FAILED)
            self.queue.update(job)
            self.queue.requeue(job)
        else:
            job.transition(JobStatus.FAILED)
            self.queue.update(job)
            log.error("job failed permanently", extra={
                "job_id": job.job_id, "attempts": job.attempts, "error": exc.message,
            })
