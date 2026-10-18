"""Redis-backed job queue with admission control and crash recovery.

Design (the pattern matters at scale — see docs/architecture/scaling-100m-users.md):

    API      LPUSH  mm:generation:jobs          (admission-checked)
    Worker   BRPOPLPUSH jobs → processing       (reliable queue semantics)
    Worker   LREM processing on completion
    Startup  recover_stuck() requeues jobs whose worker died mid-flight

Why not Celery/RabbitMQ: a single sorted structure with visibility timeouts
covers this workload's exactly-once-enough semantics with one dependency
already required for rate limits and caching. Revisit via ADR if fan-out
to multiple downstream consumers ever appears.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, cast

import redis

from ..exceptions import QueueFullError
from ..logging import get_logger
from .config import ServingConfig
from .jobs import GenerationJob, JobStatus

log = get_logger(__name__)


class RedisLike(Protocol):
    """Minimal Redis surface used here (satisfied by redis.Redis & fakes).

    Parameters and returns are deliberately ``Any``: redis-py's stubs type
    one method set across the sync *and* async clients (FieldT args,
    ResponseT returns), which is wider/narrower than the str-only fakes in
    different directions — no fixed signature satisfies both under Protocol
    variance. ``Any`` keeps the structural check meaningful (method names,
    arity, keyword acceptance) without fighting redis-py's stub variance;
    the queue's own logic treats values as str because it writes them.
    """

    def lpush(self, name: Any, *values: Any) -> Any: ...
    def brpoplpush(self, src: Any, dst: Any, timeout: Any = ...) -> Any: ...
    def lrem(self, name: Any, count: int, value: Any) -> Any: ...
    def llen(self, name: Any) -> Any: ...
    def lrange(self, name: Any, start: int, end: int) -> Any: ...
    def hset(self, name: Any, key: Any = ..., value: Any = ..., *, mapping: Any = ...) -> Any: ...
    def hgetall(self, name: Any) -> Any: ...
    def expire(self, name: Any, seconds: Any) -> Any: ...
    def ping(self) -> Any: ...
    def delete(self, *names: Any) -> Any: ...
    def rpoplpush(self, src: Any, dst: Any) -> Any: ...


class JobQueue:
    """Durable generation queue with backpressure and recovery."""

    def __init__(self, client: RedisLike, config: ServingConfig) -> None:
        self.redis = client
        self.config = config
        self.queue_key = config.queue_name
        self.processing_key = f"{config.queue_name}:processing"

    # ── Producer side (API) ───────────────────────────────────────────────
    def enqueue(self, job: GenerationJob) -> None:
        """Persist the job hash, then push its id; reject on backpressure."""
        depth = self.redis.llen(self.queue_key)
        if depth >= self.config.queue_max_depth:
            raise QueueFullError(
                "generation queue at capacity — shedding load",
                details={"depth": depth, "max": self.config.queue_max_depth},
            )
        self.redis.hset(self._job_key(job.job_id), mapping=job.to_redis())
        self.redis.expire(self._job_key(job.job_id), self.config.job_ttl_s)
        self.redis.lpush(self.queue_key, job.job_id)
        log.info(
            "job enqueued",
            extra={
                "job_id": job.job_id,
                "depth": depth + 1,
                "modality": job.modality,
            },
        )

    def get(self, job_id: str) -> GenerationJob:
        data = self.redis.hgetall(self._job_key(job_id))
        if not data:
            from ..exceptions import JobNotFoundError

            raise JobNotFoundError("unknown or expired job", details={"job_id": job_id})
        return GenerationJob.from_redis(data)

    def update(self, job: GenerationJob) -> None:
        self.redis.hset(self._job_key(job.job_id), mapping=job.to_redis())
        self.redis.expire(self._job_key(job.job_id), self.config.job_ttl_s)
        event = {"status": job.status.value, "at": time.time()}
        self.redis.lpush(f"{self._job_key(job.job_id)}:events", str(event).replace("'", '"'))

    def depth(self) -> int:
        return self.redis.llen(self.queue_key)

    def processing_depth(self) -> int:
        return self.redis.llen(self.processing_key)

    # ── Consumer side (worker) ────────────────────────────────────────────
    def dequeue(self, timeout_s: int = 5) -> GenerationJob | None:
        """Blocking pop into the processing list (reliable-queue pattern)."""
        job_id = self.redis.brpoplpush(self.queue_key, self.processing_key, timeout_s)
        if job_id is None:
            return None
        return self.get(job_id)

    def ack(self, job: GenerationJob) -> None:
        """Remove a finished job from the processing list."""
        self.redis.lrem(self.processing_key, 1, job.job_id)

    def requeue(self, job: GenerationJob) -> None:
        """Move a job from processing back to the queue (bounded retry)."""
        self.redis.lrem(self.processing_key, 1, job.job_id)
        job.transition(JobStatus.QUEUED)
        self.update(job)
        self.redis.lpush(self.queue_key, job.job_id)

    # ── Crash recovery ────────────────────────────────────────────────────
    def recover_stuck(self, max_age_s: int | None = None) -> int:
        """Requeue jobs stranded in `processing` by dead workers.

        Call at worker startup. A job is stuck if its hash shows RUNNING but
        its started_at is older than the recovery window.
        """
        max_age = max_age_s if max_age_s is not None else self.config.job_recovery_max_age_s
        recovered = 0
        for job_id in self.redis.lrange(self.processing_key, 0, -1):
            try:
                job = self.get(job_id)
            except Exception:
                self.redis.lrem(self.processing_key, 1, job_id)
                continue
            if job.status is not JobStatus.RUNNING:
                continue
            age = time.time() - (job.started_at or 0)
            if age > max_age:
                self.redis.lrem(self.processing_key, 1, job_id)
                job.status = JobStatus.QUEUED
                job.started_at = None
                self.update(job)
                self.redis.lpush(self.queue_key, job.job_id)
                recovered += 1
                log.warning("recovered stuck job", extra={"job_id": job_id, "age_s": round(age)})
        return recovered

    # ── Health ────────────────────────────────────────────────────────────
    def healthy(self) -> bool:
        try:
            return bool(self.redis.ping())
        except redis.RedisError:
            return False

    def _job_key(self, job_id: str) -> str:
        return f"mm:job:{job_id}"


def connect_redis(url: str) -> RedisLike:
    """Connection factory with sane production timeouts.

    Returns the protocol type (not redis.Redis) so call sites typecheck
    under both typing sources for redis: redis-py's inline annotations and
    the types-redis stubs, whose method signatures differ in parameter
    kinds. The cast documents that variance boundary in exactly one place.
    """
    client = redis.Redis.from_url(
        url,
        socket_connect_timeout=3,
        socket_timeout=5,
        health_check_interval=30,
        decode_responses=True,
    )
    return cast(RedisLike, client)
