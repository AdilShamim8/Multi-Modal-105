"""Job queue: reliable-queue semantics, admission control, crash recovery.

Uses an in-memory fake implementing the RedisLike protocol — the same
interface production injects, proving the queue depends only on its contract.
"""

from __future__ import annotations

import pytest

from multimodal.exceptions import JobNotFoundError, QueueFullError
from multimodal.serving.config import ServingConfig
from multimodal.serving.jobs import GenerationJob, JobStatus
from multimodal.serving.queue import JobQueue
from multimodal.serving.schemas import new_job_id


class FakeRedis:
    """Name-aware in-memory redis fake: every list/hash is keyed by its
    Redis key, exactly like the real server (event journals must never
    leak into the job queue)."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.hashes: dict[str, dict] = {}
        self.expirations: dict[str, int] = {}

    def _list(self, name: str) -> list[str]:
        return self.lists.setdefault(name, [])

    # List operations — LPUSH to head, BRPOPLPUSH pops tail → dst (FIFO).
    def lpush(self, name: str, *values: str) -> int:
        lst = self._list(name)
        for v in values:
            lst.insert(0, v)
        return len(lst)

    def llen(self, name: str) -> int:
        return len(self._list(name))

    def brpoplpush(self, src: str, dst: str, timeout: int):
        src_list, dst_list = self._list(src), self._list(dst)
        if not src_list:
            return None
        value = src_list.pop()           # tail — FIFO with LPUSH
        dst_list.append(value)
        return value

    def lrem(self, name: str, count: int, value: str) -> int:
        lst = self._list(name)
        removed = 0
        while value in lst:
            lst.remove(value)
            removed += 1
        return removed

    def lrange(self, name: str, start: int, end: int) -> list[str]:
        return list(self._list(name))

    # Hash operations
    def hset(self, name: str, mapping: dict) -> int:
        self.hashes[name] = mapping
        return 1

    def hgetall(self, name: str) -> dict:
        return self.hashes.get(name, {})

    def expire(self, name: str, seconds: int) -> bool:
        self.expirations[name] = seconds
        return True

    def ping(self) -> bool:
        return True

    def delete(self, *names: str) -> int:
        return 0


@pytest.fixture()
def queue():
    config = ServingConfig(queue_max_depth=3, job_ttl_s=3600)
    return JobQueue(FakeRedis(), config)


def job(**kwargs) -> GenerationJob:
    defaults = dict(job_id=new_job_id(), prompt="p")
    defaults.update(kwargs)
    return GenerationJob(**defaults)


class TestEnqueueDequeue:
    def test_fifo_order(self, queue):
        first, second = job(), job()
        queue.enqueue(first)
        queue.enqueue(second)
        assert queue.dequeue(timeout_s=0).job_id == first.job_id
        assert queue.dequeue(timeout_s=0).job_id == second.job_id

    def test_dequeue_moves_to_processing(self, queue):
        j = job()
        queue.enqueue(j)
        dequeued = queue.dequeue(timeout_s=0)
        assert dequeued.job_id == j.job_id
        assert queue.processing_depth() == 1
        assert queue.depth() == 0

    def test_ack_removes_from_processing(self, queue):
        j = job()
        queue.enqueue(j)
        dequeued = queue.dequeue(timeout_s=0)
        queue.ack(dequeued)
        assert queue.processing_depth() == 0

    def test_get_unknown_raises(self, queue):
        with pytest.raises(JobNotFoundError):
            queue.get("nope")

    def test_update_persists_state(self, queue):
        j = job()
        queue.enqueue(j)
        j.transition(JobStatus.RUNNING)
        queue.update(j)
        restored = queue.get(j.job_id)
        assert restored.status is JobStatus.RUNNING


class TestAdmissionControl:
    def test_sheds_at_capacity(self, queue):
        for _ in range(3):
            queue.enqueue(job())
        with pytest.raises(QueueFullError):
            queue.enqueue(job())

    def test_depth_tracking(self, queue):
        assert queue.depth() == 0
        queue.enqueue(job())
        assert queue.depth() == 1


class TestCrashRecovery:
    def test_stuck_running_job_is_requeued(self, queue):
        j = job()
        queue.enqueue(j)
        dequeued = queue.dequeue(timeout_s=0)
        dequeued.transition(JobStatus.RUNNING)
        dequeued.started_at = 0.0  # ancient — beyond recovery window
        queue.update(dequeued)

        recovered = queue.recover_stuck(max_age_s=10)
        assert recovered == 1
        assert queue.depth() == 1
        restored = queue.get(j.job_id)
        assert restored.status is JobStatus.QUEUED

    def test_fresh_running_job_is_left_alone(self, queue):
        import time

        j = job()
        queue.enqueue(j)
        dequeued = queue.dequeue(timeout_s=0)
        dequeued.transition(JobStatus.RUNNING)
        dequeued.started_at = time.time()  # just started
        queue.update(dequeued)

        assert queue.recover_stuck(max_age_s=600) == 0
        assert queue.depth() == 0

    def test_corrupt_hash_is_dropped(self, queue):
        queue.redis.queue.append("ghost-job")  # no hash behind it
        assert queue.recover_stuck(max_age_s=0) == 0


class TestHealth:
    def test_healthy_with_responsive_redis(self, queue):
        assert queue.healthy() is True
