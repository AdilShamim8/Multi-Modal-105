"""Worker loop integration with fakes: full job lifecycle, retry, drain."""

from __future__ import annotations

import pytest

from multimodal.exceptions import GenerationError
from multimodal.serving.artifacts import LocalArtifactStore
from multimodal.serving.config import ServingConfig
from multimodal.serving.jobs import GenerationJob, JobStatus
from multimodal.serving.queue import JobQueue
from multimodal.serving.schemas import new_job_id
from multimodal.serving.worker import GenerationWorker
from multimodal.utils.devices import resolve_device

from tests.fakes import FakeEngine, FakeRedis


class FailingEngine(FakeEngine):
    """Fails every execution with a platform error (retry path)."""

    def execute(self, job):  # noqa: ANN001
        self.executed.append(job.job_id)
        raise GenerationError("synthetic failure for retry test")


@pytest.fixture()
def config(tmp_path):
    return ServingConfig(
        artifact_dir=str(tmp_path / "artifacts"),
        job_recovery_max_age_s=60,
        max_batch_size=4,
    )


@pytest.fixture()
def queue(config):
    return JobQueue(FakeRedis(), config)


def job(**kwargs):
    defaults = dict(job_id=new_job_id(), prompt="worker test")
    defaults.update(kwargs)
    return GenerationJob(**defaults)


class TestPollOnce:
    def test_job_completes_end_to_end(self, config, queue, tmp_path):
        engine = FakeEngine(config)
        worker = GenerationWorker(
            queue, engine, LocalArtifactStore(str(tmp_path / "a")), config,
            max_wait_s=0.0,   # tests flush immediately — no latency budget
        )
        j = job()
        queue.enqueue(j)

        worker._poll_once()

        final = queue.get(j.job_id)
        assert final.status is JobStatus.SUCCEEDED
        assert final.artifact_keys, "artifact must be persisted"
        assert queue.depth() == 0 and queue.processing_depth() == 0
        assert engine.executed == [j.job_id]

    def test_failed_job_requeues_for_retry(self, config, queue, tmp_path):
        engine = FailingEngine(config)
        worker = GenerationWorker(
            queue, engine, LocalArtifactStore(str(tmp_path / "a")), config,
            max_wait_s=0.0,   # tests flush immediately — no latency budget
        )
        j = job()
        queue.enqueue(j)

        worker._poll_once()

        assert queue.depth() == 1, "failed job should return to the queue"
        restored = queue.get(j.job_id)
        assert restored.status is JobStatus.QUEUED
        assert restored.attempts == 1
        assert restored.error["code"] == "generation_error"

    def test_permanent_failure_after_second_attempt(self, config, queue, tmp_path):
        engine = FailingEngine(config)
        worker = GenerationWorker(
            queue, engine, LocalArtifactStore(str(tmp_path / "a")), config,
            max_wait_s=0.0,
        )
        j = job()
        queue.enqueue(j)

        worker._poll_once()   # attempt 1 → FAILED → requeued (bounded retry)
        worker._poll_once()   # attempt 2 → attempts=2 → permanent FAILED

        final = queue.get(j.job_id)
        assert final.status is JobStatus.FAILED
        assert final.attempts == 2
        assert queue.depth() == 0, "permanently failed jobs do not requeue"


class TestDrain:
    def test_flush_all_processes_pending_jobs(self, config, queue, tmp_path):
        engine = FakeEngine(config)
        worker = GenerationWorker(
            queue, engine, LocalArtifactStore(str(tmp_path / "a")), config,
            max_wait_s=0.0,   # tests flush immediately — no latency budget
        )
        for i in range(3):
            queue.enqueue(job(num_steps=30))   # same signature → one batch

        for _ in range(3):
            worker._poll_once()

        worker._drain()
        assert queue.depth() == 0
        assert len(engine.executed) == 3

    def test_recovery_requeues_orphans_at_startup(self, config, queue, tmp_path):
        engine = FakeEngine(config)
        worker = GenerationWorker(
            queue, engine, LocalArtifactStore(str(tmp_path / "a")), config,
            max_wait_s=0.0,   # tests flush immediately — no latency budget
        )
        # Simulate a crashed predecessor: job stuck in processing, RUNNING.
        j = job()
        queue.enqueue(j)
        dequeued = queue.dequeue(timeout_s=0)
        dequeued.transition(JobStatus.RUNNING)
        dequeued.started_at = 0.0
        queue.update(dequeued)

        recovered = queue.recover_stuck()
        assert recovered == 1
        assert queue.depth() == 1


class TestEngineDevice:
    def test_engine_resolves_device(self, config):
        engine = FakeEngine(config)
        assert engine.device == resolve_device()
