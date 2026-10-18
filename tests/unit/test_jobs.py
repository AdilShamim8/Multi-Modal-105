"""Job state machine: transitions, serialization round-trips, timings."""

from __future__ import annotations

import pytest

from multimodal.exceptions import GenerationError
from multimodal.serving.jobs import GenerationJob, JobStatus
from multimodal.serving.schemas import new_job_id


def make_job(**kwargs) -> GenerationJob:
    defaults = {"job_id": new_job_id(), "prompt": "a test prompt"}
    defaults.update(kwargs)
    return GenerationJob(**defaults)


class TestTransitions:
    def test_happy_path(self):
        job = make_job()
        job.transition(JobStatus.RUNNING)
        job.transition(JobStatus.SUCCEEDED)
        assert job.status is JobStatus.SUCCEEDED

    def test_fail_then_retry(self):
        job = make_job()
        job.transition(JobStatus.RUNNING)
        job.transition(JobStatus.FAILED)
        job.transition(JobStatus.QUEUED)
        job.transition(JobStatus.RUNNING)
        job.transition(JobStatus.SUCCEEDED)
        assert job.attempts == 2

    def test_invalid_skip_queued(self):
        job = make_job()
        with pytest.raises(GenerationError, match="invalid job state transition"):
            job.transition(JobStatus.SUCCEEDED)

    def test_invalid_terminal_resurrection(self):
        job = make_job()
        job.transition(JobStatus.RUNNING)
        job.transition(JobStatus.SUCCEEDED)
        with pytest.raises(GenerationError):
            job.transition(JobStatus.RUNNING)

    def test_cancel_from_queued(self):
        job = make_job()
        job.transition(JobStatus.CANCELLED)
        assert job.status is JobStatus.CANCELLED

    def test_running_sets_timestamps_and_attempts(self):
        job = make_job()
        before = job.attempts
        job.transition(JobStatus.RUNNING)
        assert job.started_at is not None
        assert job.attempts == before + 1


class TestSerialization:
    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"seed": 12345, "negative_prompt": "no blur"},
            {"modality": "video", "num_frames": 16},
            {"artifact_keys": ["job/out-0-abc.png"], "error": {"code": "x"}},
        ],
    )
    def test_redis_roundtrip(self, overrides):
        job = make_job(**overrides)
        restored = GenerationJob.from_redis(job.to_redis())
        assert restored.job_id == job.job_id
        assert restored.status is job.status
        assert restored.seed == job.seed
        assert restored.artifact_keys == job.artifact_keys
        assert restored.error == job.error

    def test_timings(self):
        job = make_job()
        assert job.queue_wait_s() is None and job.duration_s() is None
        job.started_at = job.created_at + 1.5
        job.finished_at = job.started_at + 4.0
        assert job.queue_wait_s() == 1.5
        assert job.duration_s() == 4.0
