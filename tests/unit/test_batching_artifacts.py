"""Batching scheduler and artifact store contracts."""

from __future__ import annotations

import pytest

from multimodal.exceptions import ArtifactError
from multimodal.serving.artifacts import LocalArtifactStore
from multimodal.serving.batching import BatchScheduler, batch_efficiency, signature
from multimodal.serving.jobs import GenerationJob
from multimodal.serving.schemas import new_job_id


def job(**kwargs) -> GenerationJob:
    defaults = {"job_id": new_job_id(), "prompt": "p"}
    defaults.update(kwargs)
    return GenerationJob(**defaults)


class TestSignatureGrouping:
    def test_same_geometry_shares_signature(self):
        a = job(width=512, height=512, num_steps=30)
        b = job(width=512, height=512, num_steps=30)
        assert signature(a) == signature(b)

    def test_different_steps_differ(self):
        a = job(num_steps=30)
        b = job(num_steps=50)
        assert signature(a) != signature(b)

    def test_video_signature_includes_frames(self):
        a = job(modality="video", num_frames=16)
        b = job(modality="video", num_frames=32)
        assert signature(a) != signature(b)

    def test_prompts_and_seeds_ignored(self):
        a = job(prompt="cat", seed=1)
        b = job(prompt="dog", seed=2)
        assert signature(a) == signature(b)


class TestBatchScheduler:
    def test_flushes_when_full(self):
        scheduler = BatchScheduler(max_batch=3, max_wait_s=60)
        batch = None
        for _ in range(3):
            batch = scheduler.add(job()) or batch
        assert batch is not None and len(batch.jobs) == 3

    def test_partial_batch_waits(self):
        scheduler = BatchScheduler(max_batch=4, max_wait_s=60)
        assert scheduler.add(job()) is None
        assert scheduler.pending() == 1

    def test_expired_batches_flush(self):
        scheduler = BatchScheduler(max_batch=8, max_wait_s=0.0)
        scheduler.add(job())
        flushed = scheduler.flush_expired()
        assert len(flushed) == 1 and len(flushed[0].jobs) == 1
        assert scheduler.pending() == 0

    def test_mixed_signatures_do_not_merge(self):
        scheduler = BatchScheduler(max_batch=2, max_wait_s=60)
        scheduler.add(job(num_steps=30))
        assert scheduler.add(job(num_steps=50)) is None  # different signature
        assert scheduler.pending() == 2

    def test_flush_all_drains(self):
        scheduler = BatchScheduler(max_batch=8, max_wait_s=60)
        scheduler.add(job())
        scheduler.add(job(num_steps=10))
        assert len(scheduler.flush_all()) == 2
        assert scheduler.pending() == 0

    def test_stats_track_batches_and_jobs(self):
        scheduler = BatchScheduler(max_batch=1, max_wait_s=60)
        for _ in range(2):
            scheduler.add(job())
        stats = scheduler.stats()
        assert stats["batches"] == 2 and stats["jobs"] == 2

    def test_efficiency_metric(self):
        jobs = [job() for _ in range(6)]
        assert batch_efficiency(jobs, batches=3) == 2.0
        assert batch_efficiency([], batches=0) == 0.0


class TestLocalArtifactStore:
    @pytest.fixture()
    def store(self, tmp_path):
        return LocalArtifactStore(root=str(tmp_path / "artifacts"))

    def test_save_returns_content_addressed_key(self, store):
        key = store.save("job-1", "image", b"payload-bytes", ext="png")
        assert key.startswith("job-1/")
        assert key.endswith(".png")

    def test_same_content_same_key(self, store):
        k1 = store.save("job-1", "image", b"same")
        k2 = store.save("job-2", "image", b"same")
        assert k1.split("-")[-1] == k2.split("-")[-1]  # identical digest

    def test_resolve_returns_readable_path(self, store):
        key = store.save("job-1", "image", b"data")
        path = store.resolve(key)
        from pathlib import Path

        assert Path(path).read_bytes() == b"data"

    def test_path_traversal_blocked(self, store):
        with pytest.raises(ArtifactError, match="path traversal blocked"):
            store.resolve("../../etc/passwd")

    def test_healthy_when_writable(self, store):
        assert store.healthy()

    def test_no_tmp_files_left_behind(self, store):
        store.save("job-1", "image", b"data")
        from pathlib import Path

        leftovers = list(Path(store.root).rglob("*.tmp"))
        assert leftovers == []
