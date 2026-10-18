"""Continuous batching scheduler for the generation worker.

GPU utilisation at scale depends on coalescing small jobs into batches.
This scheduler accumulates dequeued jobs for a short window and flushes
when either:

* `max_batch` jobs share the same execution signature (modality + geometry
  + sampler + steps), or
* `max_wait_s` elapses since the first job arrived (latency ceiling).

Signature grouping matters: mixing a 1024px/50-step job with a 512px/10-step
job would pad both to the worst case and waste GPU-seconds.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .jobs import GenerationJob


def signature(job: GenerationJob) -> tuple:
    """Execution-relevant grouping key (ignores prompts and seeds)."""
    if job.modality == "video":
        return ("video", job.width, job.height, job.num_frames, job.num_steps, job.sampler)
    return ("image", job.width, job.height, job.num_steps, job.sampler)


@dataclass
class Batch:
    signature: tuple
    jobs: list[GenerationJob] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)

    def age_s(self) -> float:
        return time.monotonic() - self.created_at

    def full(self, max_batch: int) -> bool:
        return len(self.jobs) >= max_batch


class BatchScheduler:
    """Collects jobs into executable batches under a latency budget."""

    def __init__(self, max_batch: int = 8, max_wait_s: float = 0.25) -> None:
        self.max_batch = max(1, max_batch)
        self.max_wait_s = max_wait_s
        self._open: dict[tuple, Batch] = {}
        self._stats = {"batches": 0, "jobs": 0, "solo_flushes": 0}

    def add(self, job: GenerationJob) -> Optional[Batch]:
        """Add a job; returns a Batch when it should execute NOW."""
        key = signature(job)
        batch = self._open.get(key)
        if batch is None:
            batch = Batch(signature=key)
            self._open[key] = batch
        batch.jobs.append(job)
        self._stats["jobs"] += 1
        if batch.full(self.max_batch):
            del self._open[key]
            self._stats["batches"] += 1
            return batch
        return None

    def flush_expired(self) -> list[Batch]:
        """Return all batches past the wait budget (called in the poll loop)."""
        expired = [b for b in self._open.values() if b.age_s() >= self.max_wait_s]
        for batch in expired:
            del self._open[batch.signature]
        self._stats["batches"] += len(expired)
        if expired and all(len(b.jobs) == 1 for b in expired):
            self._stats["solo_flushes"] += len(expired)
        return expired

    def flush_all(self) -> list[Batch]:
        """Drain everything (shutdown path — no job left behind)."""
        batches = list(self._open.values())
        self._open.clear()
        self._stats["batches"] += len(batches)
        return batches

    def pending(self) -> int:
        return sum(len(b.jobs) for b in self._open.values())

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @staticmethod
    def merge_media(batch: Batch) -> list[GenerationJob]:
        """Jobs in a batch share execution; keep them addressable for fan-out."""
        return batch.jobs


def batch_efficiency(jobs: Iterable[GenerationJob], batches: int) -> float:
    """Average jobs per batch — the utilisation KPI for this component."""
    jobs_list = list(jobs)
    if not batches:
        return 0.0
    return round(len(jobs_list) / batches, 2)
