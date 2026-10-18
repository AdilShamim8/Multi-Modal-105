"""Shared test doubles: in-memory Redis fake and deterministic fake engine.

Import as `from tests.fakes import FakeRedis, FakeEngine` (repo-root
conftest puts both `src` and the repo root on sys.path).
"""

from __future__ import annotations

import io

from multimodal.serving.config import ServingConfig
from multimodal.serving.engine import GenerationOutput, InferenceEngine


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


class FakeEngine(InferenceEngine):
    """Deterministic engine: tiny PNG per output, ~0 s 'work'."""

    def __init__(self, cfg: ServingConfig) -> None:
        super().__init__(cfg)
        self._loaded["t2i"] = True
        self.executed: list[str] = []

    def ready(self) -> bool:
        return True

    def execute(self, job):  # noqa: ANN001
        from PIL import Image

        self.executed.append(job.job_id)
        img = Image.new("RGB", (64, 64), (200, 80, 40))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return GenerationOutput(
            artifacts=[buf.getvalue()] * job.num_outputs,
            extensions=["png"] * job.num_outputs,
            seed_used=job.seed if job.seed is not None else 42,
            duration_s=0.01,
        )
