"""Timing helpers for benchmarks, latency budgets, and slow-path alerts."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Lap:
    label: str
    seconds: float


@dataclass
class Stopwatch:
    """Multi-lap stopwatch; `report()` renders a compact latency breakdown."""

    started_at: float = field(default_factory=time.perf_counter)
    laps: list[Lap] = field(default_factory=list)

    def lap(self, label: str) -> float:
        now = time.perf_counter()
        elapsed = now - self.started_at
        self.laps.append(Lap(label, elapsed))
        return elapsed

    def last_interval(self) -> float:
        if not self.laps:
            return time.perf_counter() - self.started_at
        return time.perf_counter() - (self.started_at + self.laps[-1].seconds)

    def report(self) -> str:
        parts = [f"{lap.label}={lap.seconds:.3f}s" for lap in self.laps]
        parts.append(f"total={time.perf_counter() - self.started_at:.3f}s")
        return " ".join(parts)


@contextmanager
def timer(label: str, *, warn_after_s: float | None = None) -> Iterator[Stopwatch]:
    """Time a block; logs the duration at exit (warning if over budget).

    Example:
        with timer("vae.decode", warn_after_s=0.5) as sw:
            frames = vae.decode_from_latent(z)
    """
    from ..logging import get_logger

    log = get_logger(__name__)
    sw = Stopwatch()
    try:
        yield sw
    finally:
        elapsed = time.perf_counter() - sw.started_at
        level = "warning" if warn_after_s is not None and elapsed > warn_after_s else "info"
        getattr(log, level)(
            "block timed",
            extra={"label": label, "seconds": round(elapsed, 4)},
        )
