"""Token-bucket rate limiting per API key.

Why token bucket (not fixed window): bursts are legitimate for generation
clients (open a page → fire 4 requests), while sustained abuse is still
capped. Buckets live in-process by default; a Redis-backed bucket keeps
limits correct across API replicas (config flag `distributed`).

Budget math (see docs/operations/capacity-planning.md): a T2I job costs
GPU-seconds, so the limit protects the *queue*, not just the API process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..exceptions import RateLimitError


@dataclass
class Bucket:
    tokens: float
    capacity: float
    refill_per_s: float
    updated: float = field(default_factory=time.monotonic)

    def allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_per_s)
        self.updated = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def retry_after_s(self, cost: float = 1.0) -> float:
        deficit = cost - self.tokens
        return round(max(0.0, deficit / self.refill_per_s), 3)


class RateLimiter:
    """In-process token-bucket registry keyed by client identity."""

    def __init__(self, capacity: float = 20.0, refill_per_s: float = 5.0) -> None:
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self._buckets: dict[str, Bucket] = {}

    def _bucket(self, key: str) -> Bucket:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = Bucket(tokens=self.capacity, capacity=self.capacity,
                            refill_per_s=self.refill_per_s)
            self._buckets[key] = bucket
        return bucket

    def check(self, identity: str, cost: float = 1.0) -> None:
        """Raise RateLimitError (with Retry-After info) when over budget."""
        bucket = self._bucket(identity)
        if not bucket.allow(cost):
            raise RateLimitError(
                "rate limit exceeded for API key",
                details={
                    "retry_after_s": bucket.retry_after_s(cost),
                    "capacity": self.capacity,
                    "refill_per_s": self.refill_per_s,
                },
            )

    def snapshot(self) -> dict[str, float]:
        """Live token counts — exported via /metrics debug endpoint."""
        return {k: round(b.tokens, 2) for k, b in self._buckets.items()}


def rate_limit_response_headers(error: RateLimitError) -> dict[str, str]:
    """RFC-correct response headers for 429s."""
    retry_after = error.details.get("retry_after_s", 1.0)
    return {"Retry-After": str(max(1, int(retry_after)))}
