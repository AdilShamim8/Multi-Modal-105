"""Generation job model and state machine.

States:
    QUEUED ──▶ RUNNING ──▶ SUCCEEDED
       │           │
       │           └──▶ FAILED ──▶ (retry ≤ max) ──▶ QUEUED
       └────────────────▶ CANCELLED

Every transition is validated and journaled to `mm:job:{id}:events`, giving
operators an audit trail when debugging "why did job X die at 03:00".
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ..exceptions import GenerationError


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_VALID = {
    (None, JobStatus.QUEUED),
    (JobStatus.QUEUED, JobStatus.RUNNING),
    (JobStatus.QUEUED, JobStatus.CANCELLED),
    (JobStatus.RUNNING, JobStatus.SUCCEEDED),
    (JobStatus.RUNNING, JobStatus.FAILED),
    (JobStatus.RUNNING, JobStatus.CANCELLED),
    (JobStatus.FAILED, JobStatus.QUEUED),        # bounded retry
    (JobStatus.CANCELLED, JobStatus.QUEUED),      # operator requeue
}

TERMINAL = {JobStatus.SUCCEEDED, JobStatus.CANCELLED}


@dataclass
class GenerationJob:
    """Serializable job record stored as a Redis hash."""

    job_id: str
    prompt: str
    negative_prompt: str = ""
    modality: str = "image"            # "image" | "video"
    width: int = 512
    height: int = 512
    num_frames: int = 16
    num_steps: int = 30
    guidance_scale: float = 7.5
    sampler: str = "dpm++"
    num_outputs: int = 1
    seed: int | None = None
    api_key_hint: str = "-"            # key fingerprint (never the raw key)
    status: JobStatus = JobStatus.QUEUED
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    artifact_keys: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # ── State machine ─────────────────────────────────────────────────────
    def transition(self, new_status: JobStatus) -> None:
        current: JobStatus | None = self.status
        if (current, new_status) not in _VALID:
            raise GenerationError(
                "invalid job state transition",
                details={"job_id": self.job_id, "from": current, "to": new_status},
            )
        self.status = new_status
        if new_status is JobStatus.RUNNING:
            self.started_at = time.time()
            self.attempts += 1
        if new_status in TERMINAL or new_status is JobStatus.FAILED:
            self.finished_at = time.time()

    # ── Serialization ─────────────────────────────────────────────────────
    def to_redis(self) -> dict[str, str]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return {k: json.dumps(v) if isinstance(v, dict | list) else str(v)
                for k, v in payload.items() if v is not None}

    @classmethod
    def from_redis(cls, data: dict[str, str]) -> GenerationJob:
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in ("artifact_keys", "error"):
                kwargs[key] = json.loads(value) if value else None
            elif key == "status":
                kwargs[key] = JobStatus(value)
            elif key in ("seed", "started_at", "finished_at"):
                kwargs[key] = None if value in ("", "None") else float(
                    value if "." in value or "e" in value else int(value)
                )
                if key == "seed" and kwargs[key] is not None:
                    kwargs[key] = int(kwargs[key])
            elif key in ("created_at",):
                kwargs[key] = float(value)
            elif key in ("width", "height", "num_frames", "num_steps", "num_outputs", "attempts"):
                kwargs[key] = int(value)
            elif key in ("guidance_scale",):
                kwargs[key] = float(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def queue_wait_s(self) -> float | None:
        if self.started_at is None:
            return None
        return round(self.started_at - self.created_at, 3)

    def duration_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return round(self.finished_at - self.started_at, 3)
