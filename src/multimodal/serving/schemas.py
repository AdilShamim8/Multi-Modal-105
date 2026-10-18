"""API request/response schemas (OpenAPI contract).

These models are the single source of truth for the HTTP surface: FastAPI
validates requests against them and renders the OpenAPI docs from them.
Limits encoded here are mirrored by `ServingConfig` — the schema protects
the process, the config lets operators tune without a deploy.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Modality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class Sampler(str, Enum):
    DDPM = "ddpm"
    DDIM = "ddim"
    DPM_PLUS_PLUS = "dpm++"


class GenerationRequest(BaseModel):
    """A single generation job submission."""

    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt")
    modality: Modality = Modality.IMAGE
    negative_prompt: Optional[str] = Field(
        default=None, max_length=2000, description="What to steer away from (CFG)"
    )
    width: int = Field(default=512, ge=64, le=1024, multiple_of=8)
    height: int = Field(default=512, ge=64, le=1024, multiple_of=8)
    num_frames: int = Field(
        default=16, ge=1, le=64, description="Video only: frames per clip"
    )
    num_steps: int = Field(default=30, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=30.0)
    sampler: Sampler = Sampler.DPM_PLUS_PLUS
    num_outputs: int = Field(default=1, ge=1, le=4)
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)
    sync: bool = Field(
        default=False,
        description="Wait for the result (bounded by sync timeout) instead of polling",
    )

    @field_validator("prompt", "negative_prompt")
    @classmethod
    def _strip(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if isinstance(value, str) else value

    @field_validator("num_frames")
    @classmethod
    def _frames_power(cls, value: int) -> int:
        if value < 4 or value & (value - 1) != 0:
            raise ValueError("num_frames must be a power of two >= 4 (temporal VAE)")
        return value


class JobEnvelope(BaseModel):
    """202 Accepted response — the client polls or follows the status URL."""

    job_id: str
    status: str
    created_at: float
    accepted_at: float
    links: dict[str, str] = Field(default_factory=dict)


class ArtifactOut(BaseModel):
    kind: str
    url: str
    expires_at: Optional[float] = None
    bytes: Optional[int] = None
    checksum: Optional[str] = None


class JobStatusOut(BaseModel):
    job_id: str
    status: str
    modality: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_s: Optional[float] = None
    queue_wait_s: Optional[float] = None
    artifacts: list[ArtifactOut] = Field(default_factory=list)
    error: Optional[dict[str, Any]] = None
    attempts: int = 0


class SyncGenerationOut(BaseModel):
    """Direct result for `sync=true` submissions under the timeout budget."""

    job_id: str
    status: str
    artifacts: list[ArtifactOut]
    seed_used: Optional[int] = None
    duration_s: float


class ErrorOut(BaseModel):
    """Uniform error envelope — stable `code`, human `message`."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None


class HealthOut(BaseModel):
    status: str
    version: str
    env: str
    checks: dict[str, bool] = Field(default_factory=dict)


def new_job_id() -> str:
    """Time-sortable, collision-safe job id (ULP-style, no extra dep)."""
    return f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:12]}"
