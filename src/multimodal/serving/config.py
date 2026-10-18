"""Serving-plane configuration, resolved from environment variables.

Every knob has a safe default so `python -m multimodal.cli serve` works
out of the box locally, and a matching entry in `.env.example` so
production values are documented and auditable.

Secrets policy: values are READ here, never written back to logs or
health payloads. See docs/architecture/failure-modes.md (secret leakage).
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..exceptions import ConfigError


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    return value if value not in ("", None) else default


class ServingConfig(BaseModel):
    """All runtime settings for the API + worker processes."""

    # ── Process ───────────────────────────────────────────────────────────
    env: Literal["development", "staging", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=4, ge=1, le=64)

    # ── Queue ─────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "mm:generation:jobs"
    queue_max_depth: int = Field(default=10_000, ge=1)
    job_ttl_s: int = Field(default=86_400, ge=60)
    job_recovery_max_age_s: int = Field(default=600, ge=30)

    # ── Generation limits (defence in depth with request schemas) ─────────
    max_steps: int = Field(default=100, ge=1)
    max_batch_size: int = Field(default=8, ge=1)
    max_video_frames: int = Field(default=64, ge=1)
    max_prompt_chars: int = Field(default=2000, ge=64)
    sync_timeout_s: float = Field(default=30.0, gt=0)

    # ── Models ────────────────────────────────────────────────────────────
    model_dir: str = "/var/lib/multimodal/models"
    t2i_model: str = "t2i-dit-s2"
    t2v_model: str = "t2v-dit-b2"
    default_sampler: str = "dpm++"
    default_steps: int = Field(default=30, ge=1)
    default_guidance: float = Field(default=7.5, ge=0)

    # ── Artifacts ─────────────────────────────────────────────────────────
    artifact_store: Literal["local", "s3"] = "local"
    artifact_dir: str = "/var/lib/multimodal/artifacts"
    s3_bucket: str = "multimodal-artifacts"
    s3_region: str = "us-east-1"
    s3_endpoint: str | None = None
    presign_ttl_s: int = Field(default=3600, ge=60)

    # ── Safety & auth ─────────────────────────────────────────────────────
    moderation_enabled: bool = True
    moderation_mode: Literal["blocklist", "classifier", "both"] = "blocklist"
    auth_enabled: bool = True
    api_keys: list[str] = Field(default_factory=list)

    # ── Observability ─────────────────────────────────────────────────────
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"
    service_name: str = "multimodal-api"

    @field_validator("t2i_model", "t2v_model")
    @classmethod
    def _registry_key(cls, value: str) -> str:
        if not value or any(c in value for c in "/\\"):
            raise ConfigError("model key must be a registry slug", details={"value": value})
        return value

    @classmethod
    def from_env(cls) -> "ServingConfig":
        """Build the config from MM_* environment variables."""
        keys = [k.strip() for k in (_env("MM_API_KEYS", "") or "").split(",") if k.strip()]
        if not keys:
            dev_key = _env("MM_API_KEY", "")
            keys = [dev_key] if dev_key else []
        return cls(
            env=(_env("MM_ENV", "development") or "development"),
            host=(_env("MM_API_HOST", "0.0.0.0") or "0.0.0.0"),
            port=int(_env("MM_API_PORT", "8000") or 8000),
            api_workers=int(_env("MM_API_WORKERS", "4") or 4),
            redis_url=(_env("MM_REDIS_URL", "redis://localhost:6379/0") or ""),
            queue_name=(_env("MM_QUEUE_NAME", "mm:generation:jobs") or ""),
            queue_max_depth=int(_env("MM_QUEUE_MAX_DEPTH", "10000") or 10_000),
            job_ttl_s=int(_env("MM_JOB_TTL_S", "86400") or 86_400),
            max_steps=int(_env("MM_MAX_STEPS", "100") or 100),
            max_batch_size=int(_env("MM_MAX_BATCH_SIZE", "8") or 8),
            max_prompt_chars=int(_env("MM_MAX_PROMPT_CHARS", "2000") or 2000),
            sync_timeout_s=float(_env("MM_SYNC_TIMEOUT_S", "30") or 30),
            model_dir=(_env("MM_MODEL_DIR", "/var/lib/multimodal/models") or ""),
            t2i_model=(_env("MM_T2I_MODEL", "t2i-dit-s2") or "t2i-dit-s2"),
            t2v_model=(_env("MM_T2V_MODEL", "t2v-dit-b2") or "t2v-dit-b2"),
            default_sampler=(_env("MM_DEFAULT_SAMPLER", "dpm++") or "dpm++"),
            artifact_store=(_env("MM_ARTIFACT_STORE", "local") or "local"),  # type: ignore[arg-type]
            artifact_dir=(_env("MM_ARTIFACT_DIR", "/var/lib/multimodal/artifacts") or ""),
            s3_bucket=(_env("MM_S3_BUCKET", "multimodal-artifacts") or ""),
            s3_region=(_env("MM_S3_REGION", "us-east-1") or ""),
            s3_endpoint=_env("MM_S3_ENDPOINT"),
            moderation_enabled=(_env("MM_MODERATION_ENABLED", "true") or "true").lower() == "true",
            moderation_mode=(_env("MM_MODERATION_MODE", "blocklist") or "blocklist"),  # type: ignore[arg-type]
            auth_enabled=(_env("MM_AUTH_ENABLED", "true") or "true").lower() == "true",
            api_keys=keys,
            metrics_enabled=(_env("MM_METRICS_ENABLED", "true") or "true").lower() == "true",
            tracing_enabled=(_env("MM_TRACING_ENABLED", "false") or "false").lower() == "true",
            service_name=(_env("MM_SERVICE_NAME", "multimodal-api") or ""),
        )
