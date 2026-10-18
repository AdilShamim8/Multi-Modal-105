"""Exception hierarchy for the Multi-Modal Generation Platform.

Every error raised intentionally by this codebase derives from
`MultimodalError`, so operators can catch platform faults without
swallowing programming errors (TypeError, KeyError, ...).

Serving-plane exceptions carry HTTP status codes so route handlers can map
them mechanically instead of hand-rolling responses.
"""

from __future__ import annotations

from typing import Any


class MultimodalError(Exception):
    """Base class for all platform errors."""

    #: HTTP status used when this error escapes to the API layer.
    http_status: int = 500
    #: Stable machine-readable code for clients and alerting.
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation used by the API error envelope."""
        return {"code": self.code, "message": self.message, "details": self.details}


# ── Configuration & environment ─────────────────────────────────────────────
class ConfigError(MultimodalError):
    """Invalid, missing, or mutually inconsistent configuration."""

    http_status = 500
    code = "config_error"


class EnvironmentError(MultimodalError):
    """Required environment variable or secret is absent/unusable."""

    http_status = 500
    code = "environment_error"


# ── Model & inference plane ─────────────────────────────────────────────────
class ModelNotFoundError(MultimodalError):
    """Requested model key/revision absent from the registry or disk."""

    http_status = 404
    code = "model_not_found"


class CheckpointError(MultimodalError):
    """Checkpoint corrupt, version-mismatched, or failed integrity check."""

    http_status = 500
    code = "checkpoint_error"


class DeviceError(MultimodalError):
    """Requested accelerator unavailable or unsuitable for the workload."""

    http_status = 503
    code = "device_error"


class OutOfMemoryError(MultimodalError):
    """GPU OOM detected; caller should retry with a smaller batch/steps."""

    http_status = 503
    code = "out_of_memory"


class GenerationError(MultimodalError):
    """A generation step failed after retries (sampler, VAE decode, ...)."""

    http_status = 500
    code = "generation_error"


# ── Serving plane ───────────────────────────────────────────────────────────
class SafetyError(MultimodalError):
    """Prompt or output rejected by the moderation gate."""

    http_status = 400
    code = "content_rejected"


class AuthenticationError(MultimodalError):
    """Missing/invalid API key."""

    http_status = 401
    code = "unauthenticated"


class RateLimitError(MultimodalError):
    """Client exceeded its token-bucket allowance."""

    http_status = 429
    code = "rate_limited"


class QueueFullError(MultimodalError):
    """Admission control rejected the job (backpressure / load shedding)."""

    http_status = 503
    code = "queue_full"


class JobNotFoundError(MultimodalError):
    """Unknown or expired job id."""

    http_status = 404
    code = "job_not_found"


class ArtifactError(MultimodalError):
    """Artifact store unavailable or write failed."""

    http_status = 502
    code = "artifact_error"
