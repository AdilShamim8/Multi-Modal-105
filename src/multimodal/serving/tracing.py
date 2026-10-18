"""OpenTelemetry tracing setup — optional, zero-cost when disabled.

Contract:
* `setup_tracing(config)` is idempotent and safe to call when the
  opentelemetry packages are absent (API-only containers) — it becomes a
  no-op and `is_enabled()` reports False.
* The span processor batches; generation spans are exported by the worker,
  HTTP spans by the API process. Correlation ids from
  `multimodal.logging` propagate as span attributes so traces and logs
  join on one query.

Why tracing is behind a flag: at high RPS, span export is real egress and
CPU. Development and small deployments lose nothing by leaving it off;
production turns it on with a sampled exporter.
"""

from __future__ import annotations

from typing import Optional

from ..logging import get_logger
from .config import ServingConfig

log = get_logger(__name__)

_initialized = False
_enabled = False


def setup_tracing(config: ServingConfig) -> bool:
    """Configure OTLP export if enabled and available. Returns whether
    tracing is active in this process."""
    global _initialized, _enabled
    if _initialized:  # idempotent: one exporter per process
        return _enabled
    _initialized = True

    if not config.tracing_enabled:
        log.info("tracing disabled by config")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning("tracing enabled but opentelemetry packages missing — no-op")
        return False

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=config.otel_endpoint))
    )
    trace.set_tracer_provider(provider)
    _enabled = True
    log.info("tracing initialized", extra={
        "endpoint": config.otel_endpoint,
        "service": config.service_name,
    })
    return True


def is_enabled() -> bool:
    return _enabled


def get_tracer(name: str = "multimodal"):
    """Tracer handle; a no-op tracer when OTel is absent."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NullTracer()


class _NullTracer:
    """Duck-types the OTel tracer interface for disabled deployments."""

    def start_as_current_span(self, *args, **kwargs):  # noqa: ANN002, ANN003
        import contextlib

        return contextlib.nullcontext()


def record_job_span(job):  # noqa: ANN001 - GenerationJob (avoids import cycle)
    """Attach job metadata to the current span, if tracing is active."""
    if not _enabled:
        return
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("mm.job_id", job.job_id)
            span.set_attribute("mm.modality", job.modality)
            span.set_attribute("mm.steps", job.num_steps)
            span.set_attribute("mm.correlation_id", job.correlation_id)
    except Exception:  # noqa: BLE001 — telemetry must never break the job
        pass
