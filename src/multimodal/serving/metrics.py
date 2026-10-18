"""Prometheus metrics for the serving plane.

Metric contract (alert rules in deployments/monitoring/alerts.yml depend on
these exact names):

    mm_api_requests_total{route,method,status}
    mm_api_request_latency_seconds{route}      (histogram)
    mm_generations_total{modality,status}
    mm_generation_latency_seconds{modality}    (histogram)
    mm_queue_depth                              (gauge)
    mm_active_jobs                              (gauge)
    mm_model_load_seconds                       (gauge)
"""

from __future__ import annotations

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
    )

    _HAS_PROM = True
except ImportError:  # pragma: no cover - allow torch-free metric stubs
    _HAS_PROM = False

if _HAS_PROM:
    _REQUESTS = Counter(
        "mm_api_requests_total",
        "HTTP requests processed",
        ["route", "method", "status"],
    )
    _LATENCY = Histogram(
        "mm_api_request_latency_seconds",
        "HTTP request latency in seconds",
        ["route"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    )
    _GENERATIONS = Counter(
        "mm_generations_total",
        "Generation jobs completed",
        ["modality", "status"],
    )
    _GEN_LATENCY = Histogram(
        "mm_generation_latency_seconds",
        "End-to-end generation latency in seconds",
        ["modality"],
        buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
    )
    _QUEUE_DEPTH = Gauge("mm_queue_depth", "Jobs waiting in the generation queue")
    _ACTIVE_JOBS = Gauge("mm_active_jobs", "Jobs currently executing")
    _MODEL_LOAD = Gauge("mm_model_load_seconds", "Wall time of last model load")


def record_request(route: str, method: str, status: int, elapsed: float) -> None:
    if _HAS_PROM:
        _REQUESTS.labels(route=route, method=method, status=str(status)).inc()
        _LATENCY.labels(route=route).observe(elapsed)


def record_generation(modality: str, status: str, elapsed: float) -> None:
    if _HAS_PROM:
        _GENERATIONS.labels(modality=modality, status=status).inc()
        _GEN_LATENCY.labels(modality=modality).observe(elapsed)


def set_queue_depth(depth: int) -> None:
    if _HAS_PROM:
        _QUEUE_DEPTH.set(depth)


def set_active_jobs(count: int) -> None:
    if _HAS_PROM:
        _ACTIVE_JOBS.set(count)


def set_model_load_seconds(seconds: float) -> None:
    if _HAS_PROM:
        _MODEL_LOAD.set(seconds)


def asgi_metrics_app():  # pragma: no cover - wiring lives in app.py
    """Prometheus ASGI app factory (mounted at /metrics)."""
    if not _HAS_PROM:
        from starlette.responses import PlainTextResponse

        async def _stub(_: object) -> PlainTextResponse:
            return PlainTextResponse("prometheus_client not installed\n", status_code=503)

        return _stub
    from prometheus_client import make_asgi_app

    return make_asgi_app()
