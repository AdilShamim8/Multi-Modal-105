"""FastAPI application factory.

Composition root for the serving plane: builds config, queue, engine,
artifact store, moderation gate, and middleware stack — all injectable for
tests. Production entrypoint: `python -m multimodal.cli serve`.

Failure policy: the app FAILS TO START (rather than starts degraded) when
required dependencies are misconfigured — a half-serving replica behind a
load balancer is worse than a visible deployment error.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..exceptions import JobNotFoundError, MultimodalError
from ..logging import configure_logging, get_logger
from ..version import __version__
from .artifacts import build_artifact_store
from .auth import APIKeyMiddleware
from .config import ServingConfig
from .engine import InferenceEngine
from .middleware import RequestLifecycleMiddleware
from .moderation import ModerationGate
from .queue import JobQueue, connect_redis
from .ratelimit import RateLimiter
from .routes import generation, health

log = get_logger(__name__)


def create_app(
    config: ServingConfig | None = None,
    engine: InferenceEngine | None = None,
    queue: JobQueue | None = None,
    rate_limiter: RateLimiter | None = None,
    moderation: ModerationGate | None = None,
    *,
    load_models_at_startup: bool = True,
) -> FastAPI:
    """Build the API application.

    Args:
        config: resolved settings (defaults to `ServingConfig.from_env()`).
        engine / queue / rate_limiter / moderation: injectable overrides —
            tests pass fakes; production leaves them None for real wiring.
        load_models_at_startup: warm the engine in the lifespan (False for
            fast test startup and for worker-only containers).
    """
    cfg = config or ServingConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        app.state.config = cfg
        app.state.env = cfg.env
        app.state.rate_limiter = rate_limiter or RateLimiter(
            capacity=20.0, refill_per_s=5.0
        )
        app.state.moderation = moderation or ModerationGate(
            enabled=cfg.moderation_enabled,
            mode=cfg.moderation_mode,
            max_prompt_chars=cfg.max_prompt_chars,
        )
        if queue is not None:
            app.state.queue = queue
        else:
            app.state.queue = JobQueue(connect_redis(cfg.redis_url), cfg)
        if engine is not None:
            app.state.engine = engine
        else:
            app.state.engine = InferenceEngine(cfg)
        app.state.store = build_artifact_store(cfg)

        if load_models_at_startup:
            try:
                app.state.engine.load("image")
                app.state.engine.warmup()
            except MultimodalError as exc:
                log.error("startup model load failed", extra={"error": exc.message})
                if cfg.env == "production":
                    raise
        log.info("api ready", extra={
            "version": __version__, "env": cfg.env, "port": cfg.port,
        })
        yield
        log.info("api shutting down")

    app = FastAPI(
        title="Multi-Modal Generation API",
        version=__version__,
        description=(
            "Text-to-image and text-to-video generation platform.\n\n"
            "Async flow: POST a job, poll its status, fetch artifacts from the "
            "resolved URLs. Sync flow available for interactive clients."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (order = execution order; auth first is intentional) ────
    app.add_middleware(RequestLifecycleMiddleware, metrics_enabled=cfg.metrics_enabled)
    app.add_middleware(APIKeyMiddleware, api_keys=cfg.api_keys, enabled=cfg.auth_enabled)

    # ── Routes ─────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(generation.router)

    if cfg.metrics_enabled:
        from .metrics import asgi_metrics_app

        app.mount("/metrics", asgi_metrics_app())

    # ── Error envelope: every MultimodalError becomes a stable JSON body ───
    @app.exception_handler(MultimodalError)
    async def platform_error_handler(request: Request, exc: MultimodalError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={
                **exc.to_dict(),
                "correlation_id": getattr(request.state, "correlation_id", None),
            },
        )

    @app.exception_handler(JobNotFoundError)
    async def not_found_handler(request: Request, exc: MultimodalError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    return app
