"""Request correlation and access logging middleware.

Every request gets a correlation id (inbound `X-Request-ID` honoured, else
generated) which propagates into structured logs AND into job records —
so a support ticket with a request id can be traced end-to-end through
API → queue → worker → artifact store.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..logging import get_logger, new_correlation_id
from .metrics import record_request

log = get_logger(__name__)

_SLOW_REQUEST_S = 2.0


class RequestLifecycleMiddleware(BaseHTTPMiddleware):
    """Assign correlation ids, time requests, log completion, count metrics."""

    def __init__(self, app, *, metrics_enabled: bool = True) -> None:
        super().__init__(app)
        self.metrics_enabled = metrics_enabled

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get("X-Request-ID") or new_correlation_id()
        request.state.correlation_id = correlation_id
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = correlation_id
            return response
        finally:
            elapsed = time.perf_counter() - started
            if self.metrics_enabled:
                route = request.scope.get("route")
                route_name = getattr(route, "path", request.url.path)
                record_request(route_name, request.method, status_code, elapsed)
            level = "warning" if elapsed > _SLOW_REQUEST_S or status_code >= 500 else "info"
            getattr(log, level)(
                "request completed",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_status": status_code,
                    "duration_s": round(elapsed, 4),
                    "key_hint": getattr(request.state, "key_hint", "-"),
                },
            )
