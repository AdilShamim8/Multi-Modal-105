"""API key authentication middleware.

Design points:
* Constant-time comparison — no key-content timing side channel.
* Keys are injected via config (secret store in production); the raw key
  is never logged. Only a short fingerprint (`key_hint`) is attached to
  jobs for abuse attribution.
* Health/metrics endpoints stay unauthenticated so load balancers and
  Prometheus can probe them without credentials.
"""

from __future__ import annotations

import hashlib
import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..exceptions import AuthenticationError

#: Paths that never require auth (liveness, readiness, metrics, docs).
#: Prefix match — /metrics/ (redirect target) must stay public too.
PUBLIC_PATHS = ("/healthz", "/readyz", "/metrics", "/docs", "/redoc", "/openapi.json")


def _is_public(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in PUBLIC_PATHS)


def key_fingerprint(api_key: str) -> str:
    """Stable, non-reversible hint for logs and job attribution."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:8]


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Rejects requests without a valid `X-API-Key` header."""

    def __init__(self, app, api_keys: list[str], enabled: bool = True) -> None:  # noqa: ANN001
        super().__init__(app)
        self.enabled = enabled and bool(api_keys)
        self._keys = [k.encode() for k in api_keys]

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        if not self.enabled or _is_public(request.url.path):
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "").encode()
        authorized = any(
            hmac.compare_digest(provided, expected) for expected in self._keys
        )
        if not authorized:
            error = AuthenticationError("missing or invalid API key")
            return JSONResponse(status_code=error.http_status, content=error.to_dict())
        request.state.key_hint = key_fingerprint(request.headers.get("X-API-Key", ""))
        return await call_next(request)
