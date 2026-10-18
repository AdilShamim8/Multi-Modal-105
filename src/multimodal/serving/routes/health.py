"""Health and readiness endpoints.

* /healthz — liveness: "is the process alive". Load balancers use this;
  it must never touch downstream dependencies.
* /readyz  — readiness: "can this replica serve traffic right now".
  Checks queue connectivity, artifact store writability, and engine load
  state. Failed readiness pulls the pod out of rotation without a restart.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...version import RELEASE_DATE, __version__
from ..schemas import HealthOut

router = APIRouter(tags=["system"])


@router.get("/healthz", response_model=HealthOut, summary="Liveness probe")
async def healthz(request: Request) -> HealthOut:
    return HealthOut(status="ok", version=__version__, env=request.app.state.env)


@router.get("/readyz", summary="Readiness probe")
async def readyz(request: Request) -> JSONResponse:
    checks = {}
    try:
        checks["queue"] = request.app.state.queue.healthy()
    except Exception:
        checks["queue"] = False
    try:
        checks["artifact_store"] = request.app.state.store.healthy()
    except Exception:
        checks["artifact_store"] = False
    checks["engine_loaded"] = bool(getattr(request.app.state.engine, "ready", lambda: False)())

    ready = all(checks.values()) if checks else False
    payload = {
        "status": "ready" if ready else "not_ready",
        "version": __version__,
        "release_date": RELEASE_DATE,
        "env": request.app.state.env,
        "checks": checks,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)
