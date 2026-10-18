"""Serving plane: API, queue, workers, safety, and observability.

Import cost policy: this package imports FastAPI/Redis at module scope of
its submodules, so it is only imported by serving entrypoints — never by
`multimodal` core, training code, or the CLI help path.
"""

from .config import ServingConfig
from .jobs import GenerationJob, JobStatus
from .queue import JobQueue, connect_redis
from .schemas import GenerationRequest

__all__ = [
    "ServingConfig",
    "GenerationJob",
    "JobStatus",
    "GenerationRequest",
    "JobQueue",
    "connect_redis",
    "create_app",
]


def __getattr__(name: str):  # noqa: ANN001 - PEP 562
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
