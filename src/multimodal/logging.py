"""Structured logging for training, serving, and workers.

Design goals:
1. **One line, one event.** JSON in production (log aggregators), pretty
   text on a developer terminal — selected by `MM_ENV`.
2. **Correlation.** Every record carries `correlation_id` so a single user
   request can be traced across API → queue → worker → artifact store.
3. **No import-time side effects.** Call `configure_logging()` from
   entrypoints; modules only call `get_logger(__name__)`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

_CORRELATION_ID: ContextVar[str] = ContextVar("correlation_id", default="-")

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created))}"
            f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": _CORRELATION_ID.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable format for development terminals."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{time.strftime('%H:%M:%S', time.gmtime(record.created))} "
            f"{record.levelname:<7} {record.name}:{record.lineno} "
            f"[{_CORRELATION_ID.get()[:8]}] {record.getMessage()}"
        )
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(level: str | None = None) -> None:
    """Idempotently configure the root logger from the environment."""
    env = os.getenv("MM_ENV", "development")
    level_name = (level or os.getenv("MM_LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if env == "production" else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level_name, logging.INFO))
    # Quiet the noisiest third-party loggers; platform code stays at level.
    for noisy in ("urllib3", "botocore", "httpx", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger; call `configure_logging` once per process."""
    return logging.getLogger(name)


def new_correlation_id() -> str:
    """Start a fresh correlation scope and return the id."""
    cid = uuid.uuid4().hex
    _CORRELATION_ID.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Adopt an externally supplied correlation id (e.g. from a queue message)."""
    _CORRELATION_ID.set(cid or "-")


def get_correlation_id() -> str:
    return _CORRELATION_ID.get()
