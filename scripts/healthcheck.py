#!/usr/bin/env python3
"""Container/deploy healthcheck: probe /healthz with a hard deadline.

Exit codes (Kubernetes convention): 0 healthy, 1 probe failed, 2 timeout.
Deliberately does NOT check /readyz — liveness and readiness answer
different questions (see docs/architecture/system-architecture.md).
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

HOST = os.getenv("MM_HEALTH_HOST", "127.0.0.1")
PORT = os.getenv("MM_API_PORT", "8000")
TIMEOUT_S = float(os.getenv("MM_HEALTH_TIMEOUT", "3"))


def main() -> int:
    url = f"http://{HOST}:{PORT}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
            body = response.read().decode("utf-8", "replace")
            if response.status == 200:
                print(f"healthy: {body.strip()[:120]}")
                return 0
            print(f"unhealthy status={response.status}", file=sys.stderr)
            return 1
    except urllib.error.HTTPError as exc:
        print(f"unhealthy: HTTP {exc.code}", file=sys.stderr)
        return 1
    except Exception as exc:  # timeout, connection refused
        print(f"probe failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
