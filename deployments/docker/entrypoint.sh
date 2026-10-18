#!/usr/bin/env bash
# Container entrypoint: api | worker | serve-dev
# Kept tiny on purpose — orchestration belongs in Kubernetes, not shell.
set -euo pipefail

ROLE="${1:-api}"

case "$ROLE" in
  api)
    exec python -m multimodal.cli serve \
        --host "${MM_API_HOST:-0.0.0.0}" \
        --port "${MM_API_PORT:-8000}" \
        --workers "${MM_API_WORKERS:-4}"
    ;;
  worker)
    exec python -m multimodal.cli worker \
        --modality "${MM_WORKER_MODALITY:-all}" \
        --max-batch "${MM_MAX_BATCH_SIZE:-8}"
    ;;
  serve-dev)
    # Development only: hot reload, single process, verbose logs.
    export MM_ENV=development MM_LOG_LEVEL=DEBUG
    exec python -m multimodal.cli serve --reload --host 0.0.0.0 --port "${MM_API_PORT:-8000}"
    ;;
  *)
    echo "unknown role: $ROLE (expected api | worker | serve-dev)" >&2
    exit 64
    ;;
esac
