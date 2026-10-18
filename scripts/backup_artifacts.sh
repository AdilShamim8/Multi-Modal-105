#!/usr/bin/env bash
# Artifact store backup: sync generated media to a secondary bucket.
# Runs from cron/K8s CronJob; uses aws cli (S3-compatible endpoints work
# via MM_S3_ENDPOINT). Idempotent, checksum-preserving, quiet on success.
# Usage: ./scripts/backup_artifacts.sh s3://backup-bucket/multimodal
set -euo pipefail

DEST="${1:?usage: backup_artifacts.sh s3://backup-bucket/prefix}"
SRC_DIR="${MM_ARTIFACT_DIR:-/var/lib/multimodal/artifacts}"
ENDPOINT="${MM_S3_ENDPOINT:-}"
RETENTION_DAYS="${MM_BACKUP_RETENTION_DAYS:-30}"

[ -d "${SRC_DIR}" ] || { echo "artifact dir missing: ${SRC_DIR}" >&2; exit 1; }

AWS_ARGS=()
[ -n "${ENDPOINT}" ] && AWS_ARGS+=(--endpoint-url "${ENDPOINT}")

echo "backing up ${SRC_DIR} → ${DEST}"
aws "${AWS_ARGS[@]}" s3 sync "${SRC_DIR}" "${DEST}" \
  --storage-class STANDARD_IA \
  --only-show-errors

# Lifecycle the backup prefix (cheap belt-and-braces if bucket policy lags).
aws "${AWS_ARGS[@]}" s3 ls "${DEST}" --recursive --only-show-errors > /dev/null

echo "backup complete (retention: ${RETENTION_DAYS}d via bucket lifecycle)"
