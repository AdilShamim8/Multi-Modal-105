"""Artifact storage: local filesystem and S3-compatible backends.

Contract: `save()` writes bytes and returns a storage key; `resolve()`
returns a client-fetchable URL (file path locally, presigned URL on S3).
The worker never serves media itself — clients fetch from storage/CDN, so
API capacity scales independently of media traffic (critical at the
100M-user tier; see docs/architecture/scaling-100m-users.md).
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Protocol

from ..exceptions import ArtifactError
from ..logging import get_logger

log = get_logger(__name__)


class ArtifactStore(Protocol):
    def save(self, job_id: str, kind: str, data: bytes, ext: str) -> str: ...
    def resolve(self, key: str) -> str: ...
    def healthy(self) -> bool: ...


class LocalArtifactStore:
    """Filesystem store — development and single-node deployments."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactError(f"cannot create artifact dir {root}", details={"error": str(exc)}) from exc

    def save(self, job_id: str, kind: str, data: bytes, ext: str = "png") -> str:
        digest = hashlib.sha256(data).hexdigest()[:16]
        key = f"{job_id}/{kind}-{digest}.{ext}"
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)          # atomic-ish: write then rename
        tmp.replace(path)
        log.info("artifact saved", extra={"key": key, "bytes": len(data)})
        return key

    def resolve(self, key: str) -> str:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ArtifactError("path traversal blocked", details={"key": key})
        return str(path)

    def healthy(self) -> bool:
        return os.access(self.root, os.W_OK)


class S3ArtifactStore:
    """S3-compatible store (AWS S3, MinIO, R2) with presigned resolution."""

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        endpoint: str | None = None,
        presign_ttl_s: int = 3600,
        prefix: str = "artifacts",
    ) -> None:
        try:
            import boto3  # lazy: only serving installs carry boto3
        except ImportError as exc:  # pragma: no cover
            raise ArtifactError("boto3 required for S3 artifact store") from exc
        self.bucket = bucket
        self.prefix = prefix
        self.presign_ttl_s = presign_ttl_s
        self.client = boto3.client("s3", region_name=region, endpoint_url=endpoint)

    def save(self, job_id: str, kind: str, data: bytes, ext: str = "png") -> str:
        digest = hashlib.sha256(data).hexdigest()[:16]
        key = f"{self.prefix}/{job_id}/{kind}-{digest}.{ext}"
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=f"image/{'png' if ext == 'png' else 'jpeg'}",
                ChecksumSHA256=hashlib.sha256(data).hexdigest(),
            )
        except Exception as exc:  # boto3 raises many; all are store faults
            raise ArtifactError("S3 put_object failed", details={"error": str(exc)}) from exc
        log.info("artifact uploaded", extra={"key": key, "bytes": len(data)})
        return key

    def resolve(self, key: str) -> str:
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=self.presign_ttl_s,
            )
        except Exception as exc:
            raise ArtifactError("presign failed", details={"error": str(exc)}) from exc

    def healthy(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False


def build_artifact_store(config) -> ArtifactStore:  # noqa: ANN001 - ServingConfig
    """Factory keyed on config.artifact_store."""
    if config.artifact_store == "s3":
        return S3ArtifactStore(
            bucket=config.s3_bucket,
            region=config.s3_region,
            endpoint=config.s3_endpoint,
            presign_ttl_s=config.presign_ttl_s,
        )
    return LocalArtifactStore(root=config.artifact_dir)
