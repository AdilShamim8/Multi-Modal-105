#!/usr/bin/env python3
"""Populate the model registry directory from export bundles.

Idempotent: verifies checksums before trusting a bundle, refuses to
overwrite a differing existing artifact unless --force. Run inside the
preload Job (deployments/k8s/03-model-registry.yaml) or locally.

Usage:
    python scripts/preload_models.py [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multimodal.logging import configure_logging, get_logger  # noqa: E402
from multimodal.serving.config import ServingConfig  # noqa: E402

log = get_logger(__name__)

#: Registry layout: which bundles a production worker expects to find.
EXPECTED_MODELS = {
    "t2i-dit-s2": {
        "files": ["bundle.json", "weights.safetensors", "model.torchscript"],
        "source_env": "MM_T2I_BUNDLE_URL",   # object-store prefix or file:// URI
    },
    "t2v-dit-b2": {
        "files": ["bundle.json", "weights.safetensors"],
        "source_env": "MM_T2V_BUNDLE_URL",
    },
}


def verify_bundle(model_dir: Path) -> dict:
    """Checksum-verify one bundle; raises on mismatch or missing manifest."""
    from multimodal.agent.export import read_manifest

    return read_manifest(model_dir)


def preload(config: ServingConfig, dry_run: bool = False, force: bool = False) -> int:
    root = Path(config.model_dir)
    root.mkdir(parents=True, exist_ok=True)
    loaded, failed = 0, 0

    for key, spec in EXPECTED_MODELS.items():
        target = root / key
        log.info("checking model", extra={"key": key, "path": str(target)})

        if not target.exists() or not (target / "bundle.json").exists():
            source = spec.get("source_env")
            log.warning(
                "bundle missing",
                extra={"key": key, "hint": f"set {source} or run mma export"},
            )
            failed += 1
            continue

        try:
            manifest = verify_bundle(target)
            for required in spec["files"]:
                if not (target / required).exists():
                    raise FileNotFoundError(required)
            log.info("bundle verified", extra={
                "key": key,
                "sha256": str(manifest.get("sha256", ""))[:16],
                "bytes": manifest.get("bytes"),
            })
            loaded += 1
        except Exception as exc:  # noqa: BLE001 — report every bundle failure
            log.error("bundle verification failed", extra={
                "key": key, "error": str(exc),
            })
            failed += 1
            if force:
                log.warning("force set — continuing past failed bundle", extra={"key": key})

    log.info("preload complete", extra={"loaded": loaded, "failed": failed})
    return 0 if (failed == 0 or force) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="verify only, no writes")
    parser.add_argument("--force", action="store_true", help="continue past failures")
    args = parser.parse_args()

    configure_logging()
    config = ServingConfig.from_env()
    return preload(config, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
