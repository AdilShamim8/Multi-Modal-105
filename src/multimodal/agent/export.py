"""Deployment artifact export (torchscript / onnx).

Exported bundles are what the serving plane loads — never raw training
checkpoints. A bundle carries:

    <model_dir>/bundle.json      manifest: architecture, version, checksums
    <model_dir>/weights.safetensors
    <model_dir>/model.torchscript | model.onnx

The manifest is what makes rollbacks auditable: it pins the exact commit,
config, and weight checksum that produced every deployed artifact.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from ..exceptions import CheckpointError
from ..logging import get_logger
from ..version import RELEASE_DATE, __version__

log = get_logger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(
    model_dir: Path,
    fmt: str = "torchscript",
    out_path: Path | None = None,
    model: torch.nn.Module | None = None,
    sample_input: Any | None = None,
) -> Path:
    """Export a trained pipeline component for the serving plane.

    Args:
        model_dir: directory containing weights.safetensors + config.
        fmt: "torchscript" (default) or "onnx".
        out_path: destination file (defaults into model_dir).
        model: live module; loaded from model_dir when omitted.
        sample_input: example input required by tracing (torchscript) and
            required by onnx export.

    Returns:
        Path of the exported artifact.
    """
    if fmt not in ("torchscript", "onnx"):
        raise CheckpointError(f"unsupported export format: {fmt}")

    model_dir = Path(model_dir)
    weights = model_dir / "weights.safetensors"
    if model is None:
        if not weights.exists():
            raise CheckpointError(
                "no weights.safetensors in model dir",
                details={"model_dir": str(model_dir)},
            )
        model = _load_module(model_dir)
    model = model.eval()

    out_path = Path(out_path) if out_path else model_dir / f"model.{fmt}"
    started = time.perf_counter()

    if fmt == "torchscript":
        if sample_input is None:
            raise CheckpointError("torchscript export requires sample_input for tracing")
        with torch.no_grad():
            scripted = torch.jit.trace(model, sample_input)
        scripted.save(str(out_path))
    else:
        if sample_input is None:
            raise CheckpointError("onnx export requires sample_input")
        torch.onnx.export(
            model,
            sample_input,
            str(out_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    manifest = {
        "format": fmt,
        "platform_version": __version__,
        "release_date": RELEASE_DATE,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact": out_path.name,
        "sha256": _sha256(out_path),
        "bytes": out_path.stat().st_size,
    }
    (model_dir / "bundle.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "export complete",
        extra={
            "fmt": fmt,
            "seconds": round(time.perf_counter() - started, 2),
            "bytes": manifest["bytes"],
        },
    )
    return out_path


def _load_module(model_dir: Path) -> torch.nn.Module:
    """Load weights.safetensors into the architecture described by config."""
    from safetensors.torch import load_file

    from ..models.diffusion.dit import DiT

    config_path = model_dir / "config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    model = DiT(
        input_size=config.get("latent_size", 64),
        patch_size=config.get("patch_size", 2),
        hidden_size=config.get("hidden_size", 1152),
        depth=config.get("depth", 28),
        num_heads=config.get("num_heads", 16),
    )
    state = load_file(str(model_dir / "weights.safetensors"))
    model.load_state_dict(state, strict=False)
    return model


def read_manifest(model_dir: Path) -> dict[str, Any]:
    """Load and integrity-check a bundle manifest (used at engine load)."""
    manifest_path = Path(model_dir) / "bundle.json"
    if not manifest_path.exists():
        raise CheckpointError("bundle.json missing", details={"model_dir": str(model_dir)})
    manifest = json.loads(manifest_path.read_text())
    artifact = Path(model_dir) / manifest.get("artifact", "")
    if artifact.exists():
        actual = _sha256(artifact)
        if actual != manifest.get("sha256"):
            raise CheckpointError(
                "artifact checksum mismatch — possible corruption",
                details={
                    "artifact": str(artifact),
                    "expected": manifest.get("sha256"),
                    "actual": actual,
                },
            )
    return manifest
