"""Export bundles: manifest round-trip, checksum verification, tamper detection."""

from __future__ import annotations

import json

import pytest
import torch

from multimodal.agent.export import export_checkpoint, read_manifest
from multimodal.exceptions import CheckpointError


class TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestManifest:
    def test_bundle_manifest_roundtrip(self, tmp_path):
        model = TinyModule()
        out = export_checkpoint(
            model_dir=tmp_path,
            model=model,
            sample_input=torch.randn(1, 4),
            fmt="torchscript",
        )
        assert out.exists()

        manifest = read_manifest(tmp_path)
        assert manifest["format"] == "torchscript"
        assert manifest["platform_version"] == "1.0.0"
        assert manifest["release_date"] == "2026-10-18"
        assert len(manifest["sha256"]) == 64
        assert manifest["bytes"] == out.stat().st_size

    def test_missing_manifest_rejected(self, tmp_path):
        with pytest.raises(CheckpointError, match="bundle.json missing"):
            read_manifest(tmp_path)

    def test_tampered_artifact_fails_checksum(self, tmp_path):
        model = TinyModule()
        out = export_checkpoint(
            model_dir=tmp_path,
            model=model,
            sample_input=torch.randn(1, 4),
            fmt="torchscript",
        )
        out.write_bytes(out.read_bytes() + b"tampered")  # corrupt it
        with pytest.raises(CheckpointError, match="checksum mismatch"):
            read_manifest(tmp_path)

    def test_unknown_format_rejected(self, tmp_path):
        with pytest.raises(CheckpointError, match="unsupported export format"):
            export_checkpoint(
                model_dir=tmp_path,
                model=TinyModule(),
                sample_input=torch.randn(1, 4),
                fmt="coreml",
            )

    def test_tracing_requires_sample_input(self, tmp_path):
        with pytest.raises(CheckpointError, match="sample_input"):
            export_checkpoint(
                model_dir=tmp_path,
                model=TinyModule(),
                sample_input=None,
                fmt="torchscript",
            )

    def test_scripted_module_loads_and_runs(self, tmp_path):
        model = TinyModule()
        export_checkpoint(
            model_dir=tmp_path,
            model=model,
            sample_input=torch.randn(1, 4),
            fmt="torchscript",
        )
        loaded = torch.jit.load(str(tmp_path / "model.torchscript"))
        with torch.no_grad():
            out = loaded(torch.randn(2, 4))
        assert out.shape == (2, 2)

    def test_manifest_is_plain_json(self, tmp_path):
        export_checkpoint(
            model_dir=tmp_path,
            model=TinyModule(),
            sample_input=torch.randn(1, 4),
            fmt="torchscript",
        )
        payload = json.loads((tmp_path / "bundle.json").read_text())
        assert isinstance(payload, dict) and "artifact" in payload
