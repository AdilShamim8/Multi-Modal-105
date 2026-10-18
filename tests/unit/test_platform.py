"""Metrics contract, CLI parser, and logging correlation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from multimodal.cli import _build_parser, main
from multimodal.logging import (
    configure_logging,
    get_correlation_id,
    new_correlation_id,
    set_correlation_id,
)
from multimodal.serving import metrics
from multimodal.serving.schemas import GenerationRequest


class TestMetrics:
    def test_recording_never_raises_without_prometheus(self):
        # Stub mode (no prometheus_client) must not crash callers.
        metrics.record_request("/v1/generations", "POST", 200, 0.01)
        metrics.record_generation("image", "succeeded", 1.2)
        metrics.set_queue_depth(5)
        metrics.set_active_jobs(2)
        metrics.set_model_load_seconds(3.1)

    def test_metric_names_match_alert_contract(self):
        """alerts.yml references these exact names — renaming is a breaking
        change (ADR-011). Freeze them here."""
        with open(metrics.__file__) as fh:
            source = fh.read()
        for name in (
            "mm_api_requests_total",
            "mm_api_request_latency_seconds",
            "mm_generations_total",
            "mm_generation_latency_seconds",
            "mm_queue_depth",
            "mm_active_jobs",
            "mm_model_load_seconds",
        ):
            assert name in source, f"metric {name} missing from contract"


class TestCliParser:
    def test_version_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "1.0.1" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "argv,expected",
        [
            (["train", "--pipeline", "gan", "--config", "c.yaml"], "gan"),
            (["train", "--pipeline", "t2v", "--config", "c.yaml"], "t2v"),
            (["generate", "hello world"], "hello world"),
        ],
    )
    def test_subcommand_parsing(self, argv, expected):
        parser = _build_parser()
        args = parser.parse_args(argv)
        values = vars(args)
        assert expected in values.values()

    def test_generate_flags(self):
        args = _build_parser().parse_args(
            [
                "generate",
                "p",
                "--steps",
                "12",
                "--sampler",
                "ddim",
                "--outputs",
                "2",
                "--seed",
                "7",
            ]
        )
        assert (args.steps, args.sampler, args.outputs, args.seed) == (12, "ddim", 2, 7)

    def test_invalid_pipeline_rejected(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["train", "--pipeline", "nope", "--config", "c"])

    def test_missing_command_rejected(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])


class TestLoggingCorrelation:
    def test_new_correlation_id_scopes(self):
        first = new_correlation_id()
        assert get_correlation_id() == first
        second = new_correlation_id()
        assert second != first
        assert get_correlation_id() == second

    def test_adopt_external_correlation_id(self):
        set_correlation_id("trace-from-header")
        assert get_correlation_id() == "trace-from-header"

    def test_configure_logging_idempotent(self):
        configure_logging("WARNING")
        configure_logging("WARNING")
        import logging

        assert logging.getLogger().level == logging.WARNING


class TestSchemas:
    def test_video_frames_power_of_two(self):
        with pytest.raises(ValidationError, match="power of two"):
            GenerationRequest(prompt="x", modality="video", num_frames=12)

    def test_image_defaults(self):
        req = GenerationRequest(prompt="a cat")
        assert req.modality.value == "image"
        assert req.num_frames == 16  # default valid even for images

    def test_prompt_strip(self):
        req = GenerationRequest(prompt="  padded  ")
        assert req.prompt == "padded"
