"""Inference engine: OOM degradation, deterministic seeding, encoding."""

from __future__ import annotations

import io

import pytest
from tests.fakes import FakeRedis  # noqa: F401 - import sanity for CI

from multimodal.exceptions import OutOfMemoryError
from multimodal.serving.config import ServingConfig
from multimodal.serving.engine import InferenceEngine
from multimodal.serving.jobs import GenerationJob
from multimodal.serving.schemas import new_job_id


class OomPipeline:
    """Fails with CUDA-style OOM RuntimeError unless degraded."""

    def __init__(self, fail_until_outputs: int = 1, fail_until_steps: int = 16) -> None:
        self.fail_until_outputs = fail_until_outputs
        self.fail_until_steps = fail_until_steps
        self.calls: list[dict] = []

    def generate(self, **params):
        self.calls.append(dict(params))
        if (
            params["num_outputs"] > self.fail_until_outputs
            or params["num_steps"] > self.fail_until_steps
        ):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buf, format="PNG")
        return {"images": [Image.new("RGB", (32, 32))] * params["num_outputs"]}


def make_job(**kwargs):
    defaults = {"job_id": new_job_id(), "prompt": "p", "num_outputs": 4, "num_steps": 30}
    defaults.update(kwargs)
    return GenerationJob(**defaults)


@pytest.fixture()
def engine(tmp_path):
    cfg = ServingConfig(artifact_dir=str(tmp_path))
    eng = InferenceEngine(cfg)
    eng.t2i = OomPipeline()
    eng._loaded["t2i"] = True
    return eng


class TestOOMDegradation:
    def test_oom_retries_with_smaller_batch_and_steps(self, engine):
        output = engine.execute(make_job(num_outputs=4, num_steps=30))
        # First call OOMs at full params; second call must be degraded.
        assert len(engine.t2i.calls) == 2
        first, second = engine.t2i.calls
        assert first["num_outputs"] == 4 and first["num_steps"] == 30
        assert second["num_outputs"] == 1 and second["num_steps"] == 15
        assert output.seed_used is not None

    def test_persistent_oom_raises_platform_error(self):
        class AlwaysOom:
            def generate(self, **params):
                raise RuntimeError("CUDA out of memory.")

        cfg = ServingConfig()
        eng = InferenceEngine(cfg)
        eng.t2i = AlwaysOom()
        eng._loaded["t2i"] = True
        with pytest.raises(OutOfMemoryError):
            eng.execute(make_job())

    def test_non_oom_runtime_error_propagates(self):
        class Broken:
            def generate(self, **params):
                raise RuntimeError("kernel fault")

        cfg = ServingConfig()
        eng = InferenceEngine(cfg)
        eng.t2i = Broken()
        eng._loaded["t2i"] = True
        with pytest.raises(RuntimeError, match="kernel fault"):
            eng.execute(make_job())


class TestDeterministicSeeding:
    def test_same_job_id_same_seed(self, engine):
        job = make_job()
        out1 = engine.execute(job)
        out2 = engine.execute(make_job(job_id=job.job_id))
        assert out1.seed_used == out2.seed_used

    def test_explicit_seed_respected(self, engine):
        out = engine.execute(make_job(seed=999))
        assert out.seed_used == 999

    def test_different_jobs_decorrelate(self, engine):
        a = engine.execute(make_job())
        b = engine.execute(make_job())
        assert a.seed_used != b.seed_used


class TestOutputEncoding:
    def test_outputs_are_png_bytes(self, engine):
        out = engine.execute(make_job(num_outputs=1, num_steps=8))
        assert out.artifacts and isinstance(out.artifacts[0], bytes)
        assert out.artifacts[0][:4] == b"\x89PNG"  # PNG magic

    def test_duration_reported(self, engine):
        out = engine.execute(make_job(num_steps=8))
        assert out.duration_s >= 0.0
