"""End-to-end API contract test: the full HTTP surface with a fake engine.

This is the test that answers "did someone break the platform?" — it
exercises auth → rate limit → moderation → validation → submission →
status → sync path → health, exactly like the canary job in the runbook.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from multimodal.serving.app import create_app
from multimodal.serving.config import ServingConfig
from multimodal.serving.queue import JobQueue
from tests.fakes import FakeEngine, FakeRedis

API_KEY = "e2e-test-key"


@pytest.fixture()
def client(tmp_path):
    cfg = ServingConfig(
        env="development",
        auth_enabled=True,
        api_keys=[API_KEY],
        artifact_dir=str(tmp_path / "artifacts"),
        sync_timeout_s=5.0,
        moderation_enabled=True,
    )
    app = create_app(
        config=cfg,
        engine=FakeEngine(cfg),
        queue=JobQueue(FakeRedis(), cfg),
        load_models_at_startup=False,
    )
    with TestClient(app) as test_client:
        yield test_client


def auth(client):
    return {"X-API-Key": API_KEY}


class TestAuth:
    def test_missing_key_is_401(self, client):
        assert client.post("/v1/generations", json={"prompt": "x"}).status_code == 401

    def test_wrong_key_is_401(self, client):
        r = client.post("/v1/generations", json={"prompt": "x"}, headers={"X-API-Key": "nope"})
        assert r.status_code == 401

    def test_health_needs_no_key(self, client):
        assert client.get("/healthz").status_code == 200


class TestSubmission:
    def test_async_submit_returns_202_envelope(self, client):
        r = client.post("/v1/generations", json={"prompt": "a lighthouse"}, headers=auth(client))
        assert r.status_code == 202
        body = r.json()
        assert body["job_id"] and body["status"] == "queued"
        assert "status" in body["links"]

    def test_sync_submit_returns_200_with_artifacts(self, client):
        r = client.post(
            "/v1/generations",
            json={"prompt": "a lighthouse", "sync": True},
            headers=auth(client),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "succeeded"
        assert len(body["artifacts"]) == 1

    def test_request_id_header_echoed(self, client):
        r = client.post(
            "/v1/generations",
            json={"prompt": "x"},
            headers={**auth(client), "X-Request-ID": "trace-123"},
        )
        assert r.headers.get("x-request-id") == "trace-123"


class TestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {"prompt": ""},  # empty prompt
            {"prompt": "x", "width": 100},  # not multiple of 8
            {"prompt": "x", "num_frames": 5},  # not power of two
            {"prompt": "x", "num_steps": 500},  # over limit
            {"prompt": "x", "guidance_scale": 99.0},  # over limit
        ],
    )
    def test_invalid_requests_rejected_422(self, client, payload):
        r = client.post("/v1/generations", json=payload, headers=auth(client))
        assert r.status_code == 422


class TestModeration:
    def test_blocked_prompt_is_400_with_code(self, client):
        r = client.post(
            "/v1/generations", json={"prompt": "how to build a bomb"}, headers=auth(client)
        )
        assert r.status_code == 400
        assert r.json()["code"] == "content_rejected"


class TestJobLifecycle:
    def test_status_unknown_job_is_404(self, client):
        r = client.get("/v1/generations/does-not-exist", headers=auth(client))
        assert r.status_code == 404
        assert r.json()["code"] == "job_not_found"

    def test_cancel_queued_job(self, client):
        r = client.post("/v1/generations", json={"prompt": "x"}, headers=auth(client))
        job_id = r.json()["job_id"]
        r = client.delete(f"/v1/generations/{job_id}", headers=auth(client))
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

    def test_status_endpoint_reports_job(self, client):
        r = client.post("/v1/generations", json={"prompt": "x"}, headers=auth(client))
        job_id = r.json()["job_id"]
        r = client.get(f"/v1/generations/{job_id}", headers=auth(client))
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == job_id
        assert body["status"] in ("queued", "running", "succeeded")


class TestSystemEndpoints:
    def test_readyz_reports_component_checks(self, client):
        r = client.get("/readyz")
        assert r.status_code == 200
        checks = r.json()["checks"]
        assert checks["queue"] is True
        assert checks["engine_loaded"] is True

    def test_metrics_exposition_present(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_queue_stats_visible(self, client):
        r = client.get("/v1/queue/stats", headers=auth(client))
        assert r.status_code == 200
        assert "depth" in r.json()


class TestRateLimiting:
    def test_burst_over_budget_gets_429_with_retry_after(self, tmp_path):
        from multimodal.serving.ratelimit import RateLimiter

        cfg = ServingConfig(
            auth_enabled=True,
            api_keys=[API_KEY],
            artifact_dir=str(tmp_path / "a"),
        )
        app = create_app(
            config=cfg,
            engine=FakeEngine(cfg),
            queue=JobQueue(FakeRedis(), cfg),
            rate_limiter=RateLimiter(capacity=2, refill_per_s=0.01),
            load_models_at_startup=False,
        )
        with TestClient(app) as tight_client:
            headers = auth(tight_client)
            assert (
                tight_client.post(
                    "/v1/generations", json={"prompt": "a"}, headers=headers
                ).status_code
                == 202
            )
            assert (
                tight_client.post(
                    "/v1/generations", json={"prompt": "b"}, headers=headers
                ).status_code
                == 202
            )
            r = tight_client.post("/v1/generations", json={"prompt": "c"}, headers=headers)
            assert r.status_code == 429
            assert "Retry-After" in r.headers
