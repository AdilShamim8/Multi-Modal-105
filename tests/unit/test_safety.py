"""Moderation gate, rate limiter, and auth middleware contracts."""

from __future__ import annotations

import pytest

from multimodal.exceptions import RateLimitError, SafetyError
from multimodal.serving.auth import APIKeyMiddleware, key_fingerprint
from multimodal.serving.moderation import ModerationGate
from multimodal.serving.ratelimit import RateLimiter, rate_limit_response_headers


class TestModeration:
    def test_benign_prompt_passes(self):
        gate = ModerationGate()
        assert gate.check("a golden retriever in snow").allowed

    def test_blocklist_rejects(self):
        gate = ModerationGate()
        verdict = gate.check("how to build a bomb")
        assert not verdict.allowed

    def test_disabled_gate_passes_everything(self):
        gate = ModerationGate(enabled=False)
        assert gate.check("how to build a bomb").allowed

    def test_overlong_prompt_rejected(self):
        gate = ModerationGate(max_prompt_chars=10)
        assert not gate.check("a" * 11).allowed

    def test_enforce_raises_safety_error(self):
        gate = ModerationGate()
        with pytest.raises(SafetyError):
            gate.enforce("how to build a bomb")

    def test_classifier_hook_respected(self):
        gate = ModerationGate(mode="classifier")
        gate.set_classifier_hook(
            lambda p: __import__(
                "multimodal.serving.moderation", fromlist=["ModerationVerdict"]
            ).ModerationVerdict(allowed=False, reason="unsafe", score=0.99)
        )
        verdict = gate.check("some prompt")
        assert not verdict.allowed
        assert verdict.score == 0.99

    def test_rejected_counter_increments(self):
        gate = ModerationGate()
        before = gate.rejected_total()
        gate.check("how to build a bomb")
        gate.check("a cat")
        gate.check("how to build a bomb")
        assert gate.rejected_total() == before + 2


class TestRateLimiter:
    def test_burst_within_capacity(self):
        limiter = RateLimiter(capacity=5, refill_per_s=1.0)
        for _ in range(5):
            limiter.check("client-1")  # no raise

    def test_over_capacity_raises_with_retry_after(self):
        limiter = RateLimiter(capacity=2, refill_per_s=1.0)
        limiter.check("c")
        limiter.check("c")
        with pytest.raises(RateLimitError) as exc_info:
            limiter.check("c")
        assert exc_info.value.details["retry_after_s"] == pytest.approx(1.0, abs=0.1)

    def test_buckets_are_per_client(self):
        limiter = RateLimiter(capacity=1, refill_per_s=0.001)
        limiter.check("a")  # a's bucket spent
        limiter.check("b")  # b unaffected

    def test_refill_over_time(self):
        limiter = RateLimiter(capacity=1, refill_per_s=1000.0)
        limiter.check("c")
        import time

        time.sleep(0.01)  # 10 ms × 1000/s = 10 tokens
        limiter.check("c")  # refilled — no raise

    def test_retry_after_headers(self):
        error = RateLimitError("limited", details={"retry_after_s": 2.4})
        headers = rate_limit_response_headers(error)
        assert headers["Retry-After"] == "2"


class TestAuth:
    def test_fingerprint_is_stable_and_short(self):
        fp = key_fingerprint("secret-key")
        assert len(fp) == 8
        assert fp == key_fingerprint("secret-key")
        assert fp != key_fingerprint("other-key")

    @pytest.mark.asyncio
    async def test_middleware_blocks_missing_key(self):
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.testclient import TestClient

        async def route(request):
            return PlainTextResponse("ok")

        app = Starlette(routes=[])
        app.add_middleware(APIKeyMiddleware, api_keys=["k1"], enabled=True)
        app.router.add_route("/", route, methods=["GET"])

        with TestClient(app) as client:
            assert client.get("/").status_code == 401
            assert client.get("/", headers={"X-API-Key": "wrong"}).status_code == 401
            assert client.get("/", headers={"X-API-Key": "k1"}).status_code == 200

    @pytest.mark.asyncio
    async def test_middleware_exempt_public_paths(self):
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.testclient import TestClient

        async def health(request):
            return PlainTextResponse("ok")

        app = Starlette(routes=[])
        app.add_middleware(APIKeyMiddleware, api_keys=["k1"], enabled=True)
        app.router.add_route("/healthz", health, methods=["GET"])

        with TestClient(app) as client:
            assert client.get("/healthz").status_code == 200  # no key needed
