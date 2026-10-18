"""Locust load scenario: submission + polling mix against a live API.

Run manually or via the nightly workflow:
    locust -f tests/load/locustfile.py --headless -u 50 -r 10 -t 2m \
        --host http://localhost:8000

Models the traffic mix from docs/architecture/scaling-100m-users.md:
~85% async submissions (thin), ~15% sync (bounded GPU path) plus status
polling. The point is to catch latency regressions and shed behaviour,
not to simulate real generation load.
"""

from __future__ import annotations

import os

from locust import HttpUser, between, task

API_KEY = os.getenv("MM_LOAD_API_KEY", "dev-local-key")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

PROMPTS = [
    "a golden retriever puppy playing in snow",
    "a lighthouse in a storm, cinematic lighting",
    "isometric city at sunset, pastel colors",
    "macro photo of a dew-covered spider web",
    "a red fox in an autumn forest, oil painting",
]


class GenerationUser(HttpUser):
    """One interactive client: submit, poll, occasionally sync."""

    wait_time = between(1, 3)

    def on_start(self) -> None:
        self.client.get("/healthz", headers=HEADERS, name="/healthz")

    @task(6)
    def submit_async(self) -> None:
        with self.client.post(
            "/v1/generations",
            json={
                "prompt": PROMPTS[hash(self) % len(PROMPTS)],
                "num_steps": 12,       # draft profile: keep the queue honest
                "num_outputs": 1,
            },
            headers=HEADERS,
            name="/v1/generations [async]",
            catch_response=True,
        ) as response:
            if response.status_code in (202, 429, 503):
                # 429/503 are DESIGNED outcomes under load — count, don't fail.
                response.success()
            elif response.status_code == 400:
                response.failure("moderation rejected load traffic — check policy")
            elif response.status_code != 202:
                response.failure(f"unexpected {response.status_code}")

    @task(3)
    def poll_status(self) -> None:
        # Poll a synthetic job id — 404 is the expected cheap response;
        # it exercises the same hash-lookup path real polling uses.
        with self.client.get(
            "/v1/generations/loadtest-nonexistent",
            headers=HEADERS,
            name="/v1/generations/{id} [poll]",
            catch_response=True,
        ) as response:
            if response.status_code == 404:
                response.success()
            else:
                response.failure("status path regressed")

    @task(1)
    def submit_sync(self) -> None:
        with self.client.post(
            "/v1/generations",
            json={"prompt": "load test sync", "num_steps": 4, "sync": True},
            headers=HEADERS,
            name="/v1/generations [sync]",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 202, 429, 503):
                response.success()
            else:
                response.failure(f"sync path returned {response.status_code}")
