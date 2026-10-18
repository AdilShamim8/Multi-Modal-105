# ADR-004: async-first API with bounded sync fallback

- **Status:** accepted (2026-10-18)
- **Context:** generation is throughput-bound (seconds per job); a
  request-response API that blocks on GPU completion would tie API
  capacity to GPU capacity — the worst coupling available.
- **Decision:** POST /v1/generations is **asynchronous by default**
  (202 + polling). A `sync=true` mode exists for interactive clients,
  bounded by `MM_SYNC_TIMEOUT_S` (default 30 s) — on overrun the job
  **falls back to the queue** and the client still gets a 202 envelope.
- **Consequences:**
  - (+) API replicas scale independently of GPU fleet size.
  - (+) Queue admission control is the single backpressure point — one
    knob, one alert, one runbook section.
  - (+) Sync mode never lies: it either answers within budget or hands
    the job to the path that can.
  - (−) Clients implement polling (mitigated: status URLs in the 202
    envelope; webhooks are a future additive surface).
