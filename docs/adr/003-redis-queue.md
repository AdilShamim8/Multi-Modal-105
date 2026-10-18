# ADR-003: Redis list queue with reliable-queue semantics

- **Status:** accepted (2026-10-18)
- **Context:** the serving plane needs a job queue that survives worker
  crashes, supports admission control, and exposes depth to autoscaling.
  Celery/RabbitMQ/SQS are the default answers; each brings a broker,
  serialization framework, and operational surface.
- **Decision:** Redis lists with the BRPOPLPUSH reliable-queue pattern
  (`serving/queue.py`): enqueue via LPUSH (depth-checked), consume into a
  `processing` list, ack with LREM, and requeue strays at worker startup
  (`recover_stuck`). Job state lives in hashes with TTL.
- **Alternatives rejected:**
  - Celery — a worker framework around exactly the loop we want to own
    (batching, degradation, drain); its churn policies fight GPU draining.
  - SQS — cloud lock-in for a reference architecture; visibility timeouts
    replicate our processing-list semantics anyway.
- **Consequences:**
  - (+) One dependency (Redis) already needed for rate limits/caching.
  - (+) Queue depth is a native KEDA trigger — autoscaling falls out.
  - (−) At-least-once delivery (bounded by retry ≤ 2); duplicate cost
    accepted, duplicate media prevented by content-addressed keys.
  - (−) Redis availability becomes S1 — mitigated by AOF + replica and
    the documented failover runbook (FMEA #3).
