# ADR-007: KEDA queue-depth autoscaling for the GPU tier

- **Status:** accepted (2026-10-18)
- **Context:** the GPU worker fleet is the platform's dominant cost and
  its dominant latency term. CPU-based HPA is the default answer and is
  wrong here: workers are CPU-idle while denoising — utilization
  measures nothing useful about demand.
- **Decision:** KEDA ScaledObject on the Redis queue list length, target
  8 jobs/worker, min 2 (warm floor), max 60 (cost circuit-breaker),
  cooldown 300 s; fallback floor if Redis metrics are unreachable.
- **Consequences:**
  - (+) Capacity tracks DEMAND (backlog), with a 15 s reaction time.
  - (+) The warm floor bounds cold-start latency; the cap bounds cost —
    both are explicit, reviewed numbers.
  - (+) Cooldown prevents replica flapping (GPU nodes are slow to
    provision — flapping would be pure cost).
  - (−) KEDA is one more controller in the cluster; accepted for a
    scaling signal no built-in controller provides.
