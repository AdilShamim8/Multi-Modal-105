# FAQ

> Last updated: 2026-10-18 · Real questions this architecture raises.

## General

**Is this a trained model I can use today?**
No — and the repo is honest about it. It ships the complete *system* around
the models: architectures, training loops, serving plane, evaluation gates.
Weights are absent by policy (`.gitignore` blocks binaries); you train or
load bundles via the registry. See [getting started](guides/getting-started.md).

**Why does the README call itself a "platform" if weights are missing?**
Because the engineering claim is about the *system*: admission control,
queueing, crash recovery, safety gates, autoscaling contracts. Those are
testable without pretrained weights — and 186 tests do exactly that.

**How is this different from using diffusers/HF Serving?**
diffusers is a superb model library — this repo *consumes that level of
abstraction* (pipeline-style APIs) but builds the platform around it:
reliable queueing, KEDA autoscaling, moderation, SLOs, runbooks. The
serving plane never needed to reimplement diffusion math; it needed to
engineer around it.

## Serving

**Why 202 Accepted instead of holding the connection?**
Generation is throughput-bound; held connections couple API capacity to
GPU capacity — the worst coupling available. Sync mode exists under a
30 s budget with queue fallback (ADR-004).

**What happens when the queue fills up?**
Admission control returns `503 + Retry-After` — a fast honest no. The
alert `MmQueueNearCapacity` fires at 9,000/10,000 first. See
[runbook §3](operations/runbook.md).

**A worker crashed mid-job — is the job lost?**
No. The reliable-queue pattern (BRPOPLPUSH into a processing list) plus
startup `recover_stuck()` requeues orphans; in-flight work is bounded by
the 10-minute recovery window. Tested in `tests/unit/test_queue.py`.

**Why Redis and not Kafka/SQS?**
ADR-003. One dependency already in the hot path, native KEDA trigger,
at-least-once semantics that content-addressed artifacts make
effectively-once. The moment fan-out to multiple consumer families
appears, that ADR gets revisited.

## Training

**Which denoiser should I train — U-Net or DiT?**
DiT (ADR-006): the scaling math, the video path, and the capacity
planning all target it. U-Net stays as a tested reference/comparison.

**Why does my loss sit at ~1.0 and not fall for the first hours?**
v-prediction targets are normalized — a healthy start IS ~1.0. Watch it
fall slowly; a loss near 0.01 early means the model collapsed to
trivial t≈0 predictions. Check the timestep sampler.

**When do I trust a checkpoint for promotion?**
Only through the eval gates: FID ≤ 25, IS ≥ 8, CLIP ≥ 0.25 on 10k
samples, seed recorded, EMA weights. Then `mma export` — never hand-copy.

## Reliability

**What breaks first at scale?**
GPU fleet, by design — everything else has headroom or sheds gracefully.
[The scaling doc](architecture/scaling-100m-users.md) shows the arithmetic.

**Redis died. What now?**
API drops out of the LB (readyz), queued jobs survive AOF, workers hold
their floor via KEDA fallback. Runbook §4 walks the recovery.

**How do I rotate API keys with zero downtime?**
Comma-list old+new, roll API pods, remove old key in a second pass.
Runbook §7.

**Something generated unsafe media — what do I do first?**
Incident response SEV-1 flow: kill-switch before diagnosis, preserve the
audit trail, scope by correlation id. [That doc](operations/incident-response.md)
is the checklist.
