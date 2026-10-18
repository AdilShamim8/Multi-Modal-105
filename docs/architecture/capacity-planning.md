# Capacity Planning Model

> Status: current · Last audited: 2026-10-18 · Companion: [scaling-100m-users.md](scaling-100m-users.md)

## 1. Per-job GPU cost (the atom of capacity)

Measured method — `mma generate` on an idle L4, 512×512, DPM++, batch 8,
median of 20 runs (see `scripts/benchmark_serving.py` for the harness):

| Profile | Steps | Wall/job (batched) | Throughput/GPU |
|---|---|---|---|
| fast_draft | 12 | ~1.1 s | ~7 jobs/s |
| balanced (default) | 30 | ~2.6 s | ~2.5 jobs/s |
| high_fidelity | 50 | ~4.4 s | ~1.5 jobs/s |
| video 16f/256px | 30 | ~14 s | ~0.19 jobs/s |

Capacity rule of thumb: **plan the fleet against `balanced`**, since it
is the default profile; treat video as a separate pool if its share
exceeds ~15% (its cost profile diverges by an order of magnitude).

## 2. Queueing model (Little's law, honestly applied)

```
L = λ × W        L: in-flight jobs, λ: arrival rate, W: time in system
```

With λ_peak = 926 jobs/s and a target queue-wait W_q ≤ 45 s (p95):

```
Workers needed = λ × (W_gen + W_overhead) / batch_efficiency
               = 926 × (2.6 + 0.4) / 1.8  ≈ 154 worker-GPUs at peak
```

Plus 30% for burst variance and signature fragmentation → **~200 peak
GPUs**, consistent with the scaling doc's fleet estimate (the difference
is the video share assumption; both are stated, neither is magic).

## 3. Headroom policy

| Signal | Trigger | Action |
|---|---|---|
| KEDA target (8/worker) | sustained 15 min above | scale-out to cap; ticket capacity |
| Queue p95 wait | > 60 s for 30 min | SLO breach alert; capacity review |
| Admission 503 rate | > 0.1% of submissions | IMMEDIATE capacity or load action |
| GPU util (DCGM) | < 40% fleet-wide | batching regression — investigate before scaling DOWN |

## 4. Cold-start budget

Node provisioning (~90 s) + image pull (cached, ~20 s) + model load
(~35 s measured, exported bundle) + warmup (~10 s) ≈ **2.5 min worst
case**. The warm floor of 24 GPUs exists so that no user-visible cold
start happens below a 10× traffic surge; above that, surge capacity
absorbs latency, not availability.

## 5. Cost model summary

| Lever | Effect | Where |
|---|---|---|
| Batching (signature grouping) | 1.6–2.2× throughput/GPU | config `MM_MAX_BATCH_SIZE` + scheduler |
| Step count default | linear with steps | `configs/t2i/sampling.yaml` profiles |
| Spot worker pool | −40–60% cost | terraform `gpu_type` + taints; checkpoint-resumable by design |
| Queue target (backlog/worker) | latency↔cost dial | KEDA trigger `listLength` |

Blended estimate at 200M MAU: **~$1M/year** (on-demand) with the levers
above — the model in the scaling doc §5 shows the arithmetic.
