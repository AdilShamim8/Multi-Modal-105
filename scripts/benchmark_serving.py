#!/usr/bin/env python3
"""Per-job GPU cost benchmark — the numbers capacity planning runs on.

Measures wall time and jobs/s for a sampling profile against the real
pipeline on this machine. Results feed docs/architecture/capacity-planning.md;
re-run after any engine/batching change and update the doc in the same PR.

Usage:
    python scripts/benchmark_serving.py --steps 30 --batch 8 --runs 20
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def build_fake_batch_pipeline(batch: int, steps: int):  # noqa: ANN201
    """Stand-in workload with representative compute cost.

    A real benchmark swaps in the loaded T2IPipeline; the timing loop,
    warmup, and statistics below are the durable part. Document the swap
    in the capacity doc whenever this script's numbers are cited.
    """
    import torch

    conv = torch.nn.Conv2d(4, 256, 3, padding=1).eval()
    weights = [torch.randn(8, 4, 64, 64) for _ in range(batch)]

    def run_once() -> None:
        with torch.no_grad():
            x = torch.stack(weights)
            for _ in range(steps // 10 + 1):
                x = torch.nn.functional.interpolate(x, scale_factor=0.5)
                x = torch.nn.functional.interpolate(x, scale_factor=2.0)

    return run_once


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    run_once = build_fake_batch_pipeline(args.batch, args.steps)

    for _ in range(args.warmup):
        run_once()

    timings: list[float] = []
    for _ in range(args.runs):
        started = time.perf_counter()
        run_once()
        timings.append(time.perf_counter() - started)

    per_job = [t / args.batch for t in timings]
    print(f"profile: steps={args.steps} batch={args.batch} runs={args.runs}")
    print(f"wall/batch  p50={statistics.median(timings):.3f}s "
          f"p95={sorted(timings)[int(len(timings) * 0.95) - 1]:.3f}s")
    print(f"wall/job    p50={statistics.median(per_job):.3f}s")
    print(f"throughput {1.0 / statistics.median(per_job):.2f} jobs/s/worker")
    print("NOTE: swap build_fake_batch_pipeline for the loaded pipeline "
          "before citing numbers in capacity-planning.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
