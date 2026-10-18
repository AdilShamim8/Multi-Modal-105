# ADR-010: Test strategy — CPU-first, contract-focused

- **Status:** accepted (2026-10-18)
- **Context:** a research repo with zero tests accumulated seven latent
  defects (broken entry points, unimportable DiT, unconstructible U-Net,
  crashing GAN layers) that no one could have caught by reading. The
  serving plane needed guarantees without GPU CI budgets.
- **Decision:**
  - **Tiers with distinct budgets**: unit (PR gate, < 90 s CPU),
    integration (nightly, real model stack at toy scale), e2e (HTTP
    contract with fakes), load (locust, manual/nightly).
  - **Tiny geometries over mocks**: a 8×8-latent DiT and a 32-ch VAE are
    the real code paths at ~1/1000 the cost. Mocks are reserved for
    boundaries (Redis, S3, GPU engines).
  - **Contract tests pin invariants**: ᾱ monotonicity, DDIM seed
    determinism, straight-through identity gradient, metric name
    stability — properties later work must not silently break.
  - **Test doubles shared, not copy-pasted**: `tests/fakes.py`.
- **Consequences:**
  - (+) PR feedback in minutes; nightly catches deep integration drift.
  - (+) Every fixed defect today arrived with a regression test — the
    suite is an incident log, not decoration.
  - (−) Toy-scale tests can miss precision/OOM behaviours that only
    appear at real resolution; that residual risk is owned by the
    nightly tier + benchmark harness, not ignored.
