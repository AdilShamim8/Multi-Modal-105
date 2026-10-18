# ADR-008: defence-in-depth request limits

- **Status:** accepted (2026-10-18)
- **Context:** one limit layer anywhere is a single point of failure for
  abuse: schema-only limits die to a buggy client, app-only limits die to
  replica skew, edge-only limits die to config drift.
- **Decision:** the same bounds are enforced at FOUR layers, each
  cheapest-possible at its position:
  1. **Ingress** (nginx annotations): rps 100, connections 200, body 256k
  2. **Schema** (pydantic): geometry, steps ≤ 100, frames power-of-two,
     prompt ≤ 2000 chars — 422 before code runs
  3. **App** (token bucket per key): 20 tokens, 5/s refill, Retry-After
  4. **Admission** (queue depth): 503 + Retry-After when full
- **Consequences:**
  - (+) No single bypass; abusive traffic is stopped at the cheapest
    layer it hits (edge < schema < app < queue).
  - (+) Limits are independently tunable per environment.
  - (−) The same number appears in several files — mitigated by
    documenting the mirroring in `.env.example` and the serving config
    docstring; drift shows up in config review, not incidents.
