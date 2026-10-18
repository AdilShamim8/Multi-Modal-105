# ADR-002: dataclass configs with strict YAML coercion

- **Status:** accepted (2026-10-18)
- **Context:** experiment configs were Python dataclasses constructed in
  code — no file-driven runs, no env-specific values, and no validation
  boundary between operator intent and program behaviour.
- **Decision:** keep dataclasses as the schema (type-checked, IDE-visible),
  add a YAML loader (`config/loader.py`) with `${ENV:default}`
  interpolation, `__base__` inheritance, and **unknown-key rejection**
  at coercion time.
- **Alternatives rejected:**
  - Hydra/OmegaConf — powerful but adds a dependency and its interpolation
    semantics would be another thing for every contributor to learn;
    our needs are interpolation + merge + strictness.
  - pydantic for experiment configs — validation yes, but dataclasses keep
    the research code dependency-light and are already in place.
- **Consequences:**
  - (+) A typo'd key fails at startup with the list of legal keys — not
    silently three epochs later.
  - (+) One config file works across dev/staging/prod via env vars.
  - (−) New config fields need a dataclass edit (by design: config surface
    is code-reviewed).
