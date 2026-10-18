# ADR-001: src-layout package under `multimodal`

- **Status:** accepted (2026-10-18)
- **Context:** the repo grew as flat root packages (`models/`, `training/`,
  ...) — importable only from the repo root, colliding with any tooling
  that puts the repo on `sys.path`, with console scripts referencing
  modules that did not exist.
- **Decision:** migrate everything into `src/multimodal/**` (src layout),
  preserve history with `git mv`, rewrite imports mechanically, and make
  `pyproject.toml` the sole packaging source with `py.typed` shipped.
- **Consequences:**
  - (+) Importable after `pip install -e .` from any directory; CI, IDE
    and mypy resolve the same package.
  - (+) Root directory now contains only documents and tooling — the
    "is this a product?" question is answered by `ls`.
  - (−) One-time import rewrite touched every file; mitigated by doing
    it package-by-package with syntax gates between steps.
