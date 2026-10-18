# Contributing to the Multi-Modal Generation Platform

Thank you for investing time in this project. This document defines the
contract between you and every other engineer who touches this codebase.
Following it keeps 199+ commits of history readable and the platform stable.

## Ground Rules

1. **`main` is always green.** Every commit on `main` must pass
   `make lint`, `make typecheck`, and `make test-unit`. CI enforces this.
2. **Small, atomic commits.** One logical change per commit. If a commit
   needs the word "and" in the subject, split it.
3. **Conventional commits.** Use `type(scope): subject` (see `.gitmessage`).
   Release automation derives versions from commit types.
4. **No binaries in git.** Model weights, datasets and videos live in object
   storage. Pre-commit blocks them; so should you.
5. **Research code is not an excuse.** Type hints, docstrings, and a test
   for every bug you fix. Toy project habits are how outages happen.

## Development Loop

```bash
git clone https://github.com/AdilShamim8/Multi-Modal-105.git
cd Multi-Modal-105
make install-dev          # editable install + pre-commit hooks
make lint && make test    # must be green before you commit
```

## Pull Requests

- Branch from `main`: `feat/<scope>-<slug>`, `fix/<scope>-<slug>`.
- PRs < 400 changed lines get reviewed within a day; larger ones get split.
- Fill the PR template (`.github/PULL_REQUEST_TEMPLATE.md`).
- New public APIs require: docstring, unit test, and a docs update.
- Security-relevant changes require a SECURITY review note (see SECURITY.md).

## Code Standards

| Area         | Standard                                                  |
|--------------|------------------------------------------------------------|
| Formatting   | `ruff format` (line length 100) — no debates, run the tool |
| Linting      | `ruff check` — zero warnings on changed lines               |
| Types        | `mypy` on `src/` — new code must be annotation-clean       |
| Imports      | `multimodal.*` absolute; stdlib → third-party → local      |
| Tests        | pytest; unit tests run on CPU in < 90 s                    |
| Docs         | Update `docs/` in the same commit as the behaviour change  |

## Architecture Changes

Changes that alter component boundaries, data flow, or the failure model
require an **ADR** (`docs/adr/`) before code. Read
`docs/architecture/system-architecture.md` first — if your change
invalidates a diagram, the diagram ships with your change.

## Reporting Bugs

Open an issue using the bug template; include: environment, config, log
excerpt, and a minimal reproducer. Bugs fixed without a regression test
are bugs that return.

## Recognition

Every release's CHANGELOG credits contributors by handle. Consistent,
careful work gets you into CODEOWNERS for the area you know best.

---

Maintainer: [@AdilShamim8](https://github.com/AdilShamim8) · Last updated: 2026-10-18
