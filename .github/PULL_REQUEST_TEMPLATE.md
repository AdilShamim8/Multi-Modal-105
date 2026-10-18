<!-- PR template: reviewers read this before the diff. -->

## Summary

<!-- What changed and WHY — one paragraph, no code dumps. -->

## Change type

- [ ] `feat` — new capability
- [ ] `fix` — bug fix
- [ ] `refactor` — no behaviour change
- [ ] `perf` — performance
- [ ] `test` — tests only
- [ ] `docs` — documentation only
- [ ] `build` / `ci` / `chore` — tooling or infra

## How was it validated?

<!-- Be specific: commands run, tests added, configs exercised. -->
<!-- "It works on my machine" is not validation. -->

- [ ] `make lint` green
- [ ] `make test-unit` green
- [ ] New behaviour covered by a test (or N/A justification below)

## Blast radius

<!-- Which plane is touched: core / models / serving / infra / docs -->

- [ ] No public API changes
- [ ] Public API added/changed (docs + CHANGELOG updated)
- [ ] Deployment or rollout action required (runbook section referenced)

## Operational notes

<!-- Metrics affected? New env vars? Rollback procedure? Delete if N/A. -->

## Linked issues

<!-- Closes #123 · Relates to #456 -->
