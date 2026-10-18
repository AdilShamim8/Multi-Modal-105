# Security Policy

The Multi-Modal Generation Platform handles untrusted user prompts, API keys,
and generated media. Treat security as a release blocker, not a feature.

## Supported Versions

| Version | Supported          | Notes                                   |
|---------|--------------------|------------------------------------------|
| 1.0.x   | yes (current)      | security fixes land on `main` + patch tag |
| < 1.0   | no                 | upgrade required                         |

## Reporting a Vulnerability

**Do not open a public issue for security problems.**

1. Use GitHub **Private Vulnerability Reporting** on this repository, or
2. Email the maintainer via the GitHub profile with subject `SEC: <summary>`.

Include: reproduction steps, affected component (API / worker / queue /
model), and impact assessment. You will receive an acknowledgement within
72 hours and a fix-or-mitigation timeline within 14 days. Coordinated
disclosure is honoured in the CHANGELOG for every valid report.

## Threat Model (summary)

| Threat                          | Primary control                                      |
|---------------------------------|------------------------------------------------------|
| Prompt injection → unsafe media | moderation gate pre-queue + post-generation re-check |
| Credential leakage in images    | secret scan (gitleaks) + env-only secret loading     |
| Key compromise                  | per-key rate limits, rotation, least-privilege scopes |
| Queue flooding / DoS            | admission control (`MM_QUEUE_MAX_DEPTH`) + shedding   |
| Artifact poisoning              | signed URLs, checksummed writes, bucket policies      |
| Dependency CVEs                 | Dependabot + `pip-audit` in the security workflow     |

Full details: `docs/architecture/failure-modes.md` and
`docs/operations/incident-response.md`.

## Hardening Checklist (production deployments)

- [ ] `MM_AUTH_ENABLED=true`, keys issued per-tenant from a secret store
- [ ] TLS terminated at the ingress; mTLS between API and workers
- [ ] Redis requires auth and lives in a private subnet
- [ ] S3 bucket: no public ACLs, SSE-KMS, access logs enabled
- [ ] Container images scanned (Trivy) and run as non-root UID 10001
- [ ] NetworkPolicies deny by default (see `deployments/k8s/`)

---

Last updated: 2026-10-18
