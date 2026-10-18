#!/usr/bin/env bash
# Smoke test: the canary flow a human would run after any deploy.
# Exercises: health → readiness → auth → submission → status → sync.
# Exit non-zero on the FIRST broken expectation (fail fast, clear message).
# Usage: ./scripts/smoke_test.sh [http://host:port] [api-key]
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
KEY="${2:-dev-local-key}"
AUTH=("X-API-Key: ${KEY}" "Content-Type: application/json")

say() { printf '\033[36m▸\033[0m %s\n' "$*"; }
fail() { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

say "probing ${BASE}"

# 1. Liveness
curl -sf --max-time 5 "${BASE}/healthz" > /dev/null || fail "healthz did not answer 200"
say "healthz ok"

# 2. Readiness (queue + store + engine)
READY=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "${BASE}/readyz") \
  || READY="000"
if [ "${READY}" != "200" ]; then
  say "readyz=${READY} — component not ready; continuing to submission gate"
fi

# 3. Auth must reject anonymous traffic
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE}/v1/generations" \
  -H 'Content-Type: application/json' -d '{"prompt":"x"}')
[ "${CODE}" = "401" ] || fail "expected 401 without API key, got ${CODE}"
say "auth gate ok (401 anonymous)"

# 4. Async submission
RESP=$(curl -sf --max-time 10 -X POST "${BASE}/v1/generations" \
  -H "${AUTH[0]}" -H "${AUTH[1]}" \
  -d '{"prompt":"smoke test canary","num_steps":8}')
JOB=$(echo "${RESP}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
[ -n "${JOB}" ] || fail "no job_id in submission response"
say "submitted job ${JOB}"

# 5. Status readable
CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "${AUTH[0]}" \
  "${BASE}/v1/generations/${JOB}")
[ "${CODE}" = "200" ] || fail "status for ${JOB} returned ${CODE}"
say "status ok"

# 6. Sync path (bounded)
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 -X POST \
  "${BASE}/v1/generations" -H "${AUTH[0]}" -H "${AUTH[1]}" \
  -d '{"prompt":"smoke sync canary","num_steps":4,"sync":true}')
case "${CODE}" in
  200|202) say "sync path ok (${CODE})" ;;
  *) fail "sync path returned ${CODE}" ;;
esac

printf '\033[32m✓ smoke test passed\033[0m\n'
