#!/usr/bin/env bash
# verify_repo.sh — the FULL local quality gate, one command.
# Mirrors what CI runs on a PR: compile → lint → unit → e2e → YAML sanity.
# Usage: ./scripts/verify_repo.sh [--fast]   (--fast skips e2e+integration)
set -euo pipefail

cd "$(dirname "$0")/.."
FAST="${1:-}"

say()  { printf '\033[36m▸\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m✓ %s\033[0m\n' "$*"; }
fail() { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

say "1/7 byte-compile every source file"
python3 -m compileall -q src tests scripts && ok "syntax clean"

say "2/7 lint gate (mirrors the CI lint job)"
command -v ruff >/dev/null 2>&1 \
  || fail "ruff not on PATH: pip install ruff==0.7.4 (the version CI pins)"
ruff check src tests scripts
ruff format --check src tests scripts && ok "ruff check + format clean"

say "3/7 import sanity (torch-free core path)"
python3 - <<'PY' && ok "core imports cleanly"
import sys; sys.path.insert(0, "src")
import multimodal
from multimodal.version import RELEASE_DATE
assert RELEASE_DATE == "2026-10-18", RELEASE_DATE
print("version", multimodal.__version__, RELEASE_DATE)
PY

say "4/7 YAML + JSON manifests parse"
python3 - <<'PY' && ok "manifests parse"
import json, pathlib, yaml
count = 0
for pattern, loader in (("**/*.yaml", yaml.safe_load), ("**/*.json", json.load)):
    for path in pathlib.Path(".").rglob(pattern):
        if any(part in {".git", ".venv", "__pycache__", "node_modules"} for part in path.parts):
            continue
        with path.open() as fh:
            list(yaml.safe_load_all(fh)) if path.suffix == ".yaml" else loader(fh)
        count += 1
print(f"{count} manifests parsed")
PY

say "5/7 unit tests"
python3 -m pytest tests/unit -q --maxfail=5 && ok "unit suite green"

if [ "${FAST}" = "--fast" ]; then
  say "6/7 e2e+integration SKIPPED (--fast)"
else
  say "6/7 e2e + integration"
  python3 -m pytest tests/e2e tests/integration -q --maxfail=3 && ok "e2e + integration green"
fi

say "7/7 git tree clean check"
[ -z "$(git status --porcelain)" ] && ok "working tree clean" \
  || { echo "uncommitted changes:"; git status --short; fail "commit before pushing"; }

printf '\033[32m✓ repository verification passed\033[0m\n'
