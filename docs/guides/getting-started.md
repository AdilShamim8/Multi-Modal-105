# Getting Started

> Time budget: 10 minutes to first generation · Last audited: 2026-10-18

## 1. Install

```bash
git clone https://github.com/AdilShamim8/Multi-Modal-105.git
cd Multi-Modal-105
make install-dev        # CPU torch + package + dev tools + pre-commit
```

GPU hosts: install CUDA torch instead of the CPU wheel (see
`requirements.txt` header), then `pip install -e ".[all]"`.

## 2. Sanity check

```bash
mma --version                       # multimodal-agent 1.0.0 (released 2026-10-18)
make test-unit                      # green in < 90 s on CPU
```

## 3. First generation

```bash
# One-off generation via CLI (needs a model bundle — see step 4):
mma generate "a golden retriever puppy in snow, cinematic lighting" \
    --steps 30 --outputs 2 --out ./outputs
```

## 4. Point at models

No weights ship in the repo (see `.gitignore` — binaries never enter
git). Populate the registry directory:

```bash
export MM_MODEL_DIR=/var/lib/multimodal/models
python scripts/preload_models.py --dry-run   # shows the layout it expects:
#   $MM_MODEL_DIR/t2i-dit-s2/{bundle.json, weights.safetensors}
```

Any DiT checkpoint exported via `mma export` produces a valid bundle
(manifest + checksums are verified at load).

## 5. Local serving stack

```bash
make docker-up          # api + worker + redis + prometheus + grafana
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/v1/generations \
     -H "X-API-Key: dev-local-key" -H "Content-Type: application/json" \
     -d '{"prompt":"a lighthouse in a storm","sync":true}' | jq
```

Interactive API docs at `http://localhost:8000/docs`; Grafana at
`http://localhost:3000`.

## 6. Where to go next

| Goal | Read |
|---|---|
| Understand the system | [architecture](../architecture/system-architecture.md) |
| Train something | [training guide](training.md) |
| Operate the stack | [serving guide](serving.md), [runbook](../operations/runbook.md) |
| Learn the theory | [Tutorial-105](../tutorial/Tutorial-105.md) |

## Troubleshooting

* **`ModelNotFoundError`** — `MM_MODEL_DIR` unset or bundle missing; the
  error's `details` names the exact path checked.
* **Redis connection refused** — `make docker-up` starts Redis; or point
  `MM_REDIS_URL` at your own.
* **`401` on every request** — auth is on by default; dev key is
  `dev-local-key` in the compose stack, or set `MM_AUTH_ENABLED=false`
  locally.
