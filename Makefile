# ─────────────────────────────────────────────────────────────────────────────
# Multi-Modal Generation Platform — developer workflow
# Every target is safe to run repeatedly. `make help` lists what is available.
# ─────────────────────────────────────────────────────────────────────────────

SHELL          := /bin/bash
.DEFAULT_GOAL  := help
PYTHON         ?= python3
PIP            ?= $(PYTHON) -m pip
PKG            := multimodal
export PYTHONPATH ?= src

# ── Bootstrap ────────────────────────────────────────────────────────────────
.PHONY: help install install-dev install-cpu

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies (CPU torch wheel)
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e .

install-dev: install ## Install with dev/test extras + pre-commit hooks
	$(PIP) install -e ".[dev,test]"
	pre-commit install

# ── Quality gates ────────────────────────────────────────────────────────────
.PHONY: lint format typecheck compile

lint: ## Run ruff (lint + import ordering)
	ruff check src tests scripts

format: ## Auto-format with ruff
	ruff format src tests scripts && ruff check --fix src tests scripts

typecheck: ## Run mypy (non-strict gate)
	mypy src/$(PKG)

compile: ## Byte-compile every source file (fast syntax gate)
	$(PYTHON) -m compileall -q src tests scripts

# ── Tests ────────────────────────────────────────────────────────────────────
.PHONY: test test-unit test-integration test-e2e coverage

test: test-unit ## Default = fast unit suite
test-unit: ## Run unit tests
	$(PYTHON) -m pytest tests/unit -q --maxfail=5

test-integration: ## Run integration tests (needs models on CPU, slower)
	$(PYTHON) -m pytest tests/integration -q

test-e2e: ## Run end-to-end API tests
	$(PYTHON) -m pytest tests/e2e -q

coverage: ## Unit tests with coverage report
	$(PYTHON) -m pytest tests/unit --cov=src/$(PKG) --cov-report=term-missing

# ── Serving (local) ──────────────────────────────────────────────────────────
.PHONY: serve serve-dev worker

serve: ## Launch API server (production settings)
	$(PYTHON) -m multimodal.cli serve --host 0.0.0.0 --port 8000

serve-dev: ## Launch API with hot-reload (development only)
	$(PYTHON) -m multimodal.cli serve --reload --host 127.0.0.1 --port 8000

worker: ## Launch a generation worker (consumes the Redis queue)
	$(PYTHON) -m multimodal.cli worker

# ── Training / eval entry points ─────────────────────────────────────────────
.PHONY: train-t2i train-t2v evaluate

train-t2i: ## Train the T2I diffusion model with the default config
	$(PYTHON) -m multimodal.cli train --pipeline t2i --config configs/t2i/dit_base.yaml

train-t2v: ## Train the T2V diffusion model with the default config
	$(PYTHON) -m multimodal.cli train --pipeline t2v --config configs/t2v/dit_video_base.yaml

evaluate: ## Run the evaluation suite on a checkpoint
	$(PYTHON) -m multimodal.cli evaluate --config configs/eval/default.yaml

# ── Container workflows ──────────────────────────────────────────────────────
.PHONY: docker-api docker-worker docker-up docker-down

docker-api: ## Build the API serving image
	docker build -f deployments/docker/Dockerfile.api -t multimodal/api:local .

docker-worker: ## Build the GPU worker image
	docker build -f deployments/docker/Dockerfile.worker -t multimodal/worker:local .

docker-up: ## Bring up the local docker-compose stack (api + redis + worker)
	docker compose -f deployments/docker/docker-compose.yml up -d

docker-down: ## Tear the local stack down
	docker compose -f deployments/docker/docker-compose.yml down

# ── Housekeeping ─────────────────────────────────────────────────────────────
.PHONY: clean hooks stats

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

hooks: ## (Re)install pre-commit hooks
	pre-commit install

stats: ## Show repo statistics
	@echo "tracked files : $$(git ls-files | wc -l)"
	@echo "python files  : $$(git ls-files '*.py' | wc -l)"
	@echo "lines of py   : $$(git ls-files '*.py' | xargs wc -l | tail -1)"

verify: ## Full local gate: compile, imports, manifests, unit+e2e tests
	./scripts/verify_repo.sh

verify-fast: ## Same gate, skipping e2e/integration (~60 s)
	./scripts/verify_repo.sh --fast
