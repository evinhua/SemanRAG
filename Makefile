.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Development ──────────────────────────────────────────────────
.PHONY: dev
dev:
	uv pip install -e ".[all]" || pip install -e ".[all]"

# ── Environment Setup Wizard ─────────────────────────────────────
.PHONY: env-base env-storage env-server env-validate env-security-check
env-base:
	python -m semanrag.setup_wizard env-base

env-storage:
	python -m semanrag.setup_wizard env-storage

env-server:
	python -m semanrag.setup_wizard env-server

env-validate:
	python -m semanrag.setup_wizard validate

env-security-check:
	python -m semanrag.setup_wizard security-check

# ── Quality ──────────────────────────────────────────────────────
.PHONY: lint typecheck test test-integration eval
lint:
	ruff check semanrag/ tests/

typecheck:
	mypy semanrag/

test:
	pytest tests/

test-integration:
	pytest tests/ --run-integration

eval:
	python -m semanrag.eval

# ── Frontend ─────────────────────────────────────────────────────
.PHONY: frontend-install frontend-build frontend-dev
frontend-install:
	cd semanrag_webui && bun install

frontend-build:
	cd semanrag_webui && bun run build

frontend-dev:
	cd semanrag_webui && bun run dev

# ── Docker ───────────────────────────────────────────────────────
.PHONY: docker-build docker-up docker-down docker-full-up
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-full-up:
	docker compose -f docker-compose-full.yml up -d

# ── Cleanup ──────────────────────────────────────────────────────
.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/

# ── Help ─────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "SemanRAG — available targets:"
	@echo "  dev                Install with all extras"
	@echo "  env-base           Setup wizard: base config"
	@echo "  env-storage        Setup wizard: storage config"
	@echo "  env-server         Setup wizard: server config"
	@echo "  env-validate       Setup wizard: validate config"
	@echo "  env-security-check Setup wizard: security check"
	@echo "  eval               Run evaluation harness"
	@echo "  lint               Ruff linter"
	@echo "  typecheck          Mypy type checking"
	@echo "  test               Run unit tests"
	@echo "  test-integration   Run integration tests"
	@echo "  frontend-install   Install frontend deps"
	@echo "  frontend-build     Build frontend"
	@echo "  frontend-dev       Start frontend dev server"
	@echo "  docker-build       Build Docker image"
	@echo "  docker-up          Start containers"
	@echo "  docker-down        Stop containers"
	@echo "  docker-full-up     Start full stack"
	@echo "  clean              Remove caches and build artifacts"
