# ──────────────────────────────────────────────
# llm-cli-py  Makefile
# ──────────────────────────────────────────────

.PHONY: help format check test clean check-all

.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

format:  ## Run ruff format (auto-format code)
	ruff format

check:   ## Run ruff check (linter)
	ruff check

test:   ## Run pytest
	pytest -v

install: ## Run pip install -e . (CLI only)
	pip install -e .

install-dev: ## Run pip install -e .[dev] (CLI + dev tools)
	pip install -e ".[dev]"

install-all: ## Run pip install -e .[dev] (everything)
	pip install -e ".[dev]"

check-all: format check test  ## Run all checks: format → lint → test

clean:  ## Remove all intermediate artifacts (caches, builds, egg-info)
	@echo "Removing __pycache__ directories..."
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	@echo "Removing tool caches..."
	rm -rf .pytest_cache .ruff_cache
	@echo "Removing build artifacts..."
	rm -rf dist/ build/
	@echo "Removing egg-info..."
	rm -rf src/*.egg-info/
	@echo "Done."
