.DEFAULT_GOAL := all

.PHONY: .uv
.uv:
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit
.pre-commit:
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install
install: .uv .pre-commit
	uv sync --frozen
	#pre-commit install --install-hooks

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix --fix-only

.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

.PHONY: typecheck
typecheck:
	PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run pyright

.PHONY: dev
dev:
	uv run uvicorn app:app --reload

.PHONY: all
all: format lint typecheck
