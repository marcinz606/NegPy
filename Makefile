# Variables
UV = uv run

# Default target
.PHONY: all
all: lint type test

# Install dependencies
.PHONY: install
install:
	@echo "Installing dependencies with uv..."
	@uv sync --all-groups

# Sync dependencies
.PHONY: sync
sync:
	@echo "Syncing dependencies with uv..."
	@uv sync --all-groups

# Lint checks (ruff)
.PHONY: lint
lint:
	@echo "Running lint checks (ruff)..."
	@$(UV) ruff check .

# Type checks (mypy)
.PHONY: type
type:
	@echo "Running type checks (mypy)..."
	@$(UV) mypy .

# Unit tests (pytest)
.PHONY: test
test:
	@echo "Running unit tests (pytest)..."
	@$(UV) pytest tests/

# Auto-format and fix (ruff)
.PHONY: format
format:
	@echo "Running ruff format and fix..."
	@$(UV) ruff format .
	@$(UV) ruff check --fix .

# Run the application locally
.PHONY: run
run:
	@echo "Starting NegPy Desktop..."
	@$(UV) python desktop.py

# Build the application
.PHONY: build
build:
	@echo "Building NegPy..."
	@$(UV) python build.py

# Clean up caches and build artifacts
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf build
	rm -rf dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
