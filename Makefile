# Variables
VENV = venv
PYTHON = $(VENV)/bin/python
PYTEST = $(PYTHON) -m pytest
MYPY = $(PYTHON) -m mypy
FLAKE8 = $(PYTHON) -m flake8
RUFF = $(PYTHON) -m ruff

# Default target
.PHONY: all
all: lint type test

# Style checks (flake8)
.PHONY: lint
lint:
	@echo "Running style checks (flake8)..."
	@$(FLAKE8) .

# Type checks (mypy)
.PHONY: type
type:
	@echo "Running type checks (mypy)..."
	@$(MYPY) .

# Unit tests (pytest)
.PHONY: test
test:
	@echo "Running unit tests (pytest)..."
	@$(PYTEST) tests/

# Auto-format and fix (ruff)
.PHONY: format
format:
	@echo "Running ruff format and fix..."
	@$(RUFF) format .
	@$(RUFF) check --fix .

# Run the application (Docker)
.PHONY: run-app
run-app:
	@echo "Starting DarkroomPy via Docker..."
	@$(PYTHON) start.py

.PHONY: run-app-rebuild
run-app-rebuild:
	@echo "Rebuilding and starting DarkroomPy via Docker..."
	@$(PYTHON) start.py --build

# Build Electron application (Host OS)
.PHONY: dist
dist:
	@echo "Building Electron application for host OS..."
	@PATH=$(CURDIR)/$(VENV)/bin:$(PATH) npm run dist

.PHONY: dist-win
dist-win:
	@echo "Building Electron application for Windows..."
	@echo "Note: This must be run on Windows to correctly build the Python backend."
	@PATH=$(CURDIR)/$(VENV)/bin:$(PATH) npm run dist:win

.PHONY: dist-mac
dist-mac:
	@echo "Building Electron application for macOS..."
	@echo "Note: This must be run on macOS to correctly build the Python backend."
	@PATH=$(CURDIR)/$(VENV)/bin:$(PATH) npm run dist:mac

.PHONY: dist-linux
dist-linux:
	@echo "Building Electron application for Linux..."
	@PATH=$(CURDIR)/$(VENV)/bin:$(PATH) npm run dist:linux

# Clean up caches
.PHONY: clean
clean:
	@echo "Cleaning up caches..."
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
