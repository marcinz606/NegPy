# Variables
VENV = .venv
PYTHON = $(VENV)/bin/python
PYTEST = $(PYTHON) -m pytest
MYPY = $(PYTHON) -m mypy
FLAKE8 = $(PYTHON) -m flake8
RUFF = $(PYTHON) -m ruff

# Versioning
VERSION_FILE = VERSION

# Macro to update version files
define update_version
	@if [ ! -z "$(VERSION)" ]; then \
		CURRENT_VER=$$(node -p "require('./package.json').version"); \
		if [ "$$CURRENT_VER" != "$(VERSION)" ]; then \
			echo "Updating version from $$CURRENT_VER to $(VERSION)..."; \
			npm version $(VERSION) --no-git-tag-version; \
		else \
			echo "Version is already $(VERSION), skipping npm update."; \
		fi; \
		echo "$(VERSION)" | sed 's/^v//' > $(VERSION_FILE); \
	else \
		echo "Syncing $(VERSION_FILE) from package.json..."; \
		node -p "require('./package.json').version" > $(VERSION_FILE); \
	fi
endef

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
	@echo "Starting NegPy via Docker..."
	@$(PYTHON) start.py

.PHONY: run-app-rebuild
run-app-rebuild:
	@echo "Rebuilding and starting NegPy via Docker..."
	@$(PYTHON) start.py --build

# Build Electron application (Host OS)
.PHONY: dist
dist:
	@$(call update_version)
	@echo "Building Electron application for host OS..."
	@start=$$(date +%s); \
	PATH=$(CURDIR)/$(VENV)/bin:$(PATH) npm run dist; \
	end=$$(date +%s); \
	echo "Build took $$(($$end - $$start)) seconds"

.PHONY: dist-win
dist-win:
	@$(call update_version)
	@echo "Building Electron application for Windows..."
	@echo "Note: This must be run on Windows to correctly build the Python backend."
	@start=$$(date +%s); \
	PATH=$(CURDIR)/$(VENV)/bin:$(PATH) npm run dist:win; \
	end=$$(date +%s); \
	echo "Build took $$(($$end - $$start)) seconds"

.PHONY: dist-mac
dist-mac:
	@$(call update_version)
	@echo "Building Electron application for macOS..."
	@echo "Note: This must be run on macOS to correctly build the Python backend."
	@start=$$(date +%s); \
	PATH=$(CURDIR)/$(VENV)/bin:$(PATH) npm run dist:mac; \
	end=$$(date +%s); \
	echo "Build took $$(($$end - $$start)) seconds"

.PHONY: dist-linux
dist-linux:
	@$(call update_version)
	@echo "Building Electron application for Linux..."
	@start=$$(date +%s); \
	PATH=$(CURDIR)/$(VENV)/bin:$(PATH) npm run dist:linux; \
	end=$$(date +%s); \
	echo "Build took $$(($$end - $$start)) seconds"

# Clean up caches
.PHONY: clean
clean:
	@echo "Cleaning up caches..."
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
