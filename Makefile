# Variables
UV = uv run

# Default target
.PHONY: all
all: format lint type test

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

# Type checks (ty)
.PHONY: type
type:
	@echo "Running type checks (ty)..."
	@$(UV) ty check \
		--exclude "tests/" --exclude "docs/" --exclude "build/" --exclude "dist/" --exclude ".venv/" \
		--ignore "no-matching-overload" \
		--ignore "unresolved-attribute" \
		--ignore "invalid-method-override" \
		--ignore "not-iterable" \
		--ignore "unsupported-operator" \
		--ignore "invalid-argument-type" \
		--ignore "unused-type-ignore-comment" \
		--ignore "unresolved-import" \
		--ignore "unsupported-bool-conversion" \
		--ignore "invalid-assignment" \
		--ignore "invalid-parameter-default" \
		--ignore "call-non-callable"

# Unit tests (pytest)
.PHONY: test
test:
	@echo "Running unit tests (pytest)..."
	@$(UV) pytest tests/ --cov=negpy --cov-report=term-missing

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

# Run against a locally built sane-backends whose coolscan3 has its infrared
# option un-gated. Stock sane-backends compiles that option out for every model
# (sane.h keeps SANE_FRAME_RGBI inside an "#if 0", and coolscan3.c marks infrared
# SANE_CAP_INACTIVE unless it is defined), so scanners with working IR hardware
# (LS-50 / Coolscan V, Digital ICE) report no IR channel. Deleting that one cap
# line is enough: the frame stays SANE_FRAME_RGB while n_colors becomes 4, the
# 4-sample convention _reinterpret_channels already handles. Build: make sane-rgbi-help
SANE_RGBI ?= $(HOME)/.local/share/negpy-sane-rgbi

.PHONY: run-ir
run-ir:
	@test -d "$(SANE_RGBI)/lib/sane" || { \
		echo "No patched sane-backends at $(SANE_RGBI)"; \
		echo "Run 'make sane-rgbi-help' for how to build it, or set SANE_RGBI=<prefix>."; \
		exit 1; }
	@echo "Starting NegPy Desktop with IR-enabled coolscan3 ($(SANE_RGBI))..."
	@LD_LIBRARY_PATH="$(SANE_RGBI)/lib" SANE_CONFIG_DIR="$(SANE_RGBI)/etc/sane.d" $(UV) python desktop.py

.PHONY: sane-rgbi-help
sane-rgbi-help:
	@echo "Build the IR-enabled coolscan3 backend (nothing system-wide is touched):"
	@echo ""
	@echo "  curl -L -o backends.tar.gz https://gitlab.com/sane-project/backends/-/archive/1.4.0/backends-1.4.0.tar.gz"
	@echo "  tar xzf backends.tar.gz && cd backends-1.4.0 && ./autogen.sh"
	@echo "  # in backend/coolscan3.c, delete these 3 lines from CS3_OPTION_INFRARED:"
	@echo "  #     #ifndef SANE_FRAME_RGBI"
	@echo "  #             o.cap |= SANE_CAP_INACTIVE;"
	@echo "  #     #endif"
	@echo "  # (do NOT instead #define SANE_FRAME_RGBI: the backend then reports"
	@echo "  #  frame format 0x10, which python-sane refuses as 'Invalid frame format')"
	@echo "  ./configure --prefix=$(SANE_RGBI) BACKENDS=coolscan3 --disable-translations --disable-avahi"
	@echo "  make -j\$$(nproc) && make install"
	@echo ""
	@echo "Then: make run-ir     (verify with: caps.ir_channel == True)"

# Build the application
.PHONY: build
build:
	@echo "Building NegPy..."
	rm -rf dist/
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
