# Variables
UV = uv run

# Optional per-developer settings, not in git. The usual content is NEGPY_USER_DIR,
# which moves the app's user directory (databases, caches, presets, logs) away from
# the production one. Absent, everything behaves as before.
-include .env.local
export NEGPY_USER_DIR

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

# Delete the development user directory, so the next run starts with no saved edits
# and no caches. Refuses to run against the default directory: that is the real
# install. Set NEGPY_USER_DIR (see .env.local) to a scratch path first.
# FORCE=1 skips the confirmation.
.PHONY: clear-devhome
clear-devhome:
	@dir="$$NEGPY_USER_DIR"; \
	if [ -z "$$dir" ]; then \
		echo "NEGPY_USER_DIR is not set — nothing to clear."; \
		echo "Set it in .env.local to a scratch directory, then run this again."; \
		exit 1; \
	fi; \
	default=$$($(UV) python -c "import os; os.environ.pop('NEGPY_USER_DIR', None); from negpy.kernel.system.paths import get_default_user_dir; print(get_default_user_dir())"); \
	if [ "$$(cd "$$dir" 2>/dev/null && pwd)" = "$$(cd "$$default" 2>/dev/null && pwd)" ]; then \
		echo "Refusing to clear $$dir — that is the default user directory."; \
		exit 1; \
	fi; \
	if [ ! -d "$$dir" ]; then echo "$$dir does not exist — nothing to clear."; exit 0; fi; \
	if [ "$(FORCE)" != "1" ]; then \
		printf "Delete %s and everything in it? [y/N] " "$$dir"; \
		read reply; \
		case "$$reply" in [yY]*) ;; *) echo "Cancelled."; exit 1;; esac; \
	fi; \
	rm -rf "$$dir"; \
	echo "Cleared $$dir"

# Run against a locally built sane-backends from rohanpandula's fork: coolscan3
# with infrared un-gated (stock compiles it out; LS-50 reports no IR channel),
# the frame advance corrected from the LS-30-era 1.5 in guess (38.106 mm) to the
# true 135 pitch of 38.0 mm (8 x 4.75 mm perforations — halves per-frame drift,
# A/B-measured -0.277 -> -0.143 mm/frame on an LS-50), plus load/eject parameter
# zeroing. Build: make sane-rgbi-help
SANE_RGBI ?= $(HOME)/.local/share/negpy-sane-38mm

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
	@echo "Build the IR-enabled, 38 mm-pitch coolscan3 backend (nothing system-wide is touched):"
	@echo ""
	@echo "  git clone --depth 30 --branch coolscan3-ls5000-38mm-frame-pitch \\"
	@echo "      https://gitlab.com/rohanpandula/backends.git && cd backends"
	@echo "  # the pitch fix is gated CS3_TYPE_LS5000 only — on an LS-50 widen the"
	@echo "  # gate in backend/coolscan3.c (frame_offset assignment):"
	@echo "  #     if (s->type == CS3_TYPE_LS5000 || s->type == CS3_TYPE_LS50)"
	@echo "  # if ./configure errors on AX_* macros, install autoconf-archive (or drop"
	@echo "  # ax_create_stdint_h.m4 + ax_cxx_compile_stdcxx*.m4 into m4/), rerun autogen.sh"
	@echo "  ./autogen.sh"
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
