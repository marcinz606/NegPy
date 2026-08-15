# Contributing to NegPy

Thank you for your interest in contributing to **NegPy**!

## 🙋 Claiming an Issue

Contributors with read access can't use GitHub's assignee UI, so NegPy lets you
self-assign by commenting on an issue:

- `/assign` — assign the issue to yourself
- `/unassign` — remove yourself from the issue

A 👀 reaction on your comment confirms the command ran.

## 🛠️ Development Setup

NegPy requires **Python 3.13+**. We use **uv** for environment and dependency management.

### 1. Prerequisites
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

**Scanner support (optional):**

- **SANE** (Linux/macOS) — Coolscans and other SANE film scanners. Install the system library, then the `sane` group (`uv sync --group sane` or `pip install negpy[sane]`):
  - **Linux** (Debian/Ubuntu):
    ```bash
    sudo pacman -S sane  # arch
    sudo apt install libsane-dev  # debian/ubuntu
    ```
  - **macOS**:
    ```bash
    brew install sane-backends
    ```

- **Plustek USB** (Windows/macOS/Linux) — optional [pyopticfilm](https://github.com/jboneng/pyopticfilm) driver for OpticFilm 8200i SE (`uv sync --group plustek` or `pip install negpy[plustek]`). Windows installs `libusb-package` via pyopticfilm (bundled in release builds). On Windows, bind WinUSB with Zadig for USB id `07b3:1825` before scanning (vendor/SilverFast drivers conflict). See [docs/PLUSTEK_WINDOWS.md](docs/PLUSTEK_WINDOWS.md).


### 2. Python Environment
The `Makefile` handles synchronization via `uv`. Run this to set up your environment:

```bash
make install
```


### 3. Running Locally

```bash
make run
```

#### A separate user directory for development

NegPy keeps its databases, caches, presets, logs and `override.toml` in one user
directory — `Documents/NegPy` by default. If you also use NegPy for real work, point
your development builds at a different directory, so that test edits and stale caches
cannot touch the real one.

Set `NEGPY_USER_DIR` to an absolute path. The `Makefile` reads an optional, gitignored
`.env.local`:

```make
# .env.local
NEGPY_USER_DIR = $(HOME)/negpy-devhome
```

`make run` then uses that directory, and creates it if it is missing. `~` is not
expanded — use `$(HOME)` or a full path. Without the file, nothing changes.

To start again with no saved edits and no caches:

```bash
make clear-devhome
```

The target deletes the whole directory after a confirmation (`FORCE=1` skips it), and
refuses to run if `NEGPY_USER_DIR` is unset or points at the default directory. An
`override.toml` you rely on lives in there too, so keep a copy to put back.

Two things stay outside the user directory:

- `.negpy` sidecars, which are written next to the source images. A sidecar written by a
  development build is promoted into the database of whichever install opens that image
  next. Sidecar export is off by default; if you turn it on, test against copies.
- Exports, wherever you send them.

## 🏗️ Project Structure

The codebase follows a modular architecture:

- `negpy/domain/`: Core data models, types, and interfaces.
- `negpy/features/`: Image processing logic implementations (Exposure, Geometry, Lab, etc.).
- `negpy/infrastructure/`: Low-level system implementations (GPU resources, file loaders).
- `negpy/kernel/`: Core system services (Logging, Config, caching).
- `negpy/services/`: High-level orchestration (Rendering engine, Export service).
- `negpy/desktop/`: PyQt6 UI implementation (View, Controller, Workers).
- `tests/`: Unit and integration tests.

## 📐 Coding Standards

**Always run `make format` before committing.**

### 1. Style & Formatting
- **Ruff**: Used for both linting and formatting.
- **Type Hints**: Required for all new function definitions (`ty` is enforced). Using `cast` to get around it is frowned upon.
- **Docstrings**: Use clear, concise docstrings for classes and public methods.
- **Style**: Use double quotes for strings, snake_case for variables and functions, and PascalCase for classes.

### 2. Testing
We use `pytest`. New features should include unit tests in the `tests/` directory.

```bash
make test
```

`make test` skips tests marked `slow` by default (see `addopts` in `pyproject.toml`). This includes the performance metrics suite in `tests/metrics/`.

To run the metrics tests and write a JSON results file:

```bash
NEGPY_METRICS_OUT=metrics.json uv run pytest tests/metrics/ -m "slow" -q
```

Fixtures (Canon CR2, Nikon NEF, Sony ARW, Fuji RAF, Leica DNG) are downloaded automatically from rawsamples.ch on first run (~20–30 MB each) and cached in `~/.cache/negpy-metrics/`. Tests skip gracefully if a download fails.

To point a test at a local file instead of downloading, set the per-format env var:

```bash
NEGPY_PERF_RAW_CR2=/path/to/file.CR2 \
NEGPY_METRICS_OUT=metrics.json \
uv run pytest tests/metrics/ -m "slow" -q
```

Available overrides: `NEGPY_PERF_RAW_CR2`, `NEGPY_PERF_RAW_NEF`, `NEGPY_PERF_RAW_ARW`, `NEGPY_PERF_RAW_RAF`, `NEGPY_PERF_RAW_DNG`.

### 3. Workflow (The Makefile)
The `Makefile` is the central source of truth for developer commands and executes everything via `uv run`:
- `make install`: Set up environment and sync dependencies.
- `make lint`: Run Ruff checks.
- `make type`: Run `ty` type checks.
- `make test`: Run all unit tests.
- `make format`: Auto-format code with Ruff.
- `make all`: Run lint, type, and test in sequence.
- `make clean`: Removes cache and build artifacts.
- `make clear-devhome`: Deletes the development user directory (see [above](#a-separate-user-directory-for-development)).


## 📦 Building and Packaging

To build the standalone application for your current OS:

```bash
make build
```
This will trigger the Python backend build via PyInstaller.

On macOS, you can choose the target architecture for the DMG build with `NEGPY_MACOS_ARCH`.
For an Intel build from a compatible macOS environment, run:

```bash
NEGPY_MACOS_ARCH=x86_64 make build
```
