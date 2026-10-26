# Contributing to NegPy

## Claiming an Issue

Contributors with read access cannot use GitHub's assignee UI, so comment on the issue:

- `/assign`: assign the issue to yourself
- `/unassign`: remove yourself from the issue

A 👀 reaction on your comment confirms that the command ran.

## Development Setup

NegPy requires Python 3.13+, `uv` and a Rust toolchain.

### 1. Prerequisites
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [rustup](https://rustup.rs/). `rustup` reads the toolchain version from `negpy_core/rust-toolchain.toml`.

The `negpy_core/` folder is a Rust library with NegPy's CPU kernels. The first `uv sync` compiles it. After you edit a Rust file, the next `uv run` or `make test` compiles it again. `negpy_core` builds no C or C++ code: `cargo-deny` (`negpy_core/deny.toml`) rejects a dependency that does. Its tests compare against opencv, which only the `dev` group installs; `make install` (`uv sync --all-groups`) brings it.

Scanner support (optional):

- **SANE** (Linux/macOS): Coolscans and other SANE film scanners. Install the system library, then the `sane` group (`uv sync --group sane` or `pip install negpy[sane]`):
  - **Linux**:
    ```bash
    sudo pacman -S sane  # arch
    sudo apt install libsane-dev  # debian/ubuntu
    ```
  - **macOS**:
    ```bash
    brew install sane-backends
    ```

- **Plustek USB** (Windows/macOS/Linux): the [pyopticfilm](https://github.com/jboneng/pyopticfilm) driver (**1.3.8** or later): `uv sync --group plustek` or `pip install negpy[plustek]`. On Windows, pyopticfilm installs `libusb-package`, and the scanner needs WinUSB bound with Zadig in place of the vendor/SilverFast driver (`07b3:1825` 8200i SE, `07b3:1824` 8100 V2, `07b3:181f` 8300i SE, `07b3:0c3b` 7600i v1). See [docs/PLUSTEK_WINDOWS.md](docs/PLUSTEK_WINDOWS.md).

- **Nikon Coolscan** (Windows/macOS/Linux): [nkscan](https://github.com/activexray/nkscan) 0.14 or newer, shipped as wheels: `uv sync --group nkscan` or `pip install negpy[nkscan]`. On Linux, USB needs a udev rule for vendor `04b0`, and FireWire/SCSI needs the `sg` module.


### 2. Python Environment

```bash
make install
```


### 3. Running Locally

```bash
make run
```

#### A separate user directory for development

NegPy keeps its databases, caches, presets, logs and `override.toml` in one user
directory, `Documents/NegPy` by default. If you also use NegPy for real work, give your
development builds their own directory: set `NEGPY_USER_DIR` to an absolute path in an
optional, gitignored `.env.local`, which the `Makefile` reads:

```make
# .env.local
NEGPY_USER_DIR = $(HOME)/negpy-devhome
```

`make run` then uses (and creates) that directory.

Write `$(HOME)`, not `~` or `$HOME`: make leaves `~` alone and reads `$HOME` as `$H`
followed by `OME`. make stops when the value is not absolute. On macOS your home is under
`/Users/you`, not `/home/you/...`.

To reset it:

```bash
make clear-devhome
```

This deletes the whole directory after a confirmation (`FORCE=1` skips it). It refuses to
run if `NEGPY_USER_DIR` is unset or points at the default directory. Keep a copy of any
`override.toml` you rely on.

Two things stay outside the user directory:

- `.negpy` sidecars next to the source images. The next install that opens an image
  promotes its sidecar into its database. Sidecar export is off by default; if you turn it
  on, test against copies.
- Exports.

#### Running without scanning hardware

```bash
make run-sim
```

Starts the app with a simulated camera, Scanlight and film scanners
(`NEGPY_SIMULATE_HARDWARE=1`). Only the gphoto2 module, the serial port and one scanner
backend are fake.

- **Camera**: exposes frames from a synthetic negative under the Scanlight's color and the
  body's shutter and ISO, saved as Bayer DNGs. Put the calibration ROI on the clear base
  band around the picture.
- **Scanlight**: a v4b that answers the firmware query and reports LED temperature.
- **Film Scanner**: the "Simulated" backend has a feeder (frames by index, IR, Eject), a
  prescan device (full-window preview, then crop) and a roll device (frames found on the
  strip).

## Project Structure

- `negpy/domain/`: core data models, types and interfaces.
- `negpy/features/`: image processing logic (Exposure, Geometry, Lab, etc.).
- `negpy/infrastructure/`: low-level system code (GPU resources, file loaders).
- `negpy/kernel/`: core system services (Logging, Config, caching).
- `negpy/services/`: high-level orchestration (Rendering engine, Export service).
- `negpy/desktop/`: PyQt6 UI (View, Controller, Workers).
- `negpy_core/`: Rust library for CPU kernels (`src/<area>.rs`), with its PyO3 wrappers in `src/py/`.
- `tests/`: unit and integration tests.

## Coding Standards

Always run `make format` before committing.

### 1. Style & Formatting
- Ruff lints and formats.
- Type hints are required on every new function definition (`ty` is enforced). Do not use `cast` to get around it.
- Clear, short docstrings on classes and public methods.
- Double quotes for strings, snake_case for variables and functions, PascalCase for classes.

### 2. Testing
New features include unit tests (`pytest`) in `tests/`.

```bash
make test
```

`make test` skips tests marked `slow` (see `addopts` in `pyproject.toml`), including the performance metrics suite in `tests/metrics/`. To run it and write a JSON results file:

```bash
NEGPY_METRICS_OUT=metrics.json uv run pytest tests/metrics/ -m "slow" -q
```

The fixtures (Canon CR2, Nikon NEF, Sony ARW, Fuji RAF, Leica DNG) download from rawsamples.ch on the first run (~20–30 MB each) and are cached in `~/.cache/negpy-metrics/`. A test skips if its download fails. To use a local file, set the per-format env var:

```bash
NEGPY_PERF_RAW_CR2=/path/to/file.CR2 \
NEGPY_METRICS_OUT=metrics.json \
uv run pytest tests/metrics/ -m "slow" -q
```

Available overrides: `NEGPY_PERF_RAW_CR2`, `NEGPY_PERF_RAW_NEF`, `NEGPY_PERF_RAW_ARW`, `NEGPY_PERF_RAW_RAF`, `NEGPY_PERF_RAW_DNG`.

### 3. Workflow (The Makefile)
Every developer command runs through `uv run`:
- `make install`: set up and sync the environment.
- `make lint`: Ruff checks.
- `make type`: `ty` type checks.
- `make test`: unit tests.
- `make format`: auto-format with `cargo fmt` and Ruff.
- `make rust`: `cargo fmt --check`, `cargo clippy` (warnings fail) and `cargo test` for `negpy_core`.
- `make all`: format, Rust checks, lint, type and test in sequence.
- `make clean`: remove cache and build artifacts.
- `make run-sim`: run with simulated scanning hardware (see [above](#running-without-scanning-hardware)).
- `make clear-devhome`: delete the development user directory (see [above](#a-separate-user-directory-for-development)).


## Translations

Wrap display text in `tr("English source", name=value)` and translate the whole template before inserting values. Use `mark` for catalog labels that also serve as lookup keys, then call `tr` when displaying them. Keep enum values, settings keys, paths and user text unchanged. Language is an application preference and takes effect after a restart.

Register languages in `LANGUAGES` (`negpy/kernel/system/i18n.py`), using the matching Qt catalog locale, and set their count forms in `kernel/system/text.py`. `make translations` updates every registered JSON catalog under `negpy/translations/`. Fill new entries, then run `make translations-check` to check coverage and placeholders for all catalogs. Localized panel guides use the same `<!-- panel:key -->` markers as `docs/USER_GUIDE.md`, plus `src:<hash>` of the English section they translate; `tests/test_i18n.py` names each stale panel and its new marker. Missing panels fall back to English. Include new catalog and guide files in the packaged resources.

For translation updates after releases or upstream merges, use the
[update-translations skill](.claude/skills/update-translations/SKILL.md).

## Building and Packaging

To build the standalone application for your current OS with PyInstaller:

```bash
make build
```

On macOS, `NEGPY_MACOS_ARCH` sets the DMG's target architecture. For an Intel build:

```bash
NEGPY_MACOS_ARCH=x86_64 make build
```
