# Plan: Add CLI Batch Conversion to NegPy

## Context

NegPy is a film negative-to-positive converter that currently only works through a PyQt6 GUI. The user wants to convert multiple negatives at once from the command line. The architecture already separates processing logic from the GUI cleanly — `ImageProcessor.process_export()` handles the full pipeline (file loading, demosaic, darkroom processing, color management, encoding) and returns raw bytes. The GUI's `ExportWorker` just iterates files and writes bytes to disk. The CLI replicates only this thin orchestration layer.

## Files to Create/Modify

| Action | File | Purpose |
|--------|------|---------|
| CREATE | `negpy/cli/__init__.py` | Empty package init |
| CREATE | `negpy/cli/batch.py` | CLI implementation (~180 lines) |
| CREATE | `tests/test_cli_batch.py` | Tests (~200 lines) |
| MODIFY | `pyproject.toml` | Console script entry point + package discovery |

## Implementation

### 1. `negpy/cli/__init__.py` — Empty file

### 2. `negpy/cli/batch.py` — CLI module

**Structure:**
- Set `NUMBA_THREADING_LAYER=workqueue` env var before any negpy imports (required for numba JIT)
- Mapping dicts: `MODE_MAP` (c41/bw/e6 -> `ProcessMode`), `FORMAT_MAP`, `COLOR_SPACE_MAP`
- `build_parser()` — argparse with these arguments:
  - `inputs` (positional, nargs=+): files or directories
  - `--mode c41|bw|e6` (default: c41)
  - `--format jpeg|tiff` (default: tiff — 16-bit for maximum quality, stored as `output_format`)
  - `--output DIR` (default: ./export)
  - `--color-space` choices from `ColorSpace` enum (default: adobe-rgb)
  - `--density FLOAT`, `--grade FLOAT`, `--sharpen FLOAT` (default: None = don't override base)
  - `--dpi INT`, `--print-size CM`, `--original-res` flag
  - `--filename-pattern TEMPLATE`
  - `--no-gpu` flag
  - `--settings JSON_FILE` — load full WorkspaceConfig from JSON
- `discover_files(inputs)` — resolve paths to sorted list of supported files using `SUPPORTED_RAW_EXTENSIONS` (which includes TIFF/JPEG). Walk directories recursively.
- `build_config(args)` — build `WorkspaceConfig`:
  1. Start from `DEFAULT_WORKSPACE_CONFIG` or `WorkspaceConfig.from_flat_dict(json)` if `--settings` given
  2. Override sub-configs with `dataclasses.replace()` for any CLI arg that is not None
  3. Always override: `process_mode`, `export_path`, `export_fmt`, `export_color_space`
- `main(argv=None) -> int` — main loop:
  1. Parse args, toggle `APP_CONFIG.use_gpu` if `--no-gpu` (before constructing `ImageProcessor`)
  2. Discover files, build config
  3. Construct `ImageProcessor()`
  4. For each file: `calculate_file_hash()`, call `process_export()`, write bytes, call `processor.cleanup()`
  5. Print progress to stderr, timing per file
  6. Call `processor.destroy_all()` at end
  7. Return 0 if all succeeded, 1 if any failed
- `cli_entry()` — calls `sys.exit(main())`

**Key reused components:**
- `ImageProcessor.process_export()` — `/home/user/negpy/negpy/services/rendering/image_processor.py:122`
- `render_export_filename()` — `/home/user/negpy/negpy/services/export/templating.py:8`
- `calculate_file_hash()` — `/home/user/negpy/negpy/kernel/image/logic.py:221`
- `DEFAULT_WORKSPACE_CONFIG` — `/home/user/negpy/negpy/kernel/system/config.py:35`
- `SUPPORTED_RAW_EXTENSIONS` — `/home/user/negpy/negpy/infrastructure/loaders/constants.py:13`
- `ExportWorker.run_batch()` pattern — `/home/user/negpy/negpy/desktop/workers/export.py:35` (reference for cleanup/teardown pattern)

### 3. `pyproject.toml` changes

- Change `[tool.setuptools] packages = ["negpy"]` to use find-based discovery:
  ```toml
  [tool.setuptools.packages.find]
  where = ["."]
  include = ["negpy*"]
  ```
- Add console script:
  ```toml
  [project.scripts]
  negpy = "negpy.cli.batch:cli_entry"
  ```

### 4. `tests/test_cli_batch.py`

- `TestBuildParser`: minimal args, all flags, multiple inputs, missing inputs raises, invalid mode raises
- `TestDiscoverFiles`: single supported file, unsupported skipped, directory walk, missing path warns, mixed files+dirs
- `TestBuildConfig`: defaults match, mode/exposure/lab overrides, export format+colorspace, dpi/print-size/original-res, filename pattern, settings JSON load, settings + CLI override, missing/invalid settings file
- `TestMain`: mock `ImageProcessor` to avoid real processing; test no-files-returns-1, successful batch, partial failure, output files written, no-gpu flag passes prefer_gpu=False

## Verification

1. Run `pytest tests/test_cli_batch.py -v` — all tests should pass
2. Run `python -m negpy.cli.batch --help` — should display help with all arguments
3. Run full test suite `pytest` — ensure no regressions
