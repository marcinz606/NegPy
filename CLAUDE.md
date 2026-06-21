# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Keep this file current.** When a change alters something documented here — pipeline stage order, the feature pattern, the data model, build/packaging steps, or the dev commands — update the relevant section in the same change. After adding a pipeline stage, a feature, or a shader, re-check the "CPU pipeline" stage list and the "Adding a new feature" checklist.

## Commands

```bash
make run          # Launch the desktop app
make all          # lint + type check + tests (run before committing)
make test         # pytest only
make lint         # ruff check
make type         # ty check (not mypy)
make format       # ruff format + autofix

# Single test
uv run pytest tests/test_exposure_logic.py::test_name -v
```

All commands run through `uv run`; never invoke pytest/ruff/ty directly.

## Architecture

NegPy is a film negative processing desktop app (PyQt6 + WebGPU). Images flow through a multi-stage pipeline with a CPU and GPU path.

### Data model

`WorkspaceConfig` (`negpy/domain/models.py`) is a **frozen dataclass** composed of per-feature configs (`ExposureConfig`, `LabConfig`, etc.). It is the single source of truth for an edit. Changes are always made via `dataclasses.replace(config, ...)` — never mutated in place. `to_dict`/`from_flat_dict` handle serialization to/from a flat key namespace.

### Feature pattern

Every feature under `negpy/features/<name>/` follows this structure:

- `models.py` — frozen dataclass config with defaults
- `logic.py` — pure functions operating on numpy arrays
- `processor.py` — thin wrapper with a `process(img, context) -> ImageBuffer` method
- `shaders/<name>.wgsl` — optional GPU compute shader

### CPU pipeline (`negpy/services/rendering/engine.py`)

`DarkroomEngine.process()` runs stages in order: base (geometry + normalization) → exposure → retouch → lab → local (dodge/burn) → toning → crop → finish. The cached stages (base, exposure, retouch, lab, local) go through `_run_stage()`, which hashes the stage config and skips re-execution if the hash matches the cached `CacheEntry`; toning/crop/finish run unconditionally. Source image change clears the whole cache; a process-mode change invalidates only base/exposure/retouch/lab. Note `settings.geometry` drives both the early geometry transform and the late crop stage.

### GPU pipeline (`negpy/services/rendering/gpu_engine.py`)

`GPUEngine` runs the same logical pipeline as WGSL compute shaders via `wgpu`. It has its own change-detection logic (comparing previous vs. current `WorkspaceConfig` fields) to decide how far back to re-execute. Shader sources are loaded from `negpy/features/<name>/shaders/`.

### Orchestration (`negpy/services/rendering/image_processor.py`)

`ImageProcessor` chooses between GPU and CPU paths. GPU is tried first; on failure it falls back to CPU. Export always runs at full resolution through `GPUEngine.process()` or the CPU engine.

### `PipelineContext`

Passed through every stage. Carries `scale_factor` (preview downsample ratio), `process_mode`, `active_roi`, and a `metrics` dict for inter-stage data (`content_rect`, `uv_grid`, histogram bounds, etc.).

### Desktop (MVC)

- **`AppState`** (`negpy/desktop/session.py`) — mutable dataclass for session state (current file, active tool, last render metrics, GPU toggle, etc.)
- **`AppController`** (`negpy/desktop/controller.py`) — single controller; all UI interactions call methods here; emits `config_updated` and `image_updated` signals
- **Workers** — heavy work (render, export, thumbnails) runs in `QThread`-backed worker objects in `negpy/desktop/workers/`; communicate via Qt signals
- **Sidebars** — each feature has a `negpy/desktop/view/sidebar/<name>.py` that reads from `AppState` and calls controller methods; all synced via `ControlsPanel._sync_all_sidebars()` on `config_updated`

### Adding a new feature

1. Create `negpy/features/<name>/` with `models.py`, `logic.py`, `processor.py`
2. Add a field to `WorkspaceConfig` and update `to_dict` / `from_flat_dict`
3. Insert a `_run_stage(...)` call in `DarkroomEngine.process()`
4. For GPU: add a WGSL shader, wire it into `GPUEngine` (shader path + stage index/change-detection), and add the feature's `shaders/` dir to `build.py` (`--add-data`) so it ships in the packaged app
5. Add a sidebar and register it in `ControlsPanel`
