# CLAUDE.md

Guidance for Claude Code in this repository.

> **Keep this file current.** When a change alters something documented here — stage order, the feature pattern, commands — update it in the same change.

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

NegPy is a film-negative processing desktop app (PyQt6 + WebGPU). Images flow through a multi-stage pipeline implemented twice — CPU (numpy/Numba) and GPU (WGSL via `wgpu`) — which must stay in numerical parity.

### Data model

`WorkspaceConfig` (`negpy/domain/models.py`) — frozen dataclass composed of per-feature configs; the single source of truth for an edit. Change via `dataclasses.replace`, never mutate. `to_dict`/`from_flat_dict` serialize to one **flat** key namespace — a duplicate field name across sub-configs silently clobbers.

Edits persist in SQLite (`edits.db`, keyed by content hash), optionally mirrored to `.negpy` JSON sidecars next to sources. DB wins; a loaded sidecar is promoted into the DB (`negpy/services/assets/sidecar.py`, `session.py`).

### Pipeline

- **CPU**: `DarkroomEngine.process()` (`negpy/services/rendering/engine.py`) — base (geometry + normalization) → exposure (incl. dodge/burn) → clahe → retouch → lab → toning → crop → finish. The first five stages are cached per config-hash via `_run_stage()`; the rest run unconditionally.
- **GPU**: `GPUEngine` (`negpy/services/rendering/gpu_engine.py`) — same logical stages as WGSL compute shaders from `negpy/features/<name>/shaders/`, with its own config-diff change detection.
- **Orchestration**: `ImageProcessor` (`image_processor.py`) tries GPU first, falls back to CPU; export always runs full-res. `PipelineContext` carries `scale_factor`, `process_mode`, `active_roi`, and a `metrics` dict between stages.
- **Working space**: scene-linear internally; the working OETF (Adobe RGB 1998 TRC — a pure 563/256 power, no linear segment) is applied only as the final engine step. Lab/toning compute CIELAB directly from linear, D65. Adobe RGB rather than a wide gamut because ProPhoto's imaginary primaries inflate chroma in the saturation/toning stages.

`docs/PIPELINE.md` describes each stage's behaviour and controls in depth.

### Feature pattern

Every feature lives in `negpy/features/<name>/`:

- `models.py` — frozen dataclass config with defaults
- `logic.py` — pure functions on numpy arrays
- `processor.py` — thin wrapper with `process(img, context) -> ImageBuffer`
- `shaders/<name>.wgsl` — optional GPU compute shader

### Desktop (MVC)

- `AppState` (`negpy/desktop/session.py`) — mutable session state
- `AppController` (`negpy/desktop/controller.py`) — single controller; all UI interactions call it; emits `config_updated` / `image_updated`
- Workers (`negpy/desktop/workers/`) — heavy work in QThread-backed objects, Qt-signal communication
- Sidebars (`negpy/desktop/view/sidebar/<name>.py`) — one per feature, registered in `ControlsPanel`, synced on `config_updated`

## Adding a new feature

1. Create `negpy/features/<name>/` with `models.py`, `logic.py`, `processor.py`
2. Add a field to `WorkspaceConfig`; update `to_dict`/`from_flat_dict` (watch flat-namespace collisions)
3. Insert a `_run_stage(...)` call in `DarkroomEngine.process()`
4. For GPU: add a WGSL shader, wire it into `GPUEngine` (shader path + stage index + change detection), and add the feature's `shaders/` dir to `build.py` (`--add-data`)
5. Add a sidebar and register it in `ControlsPanel`
6. Add unit tests; if the feature has both CPU and GPU paths, add a parity test (pattern: `test_gpu_curve_parity.py`)

## Style

- **Comments minimal.** Comment only non-obvious constraints the code can't express (a cache contract, an ordering requirement, a rejected-alternative trap). No comments that narrate what the next line does, restate the diff, or justify a change to a reviewer. Prefer one dense line over a paragraph; docstrings short and factual.

## Invariants & gotchas

- **CPU/GPU parity**: any change to a stage's math must land in both `logic.py` and its `.wgsl` shader. Constants mirrored as WGSL literals (histogram bins, zone density, metrics offsets) have parity tests — keep them in sync.
- **Working-space OETF + luminance row are inlined** in `lab_sharpen_h.wgsl` and `rl_init.wgsl` (Adobe RGB 1998 gamma 563/256, D65 Y row) — a TRC or primaries change must update them, not just `kernel/image/logic.py`. `LabUniforms` is declared in 7 lab shaders (lab, lab_sharpen_h/v, rl_init/blur_h/div_v/mult_v) — any field change touches all.
