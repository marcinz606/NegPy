# CLAUDE.md

Guidance for Claude Code in this repository.

> **Keep this file current.** When a change alters something documented here — stage order, the feature pattern, commands, an invariant — update it in the same change.

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

## Invariants & gotchas

- **CPU/GPU parity**: any change to a stage's math must land in both `logic.py` and its `.wgsl` shader. Constants mirrored as WGSL literals (histogram bins, zone density, metrics offsets) have parity tests — keep them in sync.
- **Cast Removal neutral-axis** is measured against the **pre-trim** resolved bounds on both paths (`exposure/processor.py` `pre_trim_bounds`, `gpu_engine.py` measures before `adj_floors`) — the film's inherent cast is a source property, so per-channel WP/BP trims must not perturb which pixels are picked as neutral. Consumption then normalizes the refs against the *adjusted* bounds. Using adjusted bounds for the measurement diverges CPU↔GPU only when trims are non-zero; `test_gpu_curve_parity.py::test_cpu_gpu_match_trims_no_dye_mute` guards it (Dye Mute, `LabConfig.chroma_damping`, otherwise masks it by scaling chroma toward neutral).
- **Print H&D curve** has four lock-step mirrors: the numba kernel (`exposure/logic.py`), the `CharacteristicCurve` chart, `exposure.wgsl`, and the uniform pack in `gpu_engine.py`. Change all together; growing a GPU uniform struct is a `_uniform_sizes` bump + pack/WGSL row.
- **Numba kernels**: decorate with `parallel_njit` (`kernel/system/parallel.py`), never raw `@njit(parallel=True)`. Per-row scratch arrays go inside the `prange` loop (thread-local); never pass `cache=True` (see `test_parallel_dispatch.py` for why).
- **Working TRC** is hardcoded in **seven** places: CPU `working_oetf_*` (`kernel/image/logic.py`, the definition) and WGSL `oetf_*` in `exposure.wgsl`, `output_encode.wgsl`, `lab.wgsl`, `clahe_apply.wgsl`, `clahe_hist.wgsl`, `metrics.wgsl`. (`charts.py` calls `working_oetf_encode` and follows automatically — keep it that way.) **Primaries/white-point** changes touch **five**: the CPU XYZ matrices plus the inline matrices in `lab.wgsl`, `clahe_apply.wgsl`, `toning.wgsl`, and the Y-row-only copy in `clahe_hist.wgsl`. Every WGSL matrix is a hand-duplicated literal; `test_pipeline_parity.py` is what catches a missed copy.
- **Coordinate spaces**: manual crop rect and analysis rect are normalized to the *transformed* (display) image; retouch strokes and dodge/burn placements are *source*-normalized and round-trip through `create_uv_grid`/`map_coords_to_geometry` — which must stay consistent with the image warp (incl. distortion k1).
- **Metrics buffer** (`_METRICS_*` offsets, `gpu_engine.py`) is append-only; offsets are mirrored as WGSL array lengths.
- **Camera capture** (`infrastructure/capture/gphoto.py`) has load-bearing libgphoto2 guards (a NULL-choice read SIGSEGVs the process; writes are async; the event queue must be drained after a still). Read that module before touching it.

## More detail

`docs/PIPELINE.md` (stage-by-stage behaviour), `docs/CAMERA_SCANNING.md`, `docs/CONTACT_SHEET_TEMPLATES.md`, `docs/CROSSTALK.md`. Deep design rationale for past decisions lives in git history (`git log -p CLAUDE.md`).
