# CLAUDE.md

Guidance for Claude Code in this repository.

> **Keep this file current.** When a change alters something documented here — stage order, the feature pattern, commands — update it in the same change.

> **Keep the user docs current, in the same change.** `docs/USER_GUIDE.md` covers every panel and control; it is also **rendered in-app** — each panel's ⓘ opens the section marked `<!-- panel:<key> -->` — so a stale doc is a stale in-app guide. Any control added, renamed, retired or given a new range/default belongs there. `docs/PIPELINE.md` covers what each stage does to the pixels: update it when stage order, the math, a mirrored constant or a default changes. Retiring a control means deleting its prose, not leaving it to rot.

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
Before commiting always run `make format`. If any non-related files got re-formatted by ruff - also commit the changes as lint fixes.

## Architecture

NegPy is a film-negative processing desktop app (PyQt6 + WebGPU). Images flow through a multi-stage pipeline implemented twice — CPU (numpy/Numba) and GPU (WGSL via `wgpu`) — which must stay in numerical parity.

### Data model

`WorkspaceConfig` (`negpy/domain/models.py`) — frozen dataclass composed of per-feature configs; the single source of truth for an edit. Change via `dataclasses.replace`, never mutate. `to_dict`/`from_flat_dict` serialize to one **flat** key namespace — a duplicate field name across sub-configs silently clobbers.

Edits persist in SQLite (`edits.db`, keyed by content hash), optionally mirrored to `.negpy` JSON sidecars next to sources. DB wins; a loaded sidecar is promoted into the DB (`negpy/services/assets/sidecar.py`, `session.py`).

**Migrations** (`negpy/domain/migrations.py`) — every legacy fixup for persisted configs lives here, not inline in `from_flat_dict`: `KEY_RENAMES` (renamed fields), `DROPPED_KEYS` (removed fields, dropped without the unknown-key warning), `RETIRED_EXPORT_FORMATS`, and `migrate_flat_config()` for value rewrites. Renaming/removing a config field or retiring an enum value means one entry here. Two exceptions stay in their dataclasses because they must run on *every* construction, not just on load: `ExposureConfig.__post_init__` (legacy grade → ISO R, `cast_removal` bool → strength) and the tuple-rehydrating `__post_init__`s. The module imports nothing from `models.py` (which imports it) — use string literals.

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
- **Shortcuts** (`negpy/desktop/view/shortcut_registry.py`) — `REGISTRY` is the single source of truth for every binding: one `ShortcutEntry(default_key, description, category)` per action id. Dispatch is the matching entry in the action map in `keyboard_shortcuts.py`. The registry also feeds the shortcut editor, the `?` overlay and `tooltip_with_shortcut()`, so a binding added here shows up in all three for free.
  **Any new user-facing toggle, tool or action gets a registry entry** — leave `default_key` empty rather than inventing a conflicting one if no obvious key is free. Check for collisions before picking: the same key on two actions makes Qt fire `activatedAmbiguously` and both go dead.

## Adding a new feature

1. Create `negpy/features/<name>/` with `models.py`, `logic.py`, `processor.py`
2. Add a field to `WorkspaceConfig`; update `to_dict`/`from_flat_dict` (watch flat-namespace collisions)
3. Insert a `_run_stage(...)` call in `DarkroomEngine.process()`
4. For GPU: add a WGSL shader, wire it into `GPUEngine` (shader path + stage index + change detection), and add the feature's `shaders/` dir to `build.py` (`--add-data`)
5. Add a sidebar and register it in `ControlsPanel`; mark its `docs/USER_GUIDE.md` section with `<!-- panel:<key> -->` above the heading (`<key>` = the `_make_section` key) — that marker is what puts the ⓘ guide on the header
6. If it adds a toggle/tool/action, add a `REGISTRY` entry in `shortcut_registry.py` plus its action-map entry in `keyboard_shortcuts.py`
7. Add unit tests; if the feature has both CPU and GPU paths, add a parity test (pattern: `test_gpu_curve_parity.py`)
8. Document it: the panel and its controls in `docs/USER_GUIDE.md`, the stage's behaviour and math in `docs/PIPELINE.md`

## Style

- **Comments minimal.** Comment only non-obvious constraints the code can't express (a cache contract, an ordering requirement, a rejected-alternative trap). No comments that narrate what the next line does, restate the diff, or justify a change to a reviewer. Prefer one dense line over a paragraph; docstrings short and factual.

## Invariants & gotchas

- **CPU/GPU parity**: any change to a stage's math must land in both `logic.py` and its `.wgsl` shader. Constants mirrored as WGSL literals (histogram bins, zone density, metrics offsets) have parity tests — keep them in sync.
- **Working-space OETF + luminance row are inlined** in `lab_sharpen_h.wgsl` and `rl_init.wgsl` (Adobe RGB 1998 gamma 563/256, D65 Y row) — a TRC or primaries change must update them, not just `kernel/image/logic.py`. `LabUniforms` is declared in 6 lab shaders (lab, lab_sharpen_h/v, rl_blur_h, rl_div_v, rl_mult_v) — any field change touches all six plus the `struct.pack` in `gpu_engine.py`, and the trailing `_pad*` floats keep the block at 48 bytes. `rl_init.wgsl` binds no uniform at all (the auto layout prunes it).
