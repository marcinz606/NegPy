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

**Flat ("for editing elsewhere") render intent.** When `settings.exposure.render_intent == RenderIntent.FLAT` (`negpy/features/exposure/models.py`), the Print stage uses `PhotometricProcessor._process_flat` (a fixed low-contrast, neutral curve via `flat_curve_params`, ignoring auto density/grade, cast removal, toe/shoulder, surround/flare), and the engine **bypasses retouch/lab/local/toning/finish** (crop still runs). This is the digital-intermediate master path. `ImageProcessor` forces the **CPU engine** for flat renders (no WGSL flat shader — guarantees numerical exactness). The desktop exposes it as a hybrid output intent: `AppState.flat_output`/`flat_format` (persisted) drive export, `flat_peek` drives a preview-only peek; `flat_master_config()` / `flat_export_config()` in `negpy/domain/models.py` derive the flat config (wide-gamut ProPhoto, 16-bit TIFF or linear DNG). DNG export uses `pidng` (core dependency) via `ImageProcessor._encode_dng_bytes`.

For **roll consistency**, flat masters only match across frames when the roll shares one normalization baseline; the Export panel shows a "Bake roll baseline" nudge (reusing `request_batch_normalization`) when flat output is on but bounds aren't locked. For **wide gamut**, flat exports of camera RAWs apply the camera's own matrix (rawpy `rgb_xyz_matrix` = LibRaw `cam_xyz`, an XYZ→camera matrix — inverted, mapped to ProPhoto, row-normalized to keep neutrals neutral) in `negpy/infrastructure/display/camera_color.py`, applied inside `ImageProcessor._load_source_f32` **before** normalization and **only** when the render intent is flat (the print path is byte-for-byte unaffected). `_load_source_f32` returns the effective working space (ProPhoto) so the encoder skips a redundant conversion; non-camera sources fall back to the default behaviour.

### GPU pipeline (`negpy/services/rendering/gpu_engine.py`)

`GPUEngine` runs the same logical pipeline as WGSL compute shaders via `wgpu`. It has its own change-detection logic (comparing previous vs. current `WorkspaceConfig` fields) to decide how far back to re-execute. Shader sources are loaded from `negpy/features/<name>/shaders/`.

### Orchestration (`negpy/services/rendering/image_processor.py`)

`ImageProcessor` chooses between GPU and CPU paths. GPU is tried first; on failure it falls back to CPU. Export always runs at full resolution through `GPUEngine.process()` or the CPU engine.

**RGB Scan (trichromatic capture).** When `settings.rgbscan.enabled` with green/blue paths set (`negpy/features/rgbscan/`), one frame is assembled from three narrowband exposures: each RAW is decoded the same way (`output_color=raw`, linear) and the red channel of the red shot, green of green, blue of blue are stacked (`assemble_rgb`, the one merge primitive both paths call). When `rgbscan.align` (default on), green/blue are sub-pixel registered to the red exposure first via phase correlation (`cv2.phaseCorrelate` on each exposure's dominant channel → `cv2.warpAffine`; estimate runs at ≤1024px then scales, an implausibly large shift is treated as a failed lock and skipped) to kill fringing from frame-to-frame capture drift. The merge happens in **both decode paths** — `ImageProcessor._load_source_f32` (full-res export, via `_decode_sensor_rgb`) and `PreviewManager.load_linear_preview_rgb` (downsampled preview; decodes/caches each exposure through the normal preview path then combines channels). Both must stay in sync — preview-only merging renders a gray frame (red exposure alone has near-zero G/B). The merged buffer is byte-format-identical to a normal single-RAW decode, so the entire downstream pipeline (normalization/inversion) is unchanged — RGB Scan is a **source-assembly toggle, orthogonal to process mode**, not a 4th process mode. Its identity is folded into `source_hash` via `rgbscan_token` (mirrors `flatfield_token`). The red exposure is the primary asset; green/blue ride along in the config like the flat-field `reference_path`. Desktop: a checkable "RGB Scan" button in the Files sidebar sets the sticky `rgbscan_mode` global; folder discovery (`AssetDiscoveryWorker`) then classifies each file by dominant Bayer-channel mean and groups consecutive triplets into one asset (green/blue paths travel in the asset dict, injected into `config.rgbscan` by `select_file`). The Files context menu's "Edit RGB Triplet…" dialog reassigns channels and toggles alignment (the align flag rides in the asset dict and `session_triplets`). **Session restore** persists the triplet map (`session_triplets`) and rebuilds it via `AssetDiscoveryWorker._attach_restored_triplets` — re-discovery from the red path alone cannot regroup a triplet, so without this a restored RGB-scan frame degrades to the red exposure (renders gray).

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
