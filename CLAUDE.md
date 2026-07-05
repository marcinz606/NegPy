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

**Edit storage & sidecars.** Edits live primarily in SQLite (`edits.db`, keyed by content hash — `StorageRepository.save/load_file_settings`). Optionally they're mirrored to plain-file **sidecars** for archival: `negpy/services/assets/sidecar.py` writes `<source-basename>.negpy` **next to the source** (JSON = `WorkspaceConfig.to_dict()`, the full edit). On load, `select_file` calls `load_or_promote` — DB first, else a sidecar beside the source, and a loaded sidecar is **promoted into the DB** so the DB wins thereafter. Sidecars are written only on real export when `ExportConfig.export_sidecars_enabled` is on (not the contact sheet), or via the Export panel's "Export Edits Sidecars" section button (`AppController.export_edit_sidecars`).

### Feature pattern

Every feature under `negpy/features/<name>/` follows this structure:

- `models.py` — frozen dataclass config with defaults
- `logic.py` — pure functions operating on numpy arrays
- `processor.py` — thin wrapper with a `process(img, context) -> ImageBuffer` method
- `shaders/<name>.wgsl` — optional GPU compute shader

### CPU pipeline (`negpy/services/rendering/engine.py`)

`DarkroomEngine.process()` runs stages in order: base (geometry + normalization) → exposure → retouch → lab → toning → crop → finish. The cached stages (base, exposure, retouch, lab) go through `_run_stage()`, which hashes the stage config and skips re-execution if the hash matches the cached `CacheEntry`; toning/crop/finish run unconditionally. Source image change clears the whole cache; a process-mode change invalidates only base/exposure/retouch/lab. Note `settings.geometry` drives both the early geometry transform and the late crop stage.

**Dodge/burn lives in the exposure stage.** Polygon masks (`negpy/features/local/`, `LocalAdjustmentsConfig`) rasterise to a per-pixel **EV map** (`compute_local_ev_map`, Σ strength·alpha) consumed by the print curve as a print-exposure offset next to the WB `cmy_offsets` (`local_ev_scale` = −log10(2)/stretch range per channel) — burns roll into paper black through the toe, dodges lift through the shoulder, like enlarger exposure. There is no separate local stage or shader: the exposure `_run_stage` is keyed on `(settings.exposure, settings.local)`, the GPU binds the EV map as a texture in the exposure pass (`ev_scale.w` gates a 1×1 dummy when no masks), and tiled export slices the full-res EV map per tile. The UI section sits on the Exposure ("tone") page.

**Flat-field profile (rig-level) + lens distortion.** Flat-field is a per-rig **profile** (DB `flatfield_profiles`: name, path, **k1**), not a per-image edit. `FlatFieldConfig` (`apply`, `reference_path`, `k1`) mirrors the active profile into the recipe but the profile is canonical — `reference_path` and `k1` are **re-seeded from the active profile on every file load** (`session.py`), so a roll shares them. The *photometric* correction (illumination falloff) is baked at **source-load** (`apply_flatfield`, before the pipeline → before any warp, so the flat stays registered to the raw sensor grid). The *geometric* correction is a single radial coefficient **k1** (poly3 / Brown-Conrady, half-diagonal–normalized so it's rotation/aspect invariant) folded into the **geometry transform** (`apply_radial_distortion` / `transform.wgsl`), gated by the same `apply` toggle (`k1_eff = k1 if apply else 0`). Being a geometry op, k1 invalidates the **base stage** (CPU `base_key`; GPU `_detect_invalidated_stage` checks `flatfield.apply`/`k1`) — **not** `source_hash`/`flatfield_token` (that would force a RAW re-decode per slider step). Scale-to-fill is baked into the same remap (`compute_distortion_scale`, no empty borders). The correction must stay consistent across the image warp **and** the coordinate mappers (`create_uv_grid` forward, `map_coords_to_geometry` inverse via `map_point_radial`) so crop/autocrop/retouch/dodge-burn placements stay aligned; CPU stages read the effective k1 from `context.metrics["distortion_k1"]`, GPU passes `k1_eff` directly. Tiled export applies it on the CPU `img_rot`/IR path (the shader geometry uniform is zeroed there).

**Scene-linear working space.** The pipeline is **scene-linear internally**: the exposure stage (`exposure/logic.py`) emits linear reflectance (`10^-D`, no OETF), and the creative stages **Lab, Toning, Finish operate on linear light** (dodge/burn happens inside the exposure stage, pre-curve). The working-space OETF — the **ProPhoto RGB (ROMM) TRC** (gamma `1.8` with a linear toe below `1/512`), `working_oetf_encode`/`working_oetf_decode` in `kernel/image/logic.py`, mirrored in WGSL `oetf_encode`/`oetf_decode` — is applied **only as the final engine step** (CPU: end of `process()`; GPU: the `output_encode` pass in `gpu_engine.process_to_texture`), so the encoded buffer composes correctly with the ProPhoto ICC at the display/export boundary (`WORKING_COLOR_SPACE`). **Retouch runs in the linear island** like the other creative stages: dust *detection* is perceptual, so on the CPU it computes the detection luma on a `working_oetf_encode`d copy while *healing* in linear (`retouch/logic.py`) — the engine no longer brackets it. The GPU keeps a single encoded perceptual region (exposure encodes → clahe/retouch operate encoded → lab's `load_lin` decodes back to linear), equivalent within parity tolerance. Lab/Toning compute CIELAB directly from linear (`rgb_to_lab_working`/`lab_to_rgb_working` use the **ProPhoto ROMM primaries at D50**, no transfer decode); CIELAB ops are encoding-invariant; the glow highlight mask computes on a `working_oetf_encode`d copy so its display-domain threshold stays valid, and chemical toning (selenium/sepia) is **density-driven** — it reads print density `D = −log10(t)` straight off the linear buffer (`toning/logic.py` `TONING_CONSTANTS`), no OETF bracket (only the B&W black point keeps one). The flat master keeps its own log encoding and is never OETF-encoded. When changing the working **TRC**, update all six sites together: the CPU `working_oetf_*` helpers, the WGSL `oetf_*` in `exposure/output_encode/lab/metrics` shaders, and the chart (`charts.py`); changing the **primaries/white point** additionally means the CPU `_PROPHOTO_TO_XYZ`/`_XYZ_TO_PROPHOTO`/`_D50_WHITE` and the inline matrices + white in `lab.wgsl`/`toning.wgsl`. The greyscale export path re-encodes luma from the working TRC to gamma 2.2 to match its `GrayGamma2.2` tag (`_apply_color_management_u16_greyscale`).

**Flat ("for editing elsewhere") render intent.** When `settings.exposure.render_intent == RenderIntent.FLAT` (`negpy/features/exposure/models.py`), the Print stage uses `PhotometricProcessor._process_flat`, which calls `apply_flat_curve` (`negpy/features/exposure/logic.py`) — a true **log-video master**: the normalized log signal (`val`) is emitted **directly** as the code value, positive-oriented, `code = clip(lift + gain·(1 − val), 0, 1)` (Cineon-like `flat_log_gain`/`flat_log_lift`; `flat_curve_params` returns gain+lift). Crucially it does **not** apply `10^-D` or the working-space OETF — `10^-D` is the log→linear *decode* that would turn the signal back into a normal-contrast positive (that decode is why earlier "flat" attempts looked like ordinary photos, not log). The result is flat/milky and fully invertible for editing. It bypasses the asymmetric print kernel (`_apply_print_curve_kernel`) entirely. Ignores auto density/grade, cast removal, toe/shoulder, surround/flare, dodge/burn masks (WB rides as a per-channel log shift). The engine **bypasses retouch/lab/toning/finish** (crop still runs). This is the digital-intermediate master path. `ImageProcessor` forces the **CPU engine** for flat renders (no WGSL flat shader — guarantees numerical exactness). The desktop exposes it as a hybrid output intent: `AppState.flat_output`/`flat_format` (persisted) drive export, `flat_peek` drives a preview-only peek; `flat_master_config()` / `flat_export_config()` in `negpy/domain/models.py` derive the flat config (full-res by default when Flat is enabled; Print/Pixels sizing honoured when explicitly selected; 16-bit TIFF or linear DNG; export colour space follows the user's selection). DNG export writes an uncompressed 16-bit LinearRaw DNG with `tifffile` via `ImageProcessor._encode_dng_bytes` → `write_dng_linear` (`negpy/services/scanning/writer.py`); `NewSubfileType=0` + the DNG version tags are required for LibRaw/rawpy to accept it.

For **roll consistency**, flat masters only match across frames when the roll shares one normalization baseline; the Export panel shows a "Bake roll baseline" nudge (reusing `request_batch_normalization`) when flat output is on but bounds aren't locked. Flat masters use the **selected export colour space** like the print path — the pipeline's ProPhoto RGB working buffer is color-managed to that space at encode time (`_apply_color_management_u16_rgb`).

### GPU pipeline (`negpy/services/rendering/gpu_engine.py`)

`GPUEngine` runs the same logical pipeline as WGSL compute shaders via `wgpu`. It has its own change-detection logic (comparing previous vs. current `WorkspaceConfig` fields) to decide how far back to re-execute. Shader sources are loaded from `negpy/features/<name>/shaders/`. The CPU-side **auto-exposure analysis** (`analyze_log_exposure_bounds` / anchor / textural / shadow-refs) is cached per source via `process_to_texture`'s `analysis_source_hash` (passed from `run_pipeline`'s augmented `source_hash`): the cache key (`_analysis_cache_key`) mirrors the CPU engine's base-stage key plus the refs/anchor/textural gating fields, so creative-slider previews reuse the meter instead of recomputing it every frame. Callers passing explicit `bounds_override` etc. (tiled export) bypass the cache.

### Orchestration (`negpy/services/rendering/image_processor.py`)

`ImageProcessor` chooses between GPU and CPU paths. GPU is tried first; on failure it falls back to CPU. Export always runs at full resolution through `GPUEngine.process()` or the CPU engine.

**RGB Scan (trichromatic capture).** When `settings.rgbscan.enabled` with green/blue paths set (`negpy/features/rgbscan/`), one frame is assembled from three narrowband exposures: each RAW is decoded the same way (`output_color=raw`, linear) and the red channel of the red shot, green of green, blue of blue are stacked (`assemble_rgb`, the one merge primitive both paths call). When `rgbscan.align` (default on), green/blue are sub-pixel registered to the red exposure first via phase correlation (`cv2.phaseCorrelate` on each exposure's dominant channel → `cv2.warpAffine`; estimate runs at ≤1024px then scales, an implausibly large shift is treated as a failed lock and skipped) to kill fringing from frame-to-frame capture drift. The merge happens in **both decode paths** — `ImageProcessor._load_source_f32` (full-res export, via `_decode_sensor_rgb`) and `PreviewManager.load_linear_preview_rgb` (downsampled preview; decodes/caches each exposure through the normal preview path then combines channels — the fast preview decode skips rawpy `half_size` for linear X-Trans/`.RAF` decodes via `is_xtrans`, since `half_size` aliases the 6×6 CFA and casts the narrowband channels; Bayer and camera-WB previews keep it). Both must stay in sync — preview-only merging renders a gray frame (red exposure alone has near-zero G/B). The merged buffer is byte-format-identical to a normal single-RAW decode, so the entire downstream pipeline (normalization/inversion) is unchanged — RGB Scan is a **source-assembly toggle, orthogonal to process mode**, not a 4th process mode. Its identity is folded into `source_hash` via `rgbscan_token` (mirrors `flatfield_token`). The red exposure is the primary asset; green/blue ride along in the config like the flat-field `reference_path`. Desktop: a checkable "RGB Scan" button in the Files sidebar sets the sticky `rgbscan_mode` global; folder discovery (`AssetDiscoveryWorker`) then classifies each file by dominant Bayer-channel mean and groups consecutive triplets into one asset (green/blue paths travel in the asset dict, injected into `config.rgbscan` by `select_file`). Toggling the button while files are already loaded re-runs discovery over the loaded exposures (`AppController.set_rgb_scan_mode` → `request_asset_discovery(replace_existing=True)`), so the mode regroups/ungroups in place (not only on the next folder load); the rebuild collects every asset's red+green+blue paths, clears `uploaded_files` before re-adding (dedup-by-hash would otherwise drop the regrouped red), and reselects the frame the user was on. The Files context menu's "Edit RGB Triplet…" dialog reassigns channels and toggles alignment (the align flag rides in the asset dict and `session_triplets`). **Session restore** persists the triplet map (`session_triplets`) and rebuilds it via `AssetDiscoveryWorker._attach_restored_triplets` — re-discovery from the red path alone cannot regroup a triplet, so without this a restored RGB-scan frame degrades to the red exposure (renders gray).

### `PipelineContext`

Passed through every stage. Carries `scale_factor` (preview downsample ratio), `process_mode`, `active_roi`, and a `metrics` dict for inter-stage data (`content_rect`, `uv_grid`, histogram bounds, etc.).

### Desktop (MVC)

- **`AppState`** (`negpy/desktop/session.py`) — mutable dataclass for session state (current file, active tool, last render metrics, GPU toggle, etc.)
- **`AppController`** (`negpy/desktop/controller.py`) — single controller; all UI interactions call methods here; emits `config_updated` and `image_updated` signals
- **Workers** — heavy work (render, export, thumbnails) runs in `QThread`-backed worker objects in `negpy/desktop/workers/`; communicate via Qt signals
- **Sidebars** — each feature has a `negpy/desktop/view/sidebar/<name>.py` that reads from `AppState` and calls controller methods; all synced via `ControlsPanel._sync_all_sidebars()` on `config_updated`
- **Contact sheet output** — `ExportConfig.contact_sheet_output_path` (persisted via `last_export_config` sticky). Empty = follow export destination rules (same-as-source → first visible frame's folder; absolute → `export_path`). Non-empty path wins on export.
- **Contact sheet templates** — `.toml` files in `~/NegPy/contact_sheets/` (`ContactSheetTemplates`). `ExportConfig.contact_sheet_template` stores the selected name; **Default** uses `contact_sheet_default_*` snapshot fields. Edits to layout spinboxes write back to the active template. See `docs/CONTACT_SHEET_TEMPLATES.md`.

### Adding a new feature

1. Create `negpy/features/<name>/` with `models.py`, `logic.py`, `processor.py`
2. Add a field to `WorkspaceConfig` and update `to_dict` / `from_flat_dict`
3. Insert a `_run_stage(...)` call in `DarkroomEngine.process()`
4. For GPU: add a WGSL shader, wire it into `GPUEngine` (shader path + stage index/change-detection), and add the feature's `shaders/` dir to `build.py` (`--add-data`) so it ships in the packaged app
5. Add a sidebar and register it in `ControlsPanel`
