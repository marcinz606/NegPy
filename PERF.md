# PERF.md — Initial-load profiling handover

Handover for the initial-file-load performance work. A fresh session should be able to
resume from this file alone.

## Goal

Slider-drag rendering is already fast (per-render caches work). The pain is **cold file
load**. Find the dominant cold-load stage with real numbers, then apply low-risk fixes.
Focus is the **GPU path** (CPU engine is only a fallback); normalization + auto-exposure
analysis run on CPU in both paths.

## Status

- **Branch:** `perf/initial-load-timing` (off `main`), commit `26783a6`. Not pushed.
- **Step 1 (instrumentation): DONE**, all green (ruff + ty + full pytest).
- **Step 2 + 3: PENDING** — blocked on capturing baseline numbers (see "Next" below).
- Full plan: `~/.claude/plans/do-performance-profiling-of-misty-blum.md`.

## Verified cost map (read from source; NOT yet measured)

Ordered heavy work on a cold load of a never-seen file:

1. **RAW decode / demosaic** — `raw.postprocess(...)` in
   `PreviewManager._load_from_open_raw` (`negpy/services/rendering/preview_manager.py`).
   LibRaw, CPU-bound, the suspected dominant cost. Preview uses `half_size=True`+`LINEAR`
   for Bayer (X-Trans can't — aliases the 6×6 CFA). Cached by content hash → only the
   first view pays it.
2. **Process-mode autodetect = a SECOND decode** — `_detect_mode`
   (`negpy/desktop/workers/render.py`) calls `PreviewManager.decode_for_detection` (another
   half-size `postprocess`) because camera-WB hides the C-41 mask. **Already gated to new
   files only:** `detect_mode = force_detect or (autodetect_enabled and
   current_file_is_new)` (`controller.py` `load_file`) and `current_file_is_new =
   saved_config is None` (`negpy/desktop/session.py`). So the double-decode hits only the
   *genuine first view of an unedited file*; revisits skip it.
3. **CPU auto-exposure analysis (blocks before GPU dispatch)** — in
   `GPUEngine.process_to_texture` (`gpu_engine.py`). On a C-41 cold load with cast-removal
   + auto-density + auto-contrast, five functions (`analyze_log_exposure_bounds`,
   `measure_shadow_log_refs`, `measure_neutral_axis`, `measure_anchor`,
   `measure_textural_range`) each get the **same** downsampled `analysis_source` and
   **independently redo `np.log10(...)` + `_block_median_grid(...)`** — the identical
   prefilter computed **5×**. All five have `_from_log` variants (see
   `negpy/features/exposure/normalization.py`) → clean hoist. This is Step 2.
4. **GPU one-time init** — `GPUEngine._init_resources` (`gpu_engine.py`) lazily compiles
   14 WGSL shaders + creates 14 pipelines on the **first** `process_to_texture` of the
   session. Paid once, but lands on the first file after launch. This is Step 3.
5. **Per-render caches** (analysis cache keyed to exclude creative sliders; texture pool;
   bind-group cache) — already effective; this is why sliders are fast. **Leave alone.**

## Step 1 — DONE: `load-timing` INFO logs

Instrumented the whole cold-load path. Every line prefixed `load-timing`, in **ms**.
Default log level is INFO (`negpy/kernel/system/override.py` `log_level="info"`) → these
print to stdout on `make run`.

| Log key | Stage | Thread |
|---|---|---|
| `decode.postprocess` | LibRaw demosaic (shows `fast=`) | worker |
| `decode.resize` | preview downsample | worker |
| `decode.total` | demosaic + orient + resize | worker |
| `load_splash_and_linear` / `load_linear_preview` | decode + file-open wrapper | worker |
| `detect` | process-mode autodetect; **`re_decode=True` = 2nd decode fired** | worker |
| `preview_worker_total` | load request → decoded buffer | worker |
| `preview_e2e` | load request → decoded buffer (incl. signal hops) | UI |
| `gpu_init` | one-time WGSL compile — **first file of session only** | render |
| `analysis` | CPU meter (bounds/refs/anchor/textural), **once per source** | render |
| `first_render` | decoded buffer → painted | UI |

**Quiet by design:** prefetch/cache-warm decodes stay at DEBUG; `analysis` + `gpu_init`
fire once per source/session (not per slider), via a `log_timings` flag threaded through
`PreviewManager` and once-per-hash guards.

**Files touched:** `preview_manager.py`, `workers/render.py`, `gpu_engine.py`
(+`import time`), `controller.py`. Additive/logging only — no behavior change.

## Next — capture baseline, then Steps 2 & 3

### Measure (do this first)

```
make run
```
Load `/home/marcin/Pobrane/_DSC0875.NEF` **fresh** (autodetect on) → read stdout for
`load-timing`. Note: the double-decode (`detect ... re_decode=True`) only shows on a
genuinely new file (no saved edit). Record the numbers, decide which stage dominates.
`samples/raw0004.dng` is absent in this checkout — use the NEF above.

### Step 2 — Share the analysis prefilter (5× → 1×; low risk)

Compute `img_log = log10(clip(analysis_source))` once, then
`prefiltered = _block_median_grid(img_log)` once (after ROI + `analysis_buffer` crop), and
feed all five analysis functions the prefiltered grid via their `_from_log` variants —
making those skip their internal log10/block-median when handed an already-prefiltered
grid (a flag, or a small shared helper owning the prefilter). Ordering already holds:
bounds first, then anchor/neutral consume bounds; the prefilter is bounds-independent.

- Primary edit: `negpy/features/exposure/normalization.py`.
- Call sites: `GPUEngine.process_to_texture` (`gpu_engine.py`, the analysis block that
  calls the five `measure_*` / `analyze_*`) and the CPU-engine equivalent
  `NormalizationProcessor.process` in `negpy/features/exposure/processor.py`.
- **Test:** assert shared-prefilter results equal the current per-function results
  (bounds/anchor/refs/textural) within tolerance on a synthetic image.

### Step 3 — Warm GPU shaders/pipelines at startup (low risk)

Call `GPUEngine._init_resources()` on a background thread at app launch so the first file
skips the one-time WGSL compile. Locate GPUEngine/ImageProcessor construction in desktop
app init; kick after the window shows. **Verify wgpu device thread-safety** — if
`create_shader_module` / `create_compute_pipeline` must run on the device's thread, warm
during idle on that thread (e.g. `QTimer.singleShot(0, ...)` post-show) instead.

## Not doing (with reason)

- **"Cache mode detection"** — effectively already implemented (`current_file_is_new =
  saved_config is None`); detection re-decode only hits the true first view. Only remaining
  sub-lever is a cheaper `decode_for_detection` (already half-sizes). Revisit only if
  Step-1 numbers show detection is a real chunk.
- **Aggressive double-decode removal** (decode once no-WB, apply WB numerically for
  preview) — touches preview white-balance behavior. Deselected; out of scope.

## Verify (any change)

`make all` (ruff + ty + pytest) green. Before/after via the `load-timing` numbers on the
same fresh NEF. Step 3 win is visible as the `gpu_init` spike moving off file #1.
