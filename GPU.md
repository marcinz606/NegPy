# GPU Acceleration Plan (Cross-Platform: WebGPU)

## Objective
Migrate image processing and rendering to GPU to achieve real-time performance on **Linux, Windows, and macOS (Apple Silicon)** using WebGPU (`wgpu-py`).

## 1. Status Overview

### Phase 1: Infrastructure (Completed)
- [x] `GPUDevice` singleton for adapter/device management.
- [x] `GPUTexture` and `GPUBuffer` wrappers.
- [x] `ShaderLoader` for WGSL management.
- [x] Integration with `rendercanvas` for Qt display.

### Phase 2: Compute Shaders (Completed)
- [x] `normalization.wgsl`: Log-Normalization with custom `log10`.
- [x] `exposure.wgsl`: Photometric H&D curve simulation.
- [x] `transform.wgsl`: Rotation/Flip with manual bilinear interpolation.
- [x] `retouch.wgsl`: Advanced 16-point circular inpainting with local grain synthesis.
- [x] `lab.wgsl`: Perceptual Luma-only USM sharpening and Spectral Crosstalk.
- [x] `clahe_hist.wgsl`, `clahe_cdf.wgsl`, `clahe_apply.wgsl`: Perceptual 3-pass Adaptive Histogram Equalization.
- [x] `toning.wgsl`: Saturation, Chemical Toning, Paper Tint, B&W mode, Final Gamma, and ROI-based Cropping.
- [x] `autocrop.wgsl`: High-performance border detection via row/column luminance reduction.
- [x] `metrics.wgsl`: 4-channel histogram calculation using atomic operations.
- [x] `layout.wgsl`: Canvas expansion, background fill, and letterboxing.

### Phase 3: Engine & Pipeline (Completed)
- [x] `GPUEngine` orchestrating a complex **10-stage** compute pipeline.
- [x] Hybrid Calibration: CPU-based density analysis passed to GPU.
- [x] Zero-copy preview path (returns `GPUTexture` wrapper).
- [x] **48-bit TIFF Export**: Preserves 16-bit per channel precision by bypassing PIL for RGB TIFF saving.
- [x] **Texture Pooling**: Intermediate textures are cached and reused to minimize VRAM churn.
- [x] **Unified Uniform Block**: Single persistent UBO with 256-byte aligned offsets for peak efficiency.
- [x] **Resource Lifecycle**: GC-based cleanup for cached textures to avoid race conditions with display widgets.

### Phase 4: UI & Controller Integration (Completed)
- [x] `ImageCanvas` refactored to `QStackedLayout` with Z-order management.
- [x] `GPUCanvasWidget` with NDC-based letterboxing and centering.
- [x] `CanvasOverlay` synchronized with GPU dimensions for accurate tool interaction.
- [x] **Async Metrics Delivery**: Transitioned to a non-blocking `map_async` pattern. Preview updates are instant, with histogram data arriving asynchronously.
- [x] **Feature Parity**: All interactive tools (Crop, WB Picker, Histogram) fully functional on GPU.

## 2. Next Milestone: Advanced Polish (Completed)
- [x] **Adaptive Tiling Workgroups**: Dynamic calculation based on hardware limits.
- [x] **Shared Histogram Context**: Optimized tiling with global CDF storage buffers.

## 3. Implementation Log (Milestone 11 - Final Optimization & Stability)

1. [x] **True Async UI Feedback**: Decoupled rendering from readback. `RenderWorker` now emits images immediately, with metrics following via a background `metrics_updated` signal.
2. [x] **Eliminated Micro-stutters**: The UI remains 100% responsive during rapid parameter changes.
3. [x] **Robust Uniform Management**: Finalized the consolidated UBO logic with 256-byte alignment.
4. [x] **Rotation-Aware Tiling**: Fixed broadcasting errors for vertical image exports.
5. [x] **Race-Condition Free Cleanup**: Transitioned to GC-managed texture lifecycles to prevent "texture destroyed" validation errors.
6. [x] **GPU Status UI**: Added real-time acceleration indicator to the session panel.
7. [x] **Runtime Mode Toggle**: Implemented explicit GPU/CPU switch in the UI, allowing users to choose their preferred backend.
