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
- [x] `retouch.wgsl`: Advanced 16-point circular inpainting with perceptual grain synthesis.
- [x] `lab.wgsl`: Perceptual Luma-only USM sharpening and Spectral Crosstalk.
- [x] `clahe_hist.wgsl`, `clahe_cdf.wgsl`, `clahe_apply.wgsl`: Perceptual 3-pass Adaptive Histogram Equalization.
- [x] `toning.wgsl`: Saturation, Chemical Toning, Paper Tint, B&W mode, Final Gamma, and ROI-based Cropping.
- [x] `autocrop.wgsl`: High-performance border detection via row/column luminance reduction.
- [x] `metrics.wgsl`: 4-channel histogram calculation using atomic operations.

### Phase 3: Engine & Pipeline (Completed)
- [x] `GPUEngine` orchestrating a complex **8-stage** compute pipeline.
- [x] Hybrid Calibration: CPU-based density analysis passed to GPU.
- [x] Zero-copy preview path (returns `GPUTexture` wrapper).
- [x] **48-bit TIFF Export**: Preserves 16-bit per channel precision by bypassing PIL for RGB TIFF saving.
- [x] **Texture Pooling**: Intermediate textures are cached and reused to minimize VRAM churn.

### Phase 4: UI & Controller Integration (Completed)
- [x] `ImageCanvas` refactored to `QStackedLayout` with Z-order management.
- [x] `GPUCanvasWidget` with NDC-based letterboxing and centering.
- [x] `CanvasOverlay` synchronized with GPU dimensions for accurate tool interaction.
- [x] **Feature Parity**: All interactive tools (Crop, WB Picker, Histogram) fully functional on GPU.

### Next Milestone: Stability & Performance

### Reliability & Polish (High Priority)
- [x] **Adaptive Export Tiling**: Implemented high-resolution tiling loop (2048px tiles) with 32px halo handling to prevent TDR and VRAM exhaustion.
- [ ] **Uniform Buffer Pooling**: Optimize uniform buffer updates to use `queue.write_buffer` on long-lived buffers instead of frequent allocations.

### Optimization (Medium Priority)
- [ ] **Resource Lifecycle**: Implement explicit reference counting or a cleanup pass for the texture pool to free memory when a session is closed.
- [ ] **Command Buffer Batching**: Consolidate pipeline executions into fewer command buffer submissions.

## 3. Implementation Log (Milestone 5 - Reliability)

1. [x] **Advanced Inpainting**: Upgraded healing logic to 16-point sampling with perceptual grain synthesis for invisible retouching.
2. [x] **Pure GPU Metrics**: Eliminated the final full-image readback by moving AutoCrop and Histogram entirely to GPU reduction shaders.
3. [x] **Perceptual Balance**: Fine-tuned sharpening and CLAHE to operate in gamma-encoded space, matching human vision.
