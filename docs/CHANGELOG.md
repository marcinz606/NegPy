# Change Log

## 0.9.4

- **Native UI**: Migrated from Streamlit/Electron to a native PyQt6 desktop application for better performance and system integration.
- **Hardware Acceleration**: Introduced WebGPU support for real-time, high-fidelity image processing (compatible with Vulkan, Metal, and DX12).
- **Improved Dust Removal**: Ported the automatic dust detection and healing algorithm to GPU with several enhancements:
    - Directional line protection to prevent accidental removal of wires/powerlines.
    - Context-aware variance analysis (9x9 window) to protect fine texture.
    - Bright-only gating to target dust specifically without affecting dark details.
- **Color Management**: Enabled ICC profile proofing in GPU mode via optimized VRAM-to-CPU readback.
- **UI Refinement**: 
    - New sidebar layout with a centralized header for app version and hardware settings.
    - Unified UI theme and improved control groupings.
    - Snap-to-integer sliders for discrete parameters like crop offset and brush sizes.
- **Stability**: Added comprehensive unit tests for the GPU infrastructure and rendering engine.

## 0.9.3

- Added white balance color picker for fine-tuning white balance (click neutral grey)
- Added manual crop options (click top left and bottom right corners to set it)
- Added basic saturation slider
- Added more border options
- Added original resolution export option
- Added Input/Output .icc profile support
- Added input icc profile for narrowband RGB (should mitigate common oversaturation issues)
- Added horizontal & vertical flip options
- UI redesign: main actions moved under the preview, film strip moved to the right.
- Add new version check on startup (Displays tooltip near the logo if new version is available)

## 0.9.2

- Make export consistent with preview (same demosaic + log bounds analysis)

## 0.9.1

- Explicit support for more raw extensions for file picker.

