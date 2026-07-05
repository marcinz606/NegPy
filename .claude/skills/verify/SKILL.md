---
name: verify
description: Drive the real NegPy desktop app headlessly and capture rendered evidence — use when verifying pipeline/UI changes end-to-end (not via pytest).
---

# Verifying NegPy changes end-to-end

The surface is the PyQt6 GUI; the render pipeline's output is the canvas.

## Recipe that works

1. **Sandbox app state** so the run never touches `~/Documents/NegPy`:
   `NEGPY_USER_DIR=<scratch>/negpy_home` (checked in `get_default_user_dir`).
   Create the subdirs from `_bootstrap_environment` (or call it) before
   `StorageRepository.initialize()`.
2. **Suppress the first-run tutorial** or every window grab is dimmed by the
   overlay: `repo.save_global_setting("tutorial_seen", True)` before
   constructing `MainWindow`.
3. **Headless display**: `xvfb-run -a -s "-screen 0 1920x1080x24" uv run python driver.py`.
   wgpu/Vulkan works fine under Xvfb on this machine (real GPU).
4. **Driver skeleton**: build the app exactly like `negpy/desktop/main.py`
   (`StorageRepository` → `DesktopSessionManager` → `AppController` →
   `MainWindow`), then drive a QTimer state machine:
   - `controller.load_file("samples/06.raw")` — same method the Files sidebar calls.
   - advance steps on `controller.image_updated` (skip `state.last_metrics["splash"]`,
     guard against multiple emissions per render, ~600 ms settle).
   - change edits like the sidebars do: `session.update_config(replace(state.config, ...))`
     + `controller.request_render()`.
   - **pixel evidence**: `window.canvas.grab()` (canvas only, overlay-free) →
     numpy stats; `window.grab().save(...)` for reviewer screenshots.
   - exit with a nonzero code on failed assertions; 120 s timeout guard.

A working driver from the dodge/burn + toning verification lives in the
session scratchpad as `drive_verify.py` — copy its `grab`/`set_cfg`/state-machine
parts rather than rewriting.

## Gotchas

- `samples/06.raw` is a dark graveyard scene — burn/dodge probes need the sky
  band (top of frame in raw coords) to see roll-off; mid-frame is near black.
- Mask vertices are raw-image normalized coords, not canvas coords; analyze
  screenshots diff-based (`before - after > threshold`) instead of guessing
  where a mask lands on the canvas.
- Status bar process label may lag when configs are set programmatically —
  don't treat it as ground truth for the active mode.
