# Coolscan Scan follow-ups — film return + whole-strip preview

Branch `feat/coolscan`. Work on the SANE Coolscan **Scan** tab, targeting the Nikon
LS-50 / Coolscan V with the SA-21 6-frame strip adapter.

## Done

### 1. Film return (eject)
- **Root-cause fix.** `eject` is a `SANE_TYPE_BUTTON`; the old `setattr(dev, "eject", True)`
  always raised `AttributeError: Buttons don't have values`, so eject never fired. Now
  pressed at the C layer via `dev.dev.set_option(index, 1)` — helper `_press_button`
  (`negpy/infrastructure/scanners/sane_backend.py`). Same path `scanimage --eject` takes.
- **Auto-return after a batch.** `run_batch` ejects the strip on a clean finish only
  (`negpy/desktop/workers/scan_worker.py`); capability-gated no-op on devices without eject.
  The on-demand ⏏ button (already in the device row) now works too.

### 2. Whole-strip preview
- **New `StripPreviewDialog`** (`negpy/desktop/view/widgets/strip_preview_dialog.py`):
  a horizontal strip of landscape frame tiles; lazy per-frame **Preview** + **Preview all**
  (single-flight, chained); a **crop window per frame**; a **Scan** checkbox per frame to
  scan a subset (skip the rest); one global feed-axis **Offset** slider.
- **Frames shown landscape.** The raster is portrait; each preview is rotated 90° to read
  like a frame on the strip. The crop rect transforms scan↔display (pinned to Qt's
  `rotate(90)`, round-trip unit-tested) so a window drawn on the rotated view still crops
  the right region. Backend unchanged.
- **Data model.** `ScannerSettings.frame_windows: dict[int, Rect]` + `selected_frames: tuple`
  (+ JSON coercion); pure `resolve_batch_selection()`; `BatchRequest` now carries
  `frames` + `frame_windows`; `run_batch` applies the per-frame window. Sidebar opens the
  dialog via **"Preview strip…"** for frame devices; the from/to spinbox range stays as the
  fallback; `_update_settings_from_ui` carries the new fields (else typing wiped the selection).

### 3. UI polish
- Landscape tiles, bigger previews, tight spacing, a subtle translucent overlay box
  (frame-number checkbox + eye preview button) on each frame.
- Horizontal scrollbar styled to match the vertical one (`modern_dark.qss`).
- Preview DPI bumped from the device floor to the smallest supported ≥ 300.

### 4. Resilience
- `ScannerService.run_scan` retries **once** on a transient `Error during device I/O` /
  `device busy` (fresh open, 0.5 s settle, cancel-aware, bounded — no loop). Covers
  preview, single, and batch.
- "Preview all" **continues past a failed frame** (records which failed in the status)
  instead of aborting the whole strip.

### Cleanup / docs
- Removed the now-orphaned single-frame `scan_preview_dialog.py`.
- Updated `docs/COOLSCAN_SCANNING.md` and `docs/CHANGELOG.md`.

### Tests
All green (`make lint`, `make type`, full `pytest`). New/updated:
`tests/scanners/test_scanner_eject.py`, `tests/test_scan_worker.py`,
`tests/scanners/test_scanner_settings.py`, `tests/test_scan_sidebar.py`,
`tests/test_strip_preview_dialog.py`, `tests/scanners/test_service.py`.

## Next steps (need the physical LS-50)

- [ ] **Verify eject physically returns the strip:** `scanimage -d '<dev>' --eject`.
      If it doesn't move film, escalate: `scanimage -d '<dev>' -A` (capture the real option
      list), then try an armed same-session eject (`frame=1` → `start(); cancel()` → press),
      or `--reset` / `--load`. Wire whichever moves the SA-21.
- [ ] **USB stability.** `control` should read `on` (autosuspend off):
      `for d in /sys/bus/usb/devices/*/idVendor; do grep -qi 04b0 "$d" && p=$(dirname "$d") && echo "$p control=$(cat $p/power/control) autosuspend=$(cat $p/power/autosuspend_delay_ms)ms"; done`
      If `auto`, the udev rule didn't take for the current enumeration — make it stick.
      Device re-enumeration (`003:006` → `003:009` across the session) is what triggers the
      "Error during device I/O".
- [ ] **Confirm cosmetic orientation on a real scan:** which way *up* the landscape reads
      (flip = one constant, `_DISPLAY_ROTATION_DEG`), and the offset-line side (left vs right).
      The crop is correct either way.
- [ ] **Confirm offset direction/scale** (coolscan3 `subframe`, mm) with a strip loaded —
      never verified live (the feeder was empty during development).
