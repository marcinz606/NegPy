# Coolscan Scan follow-ups — film return + whole-strip preview

Branch `feat/coolscan`. Work on the SANE Coolscan **Scan** tab, targeting the Nikon
LS-50 / Coolscan V with the SA-21 6-frame strip adapter.

## Done

### 1. Film return (eject)
- **Root-cause fix.** `eject` is a `SANE_TYPE_BUTTON`; the old `setattr(dev, "eject", True)`
  always raised `AttributeError: Buttons don't have values`, so eject never fired. Now
  pressed at the C layer via `dev.dev.set_option(index, 1)` — helper `_press_button`
  (`negpy/infrastructure/scanners/sane_backend.py`). Same path `scanimage --eject` takes.
  **(SUPERSEDED — `set_option` also raises on real hardware; see Session 2 below.)**
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
  **(SUPERSEDED — now the device's *lowest* DPI; see Session 2.)**

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

## Hardware verification (Nikon LS-50 ED + SA-21, 2026-07-17)

- [x] **Eject — BUG FOUND & FIXED.** `_press_button`'s `dev.dev.set_option(index, 1)`
      raises `SANE_TYPE_BUTTON ... can't be set` on real python-sane 2.9.2; all three
      python-sane button paths (setattr / set_option / set_auto_option) raise, so eject
      had **never** worked. The unit test missed it — the fake `set_option` didn't
      replicate the rejection. Fix: `eject()` now detects capability, closes the handle,
      and presses via `scanimage --eject` (the C-level `sane_control_option` SET_VALUE);
      scanimage's spurious post-eject "out of documents" exit is treated as success.
      Verified live: strip physically ejects; `SaneBackend.eject()` returns `True`.
- [x] **USB stability — PASS.** `power/control = on` (autosuspend off). No re-enumeration
      observed; the one transient "Error during device I/O" (from `--reset`) recovered on
      its own — confirms the `run_scan` retry-once path. Also learned: **`--reset` briefly
      wedges the transport, `--load` feeds/ejects the film** — the app calls neither.
- [x] **Offset direction/scale — VERIFIED, no code change.** subframe 4 mm = exactly
      **−63 px = −4.00 mm** @ 400 dpi (1:1 mm), content moves toward the raster top →
      display right, consistent with the indicator's `edge="right"`. Slider kept at 0–4 mm
      (fine nudge; hardware range is 0–37.83 mm but the crop window handles gross framing).
- [x] **Orientation — kept `_DISPLAY_ROTATION_DEG = 90`** (self-consistent with the offset
      edge). Test frame was a portrait-shot (reads sideways regardless); flip is a trivial
      one-constant change if a *landscape* frame later reads upside-down in real use.
- Note: the Coolscan firmware **auto-ejects the strip after an idle timeout**, independent
  of any command (this is what "pushed the strip out" during an AFK gap in testing).

## Session 2 — live iteration on the LS-50 (2026-07-17)

Changes made while testing against the physical scanner:

- **Eject fixed for real.** python-sane 2.9.2 cannot press a `SANE_TYPE_BUTTON`
  (setattr / set_option / set_auto_option all raise). `SaneBackend.eject()` now detects
  capability, closes the handle, and presses via `scanimage --eject`; the spurious
  post-eject "out of documents" exit is treated as success. `_press_button` removed.
  (`sane_backend.py`, `tests/scanners/test_scanner_eject.py`.)
- **USB re-enumeration recovery.** The LS-50 re-enumerates under load
  (`003:006` → `003:007`), so the cached SANE id goes stale and `sane.open` raises
  "Invalid argument". `SaneBackend._open_device` re-lists, remaps to the same scanner
  (vendor+model, else the sole same-prefix device), retries, and caches the remap; both
  scan and eject route through it. (`sane_backend.py`, `tests/scanners/test_sane_reopen.py`.)
- **Preview at the device's lowest DPI** (was smallest ≥ 300) — smaller transfer, less
  transient I/O on a flaky link. (`strip_preview_dialog._preview_dpi`.)
- **Offset indicator direction fixed.** Content shifts toward display-right as offset
  rises (verified 1:1 mm); the indicator band now grows from the **left** to track the
  slider, instead of drawing from the right. (`scan_window_label.py`, `strip_preview_dialog.py`.)
- **Preview shows a positive.** `_preview_positive` (per-channel invert + 1/99 auto-level)
  renders a rough positive through the orange mask — a framing aid, not the develop.
- **Scan straight from the dialog.** New **Scan** button applies the selection and starts
  the real scan immediately (`scan_requested()` → sidebar `_on_scan`). **Use** still just
  applies and returns.
- **Preview sizing.** Tiles scale to the dialog height at the landscape aspect
  (`_rescale_tiles` on resize/show); per-tile placeholder text removed; a single top help
  box explains the workflow (preview → crop per frame → offset + re-preview → tick → Use/Scan).

Not built (deferred): the **linear per-frame offset correction** (`base + step × frame`) for
progressive frame-pitch drift down a roll.

Updated `CLAUDE.md` (SANE eject + re-enumeration invariants) and `docs/COOLSCAN_SCANNING.md`.
All green: `make all` — 2186 passed.
