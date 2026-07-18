# Coolscan Scan follow-ups

Branch `feat/coolscan`. SANE Coolscan **Scan** tab, targeting the Nikon LS-50 /
Coolscan V with the SA-21 6-frame strip adapter; roll adapters go up to 40 frames.

## Done — sessions 1–2 (2026-07-17, committed)

- **Eject.** python-sane 2.9.2 cannot press a `SANE_TYPE_BUTTON` (setattr /
  set_option / set_auto_option all raise), so `SaneBackend.eject()` detects
  capability, closes the handle, and presses via `scanimage --eject`; the
  spurious post-eject "out of documents" exit is success. Auto-return after a
  clean batch + on-demand ⏏ button. **Verified live: strip physically ejects.**
- **USB re-enumeration recovery.** The LS-50 re-enumerates under load (stale id →
  `sane.open` "Invalid argument"); `_open_device` re-lists, remaps to the same
  scanner, retries once, caches the remap. All opens route through it.
- **Whole-strip preview** (`StripPreviewDialog`): per-frame tiles, lazy preview +
  Preview all (single-flight, continues past a failed frame), per-frame crop
  windows (scan↔rotated-display transform, round-trip tested), per-frame Scan
  checkboxes, Scan-straight-from-dialog, rough positive rendering
  (`_preview_positive`), previews at the device's lowest DPI.
- **Data model.** `ScannerSettings.frame_windows` + `selected_frames`;
  `BatchRequest(frames, frame_windows)`; `resolve_batch_selection()`;
  `run_scan` retries once on transient device I/O.
- **Hardware facts (LS-50 + SA-21).** `subframe` offset is 1:1 mm, content moves
  toward display-right; raster is portrait, shown rotated 90°; firmware
  auto-ejects the strip after an idle timeout; never issue `--load`/`--reset`
  on a seated strip; USB autosuspend must be off (udev rule).

## Session 3 (2026-07-18, uncommitted)

### Scanner session seam (roll handover — PR #497)
- **`SaneSession`** (`sane_backend.py`): exclusive device hold for batch/roll
  workflows — `backend.open_session(device_id)` opens once (via the self-healing
  `_open_device`), `session.scan(params, …)` per frame on the held handle
  (sane_cancel between frames, never closes), one release via `session.close()`
  or `session.eject()` (closes first — scanimage needs the single open slot).
  Context-manager support; `ScannerService.open_session()` passthrough.
- **Get-out-of-the-way guards:** while a session holds a device, backend
  `scan()`/`eject()` refuse it (stale *and* remapped id) and
  `list_devices()`/`refresh_devices()` reuse the cached entry instead of probing.
- NegPy's own range-batch stays one-session-per-frame on purpose (the
  transient-I/O retry depends on fresh opens).
- **Known gap:** on a held handle, `ae`/`infrared`/`autofocus`/window geometry
  are only written when *enabled*, never reset — they latch across session
  scans. Needs an always-write sweep before a roll workflow scans on a session.

### Strip preview layout
- Tiles are **constant size** (140 px tall, width from device aspect) — no more
  scale-to-window. **Rows of 6** (one SA-21 strip per row), vertical scroll;
  a 40-frame roll wraps to 7 rows. Dialog sizes itself to cols × visible rows.
- **Reading direction fixed:** rotation is now **−90°** (was +90°, which
  mirrored the feed axis inside each tile — neighbour slivers showed on the
  wrong side, content upside-down). Frame start now lands on each tile's left
  edge, so tiles 1..N read continuously like the physical strip; the offset
  indicator moved to the right edge accordingly.

### Offset & drift
- **Drift slider** (`frame_offset_modifier_mm`, ±1.00 mm/frame): frame N scans
  at `max(0, base + (N−1)·drift)` — position-based, so scanning frames 3–6
  still drifts by physical position. Applied identically in preview requests
  and `run_batch`. Persisted, shown in the sidebar status.
- **Offset slider** widened to 0–10.0 mm. **Preview DPI** is a dropdown
  (defaults to lowest).
- **Bug fixed — drift never reached hardware:** `_update_settings_from_ui`
  rebuilt `ScannerSettings` from scratch and silently wiped
  `frame_offset_modifier_mm` right before the batch was built. Now uses
  `dataclasses.replace` so *every* non-UI field survives UI edits (kills the
  whole omission class). Regression-tested at the `_on_scan → BatchRequest` level.
- **`subframe` always written** (including 0.0) — options latch on an open
  handle; a session scan must reset a previous frame's offset.
- **Negative offset: built, hardware-tested, REMOVED.** The previous-frame
  borrow (frame N at −x = frame N−1 at pitch−x) produced compounding
  mis-framing on the real scanner and was ripped out. Everything clamps ≥ 0;
  the strip-registration recovery is eject + reseat, not negative offsets.
  Do not re-propose.
- **Indicator is ABSOLUTE:** the cut band is the frame's effective offset from
  the left edge — never a delta against the previewed offset (that version
  confused on hardware; per-tile `previewed_offset` tracking deleted). Frames
  floored at 0 by negative drift pin the dashed line to the left edge so the
  slider visibly acts.

### Sidebar
- Autofocus + Auto-exposure moved directly below Depth, left-aligned
  (label-spanning rows).
- Gating status: AE is capability-gated (disabled + tooltip when unsupported);
  autofocus is **ungated** — always shown, silently skipped by the backend when
  the device lacks the option.

### Housekeeping
- `origin/main` merged (CLAUDE.md trim conflict resolved — kept main's slimming
  plus the branch-only SANE bullets and a new SANE-sessions invariant).
- Model names in docs say "Coolscan", not LS-50/LS-5000.

Tests all green: full pytest (2239 passed), `make lint`, `make type`. New suites:
`tests/scanners/test_sane_session.py`, plus updates across
`test_strip_preview_dialog.py`, `test_scan_worker.py`, `test_scan_sidebar.py`,
`test_sane_window.py`, `test_coolscan_ir.py`.

## Session 4 (2026-07-18 evening) — agent-driven hardware debugging

Live LS-50 battery (8 scans @ 400 dpi through the real `SaneBackend`, offline
cross-correlation analysis). Findings, all empirical:

- **Offset scale/direction verified 1:1 mm** through NegPy's full code path
  (requested 1/2/4 mm → measured 1.02/1.97/4.00 mm, content toward raster top).
- **The scan blacks out ~38.0 mm past the frame start on EVERY frame** —
  measured `offset + delivered ≈ 37.96–38.00 mm` for offsets 1–8 mm, mid-strip
  included. The earlier in-app "smudge" was this black overrun stretched by
  preview normalization; the earlier "last frame only" theory was wrong.
  → `_frame_extent_cap` now shortens **every** offset scan by
  `1 − offset/subframe_max`; re-verified live: capped scan = 509 lines,
  **0 black rows**.
- **Negative offsets are physically impossible.** The previous-frame borrow
  (frame N−1 @ pitch−x) delivered a 1.33 mm sliver + black — the device cannot
  scan across a frame boundary. Borrow removed for good; effective offsets
  floor at 0 (dialog, worker, backend); the preview pins the cut line at the
  tile edge for floored frames. Closed permanently — this is hardware truth,
  not a design choice.
- **Drift batch verified end-to-end** through the real `ScanWorker`
  (frames 2–3, base 1.0 + drift 0.5): frame 3 landed at **+2.03 mm** (expected
  +2.0), capped to 564 lines (expected 563), 0 black rows.
- Testing gotcha for future live runs: **suppress `worker.eject` when driving
  `run_batch` on real hardware** — a clean finish auto-ejects the strip.

## Next steps

1. **Commit sessions 3–4** in logical chunks (session seam / preview layout /
   offset+drift+cap / sidebar), then push.
2. **Session option hygiene:** always-write/reset `ae`, `infrared`, `autofocus`,
   and window geometry in `_scan_on_device` so held-handle scans can't inherit a
   previous frame's state (prerequisite for the roll workflow).
4. **Reply on PR #497** with the seam shape: `service.open_session(device_id)` →
   `session.scan()` per frame → `session.eject()`/`close()`.
5. Optional: gate the autofocus checkbox like AE (probe a usable `autofocus`
   option into `ScannerCapabilities`).
6. Docs: refresh `docs/COOLSCAN_SCANNING.md` + `docs/CHANGELOG.md` for drift,
   preview-DPI dropdown, grid layout, and the session seam.
