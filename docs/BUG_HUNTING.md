# Bug hunting

Open, unconfirmed leads worth a proper look later — not user-facing, not kept in
sync with the code the way `USER_GUIDE.md`/`PIPELINE.md` are. Each entry states what
is confirmed by reading the code versus what is still a guess, so whoever picks it
up isn't starting from zero. An entry marked **Fixed** stays here as the record of
what the symptom actually was and why, since none of this is written anywhere else.

## Fixed: stale splash preview overwrites an already-correct render

**Reported, case 1:** navigate to a frame right after using Apply to All Roll
(Calibration, in the report) and it renders with a strong wrong color cast
("weirdly red"). **Case 2:** a frame's preview shows inverted (negative-looking, not
positivized); clicking the control meant to fix that ("deinverts" — likely Positive
or Linear RAW) changes something but still looks wrong, and only navigating away and
back (no further edit) shows it correctly. Frequency assessment moved three times on
re-inspection: "very rare," then "more normal than I thought," then — specifically
for negatives (Color/B&W), as opposed to Slide — **"almost every time there is a
negative there, so common."**

**Root cause:** `AppController._on_splash_preview` (`negpy/desktop/controller.py`)
paints the file's embedded camera/scanner JPEG thumbnail immediately, before the
real decode+pipeline render is ready, to avoid a blank canvas — "the real render
replaces it" per its own comment. But it had no guard against the reverse: if the
real render for that same file lands *first* and the splash arrives after (a
backlogged splash-decode worker, or a neighbour whose full pipeline finished from an
earlier prefetch before its own splash request reached the front of the queue),
`_on_splash_preview` painted over the already-correct positive with the raw,
un-inverted embedded thumbnail anyway — nothing checked whether a real render had
already landed. An un-inverted color negative reads as a strong orange-masked cast
("weirdly red"); an un-inverted image in general just looks like a negative
("inverted"). This is invisible on Slide, where the source is already close to
positive, and glaring on a negative — exactly the frequency split reported.

Case 2's "still looking weird" after clicking Positive/Linear RAW fits the same
mechanism: either of those changes `effective_linear_raw`, part of
`source_token()`, so `apply_config` treats it as `needs_decode` and calls
`load_file()` again — re-running the entire splash → decode → render sequence, and
so re-opening the same race a second time on the same click.

**Fix:** `_on_splash_preview` now checks, under `state.metrics_lock`, whether
`last_metrics["splash"]` is already `False` and `last_metrics["source_hash"]`
already equals this file's hash (i.e. `_on_render_finished` already delivered the
real render for *this exact file*) before painting, and skips the paint if so. Keyed
on the file, not just "a render happened," so a legitimate splash for a freshly
opened frame is never swallowed by the frame just left.

Tests: `tests/test_controller.py::TestSplashPreviewRaceGuard`.

### Other mechanisms noted during the investigation, not ruled out

These were considered as alternative or contributing explanations before the splash
race was confirmed as sufficient to explain both reported cases. Left here in case
the symptom recurs after this fix and one of these turns out to matter too.

- **Navigate-back render memo** (`AppController.load_file`, ~1852–1902, and
  `_render_memo_key`, ~1770–1790) paints a memoized buffer for the target hash
  immediately on frame switch if `_render_memo.get(target_hash,
  _render_memo_key())` hits. The key is computed from the already-hydrated
  `self.state.config` (roll defaults included), so a stale hit should require the
  exact same config to have been memoized earlier under that key — not ruled out,
  but no longer needed to explain the reported symptom.
- **Neighbor prefetch** (`AppController._schedule_prefetch_neighbors`, ~2063–2105)
  keys its warm decode on each neighbour's own **raw saved settings**
  (`repo.load_file_settings`), not the roll-resolved effective config — a gap in
  principle, but the `PreviewLoadTask` it builds doesn't carry Crosstalk/sensor
  fields at all, so it can't by itself produce the reported color cast.
- `DarkroomEngine._run_stage`'s per-config-hash cache (CPU) and `GPUEngine`'s own
  config-diff change detection were never read with this race in mind — still open
  if a related symptom shows up that the splash-race fix doesn't cover.

## Open: GPU's transfer-path "final_bounds" metric is the wrong bounds

Found while wiring White/Black Point into the transfer path (`transfer.py`'s fixed
window). **Confirmed by reading the code:** `GPUEngine.process_to_texture`
(`negpy/services/rendering/gpu_engine.py`, ~829-833) computes the `final_bounds`
metric it publishes from `resolve_bounds_detailed()` — the *measured* path's bounds
— unconditionally, even when `is_transfer_path()` is true and the render itself uses
`transfer_bounds()` (folded with WP/BP) instead. The CPU engine's equivalent metric
(`NormalizationProcessor._process_transparency`, `processor.py`) is correct: it
publishes the bounds actually used. This only affects a diagnostic value read by
metrics/histogram panels, not a pixel the render draws — the GPU shader itself
already uses the right window. Not fixed here because it is a metrics-accuracy gap
on a path this change didn't touch, not a rendering bug; worth a proper look with
its own test.

## Open: WGSL `is_transfer` flag does not check `positive_source`

**Confirmed by reading the code:** `normalization.wgsl`'s `is_transfer` (line ~43) is
`is_e6 && params.normalize_flag == 0u` — it takes the transfer path only for
Slide-without-Normalize, unlike Python's `is_transfer_path()`, which also takes it
for any mode with `positive_source` true. **Guess, not verified:** likely benign,
since a positive-source capture has no `cam_xyz` to fold (`camera_to_working_matrix`
returns `None`), so the shader's camera-matrix step is already an identity for that
case regardless of which flag gates it. Not confirmed against an actual GPU render
of a Positive frame in Color or B&W — worth a parity test alongside
`test_positive_source_matches` if this path is touched again.
