# CLAUDE.md

Guidance for Claude Code in this repository.

> **Keep this file current.** A change that alters stage order, the feature pattern or the commands updates it in the same change.

> **Keep the user docs current, in the same change.** `docs/USER_GUIDE.md` covers every panel and control and is **rendered in-app** — each panel's ⓘ opens the section marked `<!-- panel:<key> -->` — so a stale doc is a stale in-app guide. A control added, renamed, retired or given a new range or default belongs there. `docs/PIPELINE.md` covers what each stage does to the pixels: update it when stage order, the math, a mirrored constant or a default changes. Retiring a control deletes its prose.

> **Leave `docs/CHANGELOG.md` alone unless asked.** It is written per release, not per change.

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

All commands run through `uv run`; never invoke pytest/ruff/ty directly. Run `make format` before committing; if it reformats unrelated files, commit those as lint fixes.

## Architecture

NegPy is a film-negative processing desktop app (PyQt6 + WebGPU). Images flow through a multi-stage pipeline implemented twice — CPU (numpy/Numba) and GPU (WGSL via `wgpu`) — which must stay in numerical parity.

### Data model

`WorkspaceConfig` (`negpy/domain/models.py`) — frozen dataclass of per-feature configs, the single source of truth for an edit. Change via `dataclasses.replace`, never mutate. `to_dict`/`from_flat_dict` serialize to one **flat** key namespace, so a duplicate field name across sub-configs silently clobbers.

Edits persist in SQLite (`edits.db`, keyed by content hash), optionally mirrored to `.negpy` JSON sidecars next to sources. DB wins; a loaded sidecar is promoted into the DB (`negpy/services/assets/sidecar.py`, `session.py`).

On Windows, `desktop.py` checks data-folder access before importing app configuration. A blocked default folder raises a native dialog that suggests Local AppData and lets the user pick another; no data is copied. The choice is saved under Local AppData and outranks Documents; `NEGPY_USER_DIR` outranks the record.

**Migrations** (`negpy/domain/migrations.py`) — every legacy fixup for persisted configs lives here, not inline in `from_flat_dict`: `KEY_RENAMES` (renamed fields), `DROPPED_KEYS` (removed fields, dropped without the unknown-key warning), `RETIRED_EXPORT_FORMATS`, and `migrate_flat_config()` for value rewrites. Renaming or removing a config field, or retiring an enum value, means one entry here. Two exceptions stay in their dataclasses because they must run on *every* construction, not just on load: `ExposureConfig.__post_init__` (legacy grade → ISO R, `cast_removal` bool → strength) and the tuple-rehydrating `__post_init__`s. The module imports nothing from `models.py`, which imports it, so use string literals.

Migrations that rewrite *rows* rather than a config payload need a repository, so they live in `services/assets/migrations/` instead, one module per migration (e.g. `hash.py` for edits saved under a superseded content hash, `roll_fields.py` for card locks when a field becomes a roll default or a card splits). Most run once at startup from `desktop/main.py`.

**Composite membership** (`services/assets/composites.py`) — which files a stitch or an HDR merge is made of is a user decision that nothing in the files records, so it is stored per primary path and lives until the composite is dissolved, not until the file list changes. Every asset discovery re-attaches from it and drops the parts it consumed. `_persist_session` upserts, never rewrites: the open files are one folder, the store is all of them.

**Rolls** (`services/assets/rolls.py`) — a Roll is a named, openable group of frames and the Library's only unit: a recognized folder, or a virtual roll built by hand from whatever the Film Strip holds. Folders are an import mechanism, not a live view. A roll also carries its own Half Frame state (`half_frame_mode_by_roll`) and roll-wide defaults for the Film Mode, Calibration, Crop, Roll Analysis, Metering, Raw Decode, Lens, Flat Field and Metadata cards (`ROLL_DEFAULT_FIELDS`, one `(config section, fields)` pair per card, so a card is not tied to `ProcessConfig`; the Optics section drives Lens and Flat Field through `controls_panel._SECTION_CARDS`, one scope pair over both); a frame that diverges is *locked* on that card until an Apply pushes it back out (`AppController.apply_roll_cards_to_roll` / `_to_selected`). What a card holds is the roll's or the rig's — the film edge, the scanning lens, the light, the stock's White/Black Point — never one frame's own placement: the crop rectangle, the rotation, the easel movements and the capture frame number stay per-frame. Every section header carries the same Frame/Roll pair (`CollapsibleSection.set_scope_buttons`); a frame-level card has no roll default and reads Roll from `rolls.section_push`, a record of what a whole-roll apply put there, which never overlays onto another frame. A roll also holds **scenes** (`rolls.roll_scenes`): hand-picked groups keyed by unforked hash, one scene per frame, each with its own normalization baseline that Roll Analysis skips.

**Roll-scoped edit fork** (`services/assets/rolls.py`) — a photo shared by more than one Roll normally has one edit, the same wherever it is opened from. An explicit per-frame fork gives it an independent edit for one Roll alone, keyed the same way a half-frame scan's two halves already are: the content hash suffixed (`roll_edit_hash`, `#roll:<id>`), not a new column. `AppController._apply_roll_forks()` rewrites a discovered asset's hash to its fork on open; `load_or_promote()`'s `forked` flag keeps a fork from ever falling back to the shared edit's path-based recovery or sidecar. Triage marks and legacy-hash migration read the roll suffix back off (`unforked_hash`), since a keep/reject judgement and a superseded-hash carry-over both belong to the physical scan, not to one Roll's fork of it.

**Search by meaning** (`services/assets/semantic_model.py`, `embeddings.py`, `clip_tokenizer.py`) — opt-in CLIP (ONNX) search over one vector per frame in the `image_embeddings` table, keyed by `MODEL_VERSION` so a model swap leaves old vectors unread. Only the inference engine ships; the weights download on first use. `workers/embedding.py` indexes exactly like `ThumbnailWorker`.

**Gear catalog** (`features/metadata/gear_*.py`, `services/assets/gear_match.py`) — bundled reference gear plus the user's own. A newly imported roll's folder name is matched against it (shared words, and a delimiter-free run for abbreviations); more than one candidate for a field counts as no match.

### Pipeline

- **CPU**: `DarkroomEngine.process()` (`negpy/services/rendering/engine.py`) — base (geometry + normalization) → exposure (incl. dodge/burn) → clahe → lab → alt process → toning → crop → finish. The first four stages are cached per config-hash via `_run_stage()`; the rest run unconditionally. The alt-process stage (lith or cyanotype, never both) is B&W-only and off by default; when off, both engines skip it rather than run an identity pass.
- **GPU**: `GPUEngine` (`negpy/services/rendering/gpu_engine.py`) — the same logical stages as WGSL compute shaders from `negpy/features/<name>/shaders/`, with its own config-diff change detection.
- **Orchestration**: `ImageProcessor` (`image_processor.py`) tries GPU first and falls back to CPU. Export always runs full-res, with CPU stage caching off (`PipelineContext.cache_stages`). Linear DNG decode, CPU saturation and unsharp masking use row blocks to bound temporary storage. `PipelineContext` carries `scale_factor`, `process_mode`, `active_roi` and a `metrics` dict between stages.
- **Embedded lens correction** (`features/lens`) is a single-file decode step shared by preview and export: flat-field, lens warp, then sensor unmix and user geometry. Its independent distortion and CA settings and flat-field token belong to the source identity. It is disabled for composite setup, composite assembly and RGB+IR sources.
- **Source bakes** run before either engine, on the linear source: flat-field, sensor unmix and every defect repair (IR, detected specks, painted heal strokes). Both engines re-upload that source per frame, so a bake reaches them parity-free and needs no shader. Each bake folds a token into `source_hash` to invalidate the engine cache.
- **Working space**: scene-linear internally; the working OETF (Adobe RGB 1998 TRC — a pure 563/256 power, no linear segment) is the final engine step. Lab/toning compute CIELAB directly from linear, D65. Adobe RGB rather than a wide gamut because ProPhoto's imaginary primaries inflate chroma in the saturation and toning stages.

`docs/PIPELINE.md` describes each stage's behaviour and controls in depth.

### Feature pattern

Every feature lives in `negpy/features/<name>/`:

- `models.py` — frozen dataclass config with defaults
- `logic.py` — pure functions on numpy arrays
- `processor.py` — thin wrapper with `process(img, context) -> ImageBuffer`
- `shaders/<name>.wgsl` — optional GPU compute shader

One exception: `features/altprocess/` holds only `models.py`. Lith and cyanotype are mutually exclusive, so they share the Alternative Processes panel and one `AltProcessConfig`; their logic and shaders stay in `features/lith/` and `features/cyanotype/`.

`features/lens/warps.py` holds frozen lens models with `has_distortion`, `has_ca`, and
`remap(...)`, as defined by `LensWarp` in `models.py`. `logic.py` applies their maps in
row blocks. File readers are registered in `infrastructure/loaders/lens_metadata.py`.

### Desktop (MVC)

- `AppState` (`negpy/desktop/session.py`) — mutable session state
- `AppController` (`negpy/desktop/controller.py`) — single controller; all UI interactions call it; emits `config_updated` / `image_updated`
- Workers (`negpy/desktop/workers/`) — heavy work in QThread-backed objects, Qt-signal communication
- Source loaders (`negpy/infrastructure/loaders/`) own `load_bounded_preview(...)`. It returns an oriented RGB image within the requested long edge, or `None` when the loader cannot keep the decode bounded. Automatic thumbnails never use a full camera RAW demosaic. `LoaderFactory.estimate_linear_preview_prefetch_memory(...)` admits bounded cooperative decodes and conservatively small non-LibRaw decodes. Long LibRaw calls do not run as neighbor prefetch.
- Sidebars (`negpy/desktop/view/sidebar/<name>.py`) — one per feature, registered in `ControlsPanel`, synced on `config_updated`
- The Film Strip's thumbnail grid (`ThumbnailGridView` in `files.py`) owns click-driven selection itself — `mousePressEvent`/`mouseMoveEvent`/`mouseReleaseEvent` decide plain/Shift/Ctrl once, from the modifiers at press, and reapply that on every later stage. `QAbstractItemView`'s own selection handling recomputes independently at each stage instead, reading modifiers fresh each time, which is what made Shift/Ctrl-click erratic before this — don't call `setSelection`/rely on the base class's mouse handling for this view.
- **Shortcuts** (`negpy/desktop/view/shortcut_registry.py`) — `REGISTRY` is the single source of truth for every binding: one `ShortcutEntry(default_key, description, category)` per action id, dispatched through the matching entry in the action map in `keyboard_shortcuts.py`. It also feeds the shortcut editor, the `?` overlay and `tooltip_with_shortcut()`.
  **Any new user-facing toggle, tool or action gets a registry entry** — leave `default_key` empty rather than inventing a conflicting one. Check for collisions before picking: the same key on two actions makes Qt fire `activatedAmbiguously` and both go dead. `docs/KEYBOARD.md` is generated, so run `uv run python -m negpy.desktop.view.keyboard_doc` after a registry change. Copy that names a key reads it through `key_for`/`label_with_shortcut`, never as a literal.

## Adding a new feature

1. Create `negpy/features/<name>/` with `models.py`, `logic.py`, `processor.py`
2. Add a field to `WorkspaceConfig`; update `to_dict`/`from_flat_dict` (watch flat-namespace collisions)
3. Insert a `_run_stage(...)` call in `DarkroomEngine.process()`
4. For GPU: add a WGSL shader, wire it into `GPUEngine` (shader path + stage index + change detection), and add the feature's `shaders/` dir to `build.py` (`--add-data`)
5. Add a sidebar and register it in `ControlsPanel`, building every control from the factories in **UI conventions** below. Sections come from `widgets/collapsible.make_section`; mark the panel's `docs/USER_GUIDE.md` section with `<!-- panel:<key> -->` above the heading to get the ⓘ guide
6. If it adds a toggle/tool/action, add a `REGISTRY` entry in `shortcut_registry.py` plus its action-map entry in `keyboard_shortcuts.py`
7. Add unit tests; if the feature has both CPU and GPU paths, add a parity test (pattern: `test_gpu_curve_parity.py`)
8. Document it: the panel and its controls in `docs/USER_GUIDE.md`, the stage's behaviour and math in `docs/PIPELINE.md`

## UI conventions

**A new control reuses an existing one. It never introduces a new look.** Find the closest control already in the app, call the same factory with the same tokens, and copy nothing. A new size, colour, width, spacing value, button shape or toggle idiom needs the user's agreement first: the panels sit in one tab stack, so a private look is visible beside the shared one.

- **Controls come from a factory**, never a bare `QPushButton` + `setStyleSheet`. All live in `styles/templates.py`; `BaseSidebar._tool_toggle` and friends are thin wrappers. `tool_toggle` (icon-only or icon+label toggle), `labeled_toggle` (checkable), `labeled_action` (its one-shot twin; `primary=True` for the panel's one call to action), `icon_button` (icon-only action), `templates.field_label` (label beside a combo/entry), `templates.hint_label` (a line of help under a control), `section_subheader` (grouping), `CollapsibleSection` (a panel section, and the only reset affordance), `CompactSlider` (slider with a hidden spin readout). Booleans in a panel are toggle buttons; `QCheckBox` is for a list of options in a form.
- **Type**: four size tokens in `styles/theme.py` — `font_size_small` (12, caption/hint), `font_size_base` (13, body and the QSS global), `font_size_header` (14, section), `font_size_title` (16, dialog title), plus `font_size_display` for the wordmark and `font_size_micro` (9) for chart axes only. All in px; the sheet reads them as `@font_size_basepx`. Never a literal size in a stylesheet string or a `setPixelSize`.
- **Colour**: `text_primary` body, `text_secondary` secondary copy, `text_hint` captions and hints, `warn_amber` advisories, `error` errors and invalid input (not the accent, which means selected/armed), `status_success` a good state. `text_muted` is the **disabled** grey — 2.6:1 on the panel, so never on text a user has to read. Channel colours have a fill tier (`channel_red`) and a text tier (`channel_red_text`). Every other colour is a token too, read from QSS as `@name`; `tests/test_theme_tokens.py` fails on a literal hex outside a painter alpha wash.
- **Geometry**: `ICON_BUTTON_WIDTH`, `FIELD_LABEL_WIDTH`, `default_button_height()`, `SCAN_BUTTON_HEIGHT` (the Scan buttons only) and the `THEME.space_*` scale. A row that needs a width already has one. A panel body has no side inset of its own; the section card insets.
- **Slider metadata**: unit in `unit=` (`"%"`, `" st"` stops, `" R"` ISO-R points, `" px"` — space before a word, none before a symbol), never in the label; decimals from `step`/`precision`.
- **Dialogs**: a hand-rolled footer calls `templates.pin_dialog_default(default, *others)`, which pins Enter, opts the rest out of `autoDefault` and marks the one filled button; a dialog with buttons in its body passes `scope=self` once built; a `QDialogButtonBox` gets `pin_button_box(box)`. Verbs: **Cancel** before one action verb (OK for a plain form, else Apply, Save, Scan); **Close** alone on a view-only dialog. Delete confirmations go through `view/confirm.py`. Do not re-declare the dialog background; the sheet paints it. `tests/test_dialog_footers.py` walks every dialog.
- **Labels**: control names Title Case ("Toe Width", "Paper White"); a label beside a combo/entry sentence case ("Film stock", "Input gamma"). Same concept, same words in every panel, so grep for the words before writing a new label.
- **Copy**: American spelling in every user-visible string and in `docs/` (Color, gray, center). Buttons, menu items and dialog/message-box titles are Title Case; a message-box title is the feature name. An item that opens a dialog, a picker or a confirmation ends in `…` (the character, never three dots). The feature is "Flat Field". No emoji.
- **Tooltips**: every control gets one, through `wrap_tooltip()` so it wraps. A shortcut-bearing widget is tooltipped in `controls_panel.apply_shortcut_tooltips()` only; a local `setToolTip` there is overwritten.

If no existing control fits, say so and propose the addition. Do not ship a one-off.

## Style

- Use **ASD-STE100 Simplified Technical English**
- **Comments minimal.** Comment only non-obvious constraints the code can't express (a cache contract, an ordering requirement, a rejected-alternative trap). Never narrate what the next line does, restate the diff or justify a change to a reviewer. One dense line beats a paragraph; docstrings short and factual.
- **Write the state, not the change.** Code, comments and docs describe how the thing works now. Never "used to", "no longer" or a war story about the bug: the symptom belongs in the commit message, and only there. A constraint that exists *because* of a past bug is written as the rule ("detection runs once, upstream of both engines"), not as its history.
- **No measurements in comments or docs.** Keep out timings, frame rates, file sizes, speedups, error deltas and sample counts from a test run ("113→31 ms", "2.7% further right", "one 35mm night scan"): they are true of one machine and one file, and they rot silently. State the constraint instead ("this scan is cached because it is the slowest step in a preview"). Numbers belong in the commit message, a test assertion or a report. Exceptions: a value the code depends on (a threshold, a limit, a unit) and a documented control range.
- **Budget the prose.** An inline comment is 1–2 lines, a docstring 1–4. A new control gets **one** `docs/USER_GUIDE.md` bullet; a stage change gets at most a short paragraph in `docs/PIPELINE.md`. Over budget means the point is buried: cut, don't reformat. Skip emphasis and rhetoric (bold, "on purpose", "this is not an optimization", em-dash asides, punchlines).
- **Say it once.** A paragraph that states one constraint three ways buries it. A constraint shared by a function and its caller goes in the function that enforces it; the caller says nothing.
- **No notes to a reviewer.** A comment asking someone to confirm, verify or check something is stale the day it merges, and it survives the answer. Open questions and things still to test go in the PR description. `TODO` is for work the code needs, not a message to a person.
- **End on the constraint.** The last sentence of a comment, a docstring or a doc paragraph is a fact, not a conclusion about it ("the one question the pipeline cannot answer about itself", "this is what Esc is for").
