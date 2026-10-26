# CLAUDE.md

Guidance for Claude Code in this repository. It holds rules and traps the code does not show; read the code for how a module works.

> **Keep this file current.** A change that alters stage order, the feature pattern or the commands updates it in the same change.

> **Keep the user docs current, in the same change.** `docs/USER_GUIDE.md` covers every panel and control and is **rendered in-app**: each panel's ⓘ opens the section marked `<!-- panel:<key> -->`, so a stale doc is a stale in-app guide. Update it when a control is added, renamed, retired or given a new range or default. Retiring a control deletes its prose. **Writing** below sets its voice and budgets. `docs/PIPELINE.md` covers what each stage does to the pixels: update it when stage order, the math, a mirrored constant or a default changes.

> **Keep the translations current, in the same change.** Wrap every user-visible string in `tr` (`mark` for a lookup key). A change that adds or rewords display text runs `make translations` and fills every registered catalog. A `docs/USER_GUIDE.md` change runs `make translations`, updates the same panels in each `negpy/translations/USER_GUIDE.<locale>.md` and sets their `src:` hash. The `update-translations` skill has the steps; `tests/test_i18n.py` fails on a missing entry, a bare display string or a stale guide panel, and prints the fix. Translate a label for its use, not its dictionary entry: read the call site, because a homonym (Draw, Free, Stop, Order, Stock, Capture, Grain, Feather) has a wrong sense in every language, and a button is a short noun or verb, never a polite sentence.

> **Leave `docs/CHANGELOG.md` alone unless asked.** It is written per release, not per change.

## Commands

```bash
make run          # Launch the desktop app
make run-sim      # Launch with a simulated camera, Scanlight and film scanners
make all          # format + Rust checks + lint + type check + tests (run before committing)
make rust         # cargo fmt --check + clippy -D warnings + cargo test for negpy_core
make test         # pytest only
make lint         # ruff check
make type         # ty check (not mypy)
make format       # cargo fmt + ruff format + autofix
make translations # Extract translation sources and preserve existing translations

# Single test
uv run pytest tests/test_exposure_logic.py::test_name -v
```

All commands run through `uv run`; never invoke pytest/ruff/ty directly. `uv run` compiles `negpy_core` again after a Rust file changes (`tool.uv.cache-keys`). Run `make format` before committing; if it reformats unrelated files, commit those as lint fixes.

## Architecture

NegPy is a film-negative processing desktop app (PyQt6 + WebGPU). Images flow through a multi-stage pipeline implemented twice, CPU (numpy and `negpy_core`) and GPU (WGSL via `wgpu`), which must stay in numerical parity.

### Data model

`WorkspaceConfig` (`negpy/domain/models.py`) is a frozen dataclass of per-feature configs and the single source of truth for an edit. Change it via `dataclasses.replace`, never mutate. `to_dict`/`from_flat_dict` serialize to one **flat** key namespace, so a duplicate field name across sub-configs silently clobbers.

Edits persist in SQLite (`edits.db`, keyed by content hash), optionally mirrored to `.negpy` sidecars. The DB wins; a loaded sidecar is promoted into the DB.

**Migrations** (`negpy/domain/migrations.py`) hold every legacy fixup for persisted configs, never inline code in `from_flat_dict`. Renaming or removing a config field, or retiring an enum value, means one entry there (`KEY_RENAMES`, `DROPPED_KEYS`, `RETIRED_EXPORT_FORMATS` or `migrate_flat_config()`). Only a fixup that must run on every construction stays in a `__post_init__`. The module imports nothing from `models.py`, which imports it, so use string literals. Migrations that rewrite rows need a repository and live in `services/assets/migrations/`, one module each.

**Bump `BOUNDS_VERSION`** (`features/exposure/normalization.py`) in any change that alters measured bounds for unchanged settings, together with `THUMBNAIL_RENDER_VERSION`; `test_a_changed_bounds_analysis_bumps_bounds_version` fails on an unbumped one.

**Composite membership** (`services/assets/composites.py`): the files of a stitch or HDR merge are a user decision that no file records. It is stored per primary path and lives until the composite is dissolved. `_persist_session` upserts, never rewrites: the open files are one folder, the store is all of them.

**Rolls** (`services/assets/rolls.py`): a Roll is the Library's only unit; folders are an import mechanism, not a live view. A roll holds defaults for some cards (`ROLL_DEFAULT_FIELDS`, one `(config section, fields)` pair per card; `controls_panel._SECTION_CARDS` for a section that drives two cards). A frame that diverges is locked on that card until Roll pushes it back out. A card holds the roll's or the rig's values, never one frame's crop or placement: the crop and its settings, rotation, easel movements and capture frame number stay per-frame.

**Roll-scoped edit fork**: a photo in more than one Roll has one edit, unless a per-frame fork gives it its own edit for one Roll. A fork is the content hash suffixed `#roll:<id>` (`roll_edit_hash`), like a half-frame's `#N`. Triage marks and legacy-hash migration strip the suffix (`unforked_hash`), because they belong to the physical scan.

**Film edge print** (`services/film_edge/`): a stock's own marks, DX number and respool source (`prints_as`) are the `edge` object in `gear/film_stocks.json`, and each entry names its `source`. The `edge-print-from-photo` skill adds a stock from a photo of the film.

### negpy_core (Rust)

`negpy_core/` is a PyO3 extension (abi3-py313, maturin, a `uv` workspace member) that holds every CPU kernel. It is the source of truth for CPU pixels; the GPU shaders follow it through the parity tests.

- A kernel's Python entry point keeps its NumPy-style signature and calls into `negpy_core`. No Python implementation, runtime switch or fallback sits beside it.
- Core code in `src/<area>.rs` takes slices or `ndarray` views and plain numbers, never config. `src/py/<area>.rs` holds only the wrappers: dtype, shape and contiguity checks, `py::detach` around the work, errors as Python exceptions. No panic crosses into Python.
- `cargo test` builds without Python: PyO3 is behind the `python` feature, which maturin turns on.
- Pure Rust: no dependency compiles C or C++ (`deny.toml` bans `cc`, `cmake`, `pkg-config`, `bindgen`). Codecs that need C stay on their Python wheels. A new crate's license goes in `NOTICE.md`.
- No `unsafe` except SIMD, each with a reason. Names mirror the Python names.
- `mul_add` is one instruction only on a CPU with FMA; the release baseline has none, so a loop that uses it takes `#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]`. `mul_add` is exactly fused on every path, so the bits never depend on the CPU.
- Hot loops: an element-wise pass goes through `par_zip` (rayon blocks; `par_iter_mut().zip` makes a task per element), and its closure is `move`. A capture read through a reference can alias the output, so LLVM reloads it per element and does not vectorize.
- A float-to-int `as` cast saturates, and LLVM leaves such a loop scalar. Clamp first, then take the integer part as `convert.rs::trunc_bits` does (add 2^52, read the low bits).
- **Pinned output**: a kernel's output is deterministic and the same on every release OS. `tests/test_core_same_bits.py` (run on each OS by `ci.yml` job `native`) and the pins in `tests/test_core_*.py` hold output hashes (`tests/core_pin.py`). A change that moves the bits updates its pins in the same change and follows the `BOUNDS_VERSION` and `THUMBNAIL_RENDER_VERSION` rules. Direct oracles against one `cv2` or NumPy call stay where a kernel copies a library primitive; the `cv2` ones run on Linux x86-64 only (`tests/core_oracle.py`). Kernels that replaced NumPy code keep its arithmetic, so keep it when editing them: the linear percentile (virtual index in f64, interpolation in f32, the `t >= 0.5` branch from the upper value), float32 sums in NumPy's order, NEP 50 promotion (a Python float joins float32 math as float32), Python `round` ties to even. No fast-math. Threads only over independent outputs, or integer reductions, so the core count never changes a result. A reduction with a large accumulator takes one chunk per thread (rayon's `fold` makes one per split job) and enough work per chunk to repay clearing and merging it.
- `negpy_core` holds every CPU kernel; NegPy has no Numba and imports no `cv2` (`tests/test_no_cv2.py`). opencv-python is only a dev dependency: the tests use it as their oracle. Every binding runs its work through `py::detach(py, n, f)`, never `py.detach`: it releases the GIL, keeps work under 64K elements on the calling thread and enters the all-core pool once per call (`tests/test_core_ir.py` checks). A thread marked by `negpy_core.set_background_thread(True)` (the thumbnail refresh) enters a pool of a quarter of the cores (at least two) instead, so the live preview keeps the rest. The app has no single-core mode.
- Transcendental functions (`log10`, `exp`, `pow`, ...) in array loops come from `src/mathf.rs` (branch-free, so they vectorize; `libm` is too slow there), scalar ones from the `libm` crate. Both give the same bits on every CPU.
- A speed change keeps the output pins and is never slower with every core in use, as users run the app.

### Pipeline

- **CPU**: `DarkroomEngine.process()` (`negpy/services/rendering/engine.py`): base (geometry + normalization) → exposure (incl. dodge/burn) → clahe → lab → alt process → toning → crop → finish. The first four stages are cached per config-hash via `_run_stage()`. Base and exposure route on `process.path.render_path`: `PRINT` runs `features/exposure/`; `TRANSFER` and `POSITIVE` run `features/transparency/`. `features/exposure/` holds no slide branch and never imports `features/transparency/` (`tests/test_slide_separation.py`).
- **GPU**: `GPUEngine` (`gpu_engine.py`) runs the same stages as WGSL compute shaders from `negpy/features/<name>/shaders/`, with its own config-diff change detection. `ImageProcessor` tries GPU first and falls back to CPU. Export runs full-res with CPU stage caching off.
- **Source bakes** (flat-field, sensor unmix, light leak reduction, every defect repair) run on the linear source before either engine, so a bake needs no shader and has parity for free. Each bake folds a token into `source_hash` to invalidate the engine cache.
- **Working space**: scene-linear internally; the Adobe RGB 1998 TRC (a pure 563/256 power) is the final engine step. Lab/toning compute CIELAB directly from linear, D65. Adobe RGB, not a wide gamut, because ProPhoto's imaginary primaries inflate chroma in saturation and toning.

`docs/PIPELINE.md` describes each stage's behavior and controls in depth.

### Feature pattern

Every feature lives in `negpy/features/<name>/`:

- `models.py`: frozen dataclass config with defaults
- `logic.py`: pure functions on numpy arrays
- `processor.py`: thin wrapper with `process(img, context) -> ImageBuffer`
- `shaders/<name>.wgsl`: optional GPU compute shader

`features/transparency/` has no `models.py`: its controls are Exposure and Process fields, and `tests/test_slide_separation.py` classifies every new `ExposureConfig` field. `features/altprocess/` holds only the shared `AltProcessConfig` of the mutually exclusive lith, cyanotype and Sabattier stages. `features/leak/` has no `processor.py`: light leak reduction is a source bake (`bake.py`).

### Desktop (MVC)

- `AppController` (`desktop/controller.py`) is the single controller; all UI interactions call it, and it emits `config_updated` / `image_updated`. `AppState` (`desktop/session.py`) holds session state. Heavy work runs in QThread workers (`desktop/workers/`).
- **Internationalization** (`kernel/system/i18n.py`): the saved `ui_language` loads before desktop modules import, so module constants translate on import and a language change needs a restart. Keep enum values, settings keys, paths and user text untranslated; a translated choice keeps the stored value in its item data. Ship new catalogs and guides in `build.py` and package data.
- Automatic thumbnails never use a full camera RAW demosaic: loaders bound the decode in `load_bounded_preview(...)`.
- **Bump `THUMBNAIL_RENDER_VERSION`** (`services/assets/thumbnail_fingerprint.py`) in any change that alters rendered pixels for unchanged settings, a changed default included (`test_a_changed_default_bumps_the_render_version`). Compare against main, not against earlier commits on the branch: a branch bumps it at most once over main. A config section that never reaches the pixels belongs in `NON_PIXEL_SECTIONS`; an export-only or label field in a hashed section belongs in `_UNHASHED_FIELDS`.
- The Film Strip grid (`ThumbnailGridView` in `files.py`) owns click selection: it reads Shift/Ctrl once, at press. Never call `setSelection` or rely on the base class's mouse handling there, because `QAbstractItemView` reads fresh modifiers at each stage.
- **Shortcuts**: `REGISTRY` in `shortcut_registry.py` is the single source of truth for every binding, dispatched through the action map in `keyboard_shortcuts.py`. **Any new user-facing toggle, tool or action gets a registry entry.** Leave `default_key` empty instead of inventing a conflicting one: the same key on two actions makes Qt fire `activatedAmbiguously` and both go dead. Run `uv run python -m negpy.desktop.view.keyboard_doc` after a registry change (`docs/KEYBOARD.md` is generated). Copy that names a key reads it through `key_for`/`label_with_shortcut`, never as a literal. A shortcut-bearing tooltip comes from `tooltip_with_shortcut`: the action ids it hides in the tooltip drive Shortcut Hints (`view/shortcut_hints.py`).

## Adding a new feature

1. Create `negpy/features/<name>/` with `models.py`, `logic.py`, `processor.py`
2. Add a field to `WorkspaceConfig`; update `to_dict`/`from_flat_dict` (watch flat-namespace collisions)
3. Insert a `_run_stage(...)` call in `DarkroomEngine.process()`
4. For GPU: add a WGSL shader, wire it into `GPUEngine` (shader path + stage index + change detection), and add the feature's `shaders/` dir to `build.py` (`--add-data`)
5. Add a sidebar and register it in `ControlsPanel`, building every control from the factories in **UI conventions** below. Sections come from `widgets/collapsible.make_section`; mark the panel's `docs/USER_GUIDE.md` section with `<!-- panel:<key> -->` above the heading to get the ⓘ guide
6. If it adds a toggle/tool/action, add a `REGISTRY` entry in `shortcut_registry.py` plus its action-map entry in `keyboard_shortcuts.py`
7. Add unit tests; if the feature has both CPU and GPU paths, add a parity test (pattern: `test_gpu_curve_parity.py`)
8. Document it: the panel and its controls in `docs/USER_GUIDE.md`, the stage's behavior and math in `docs/PIPELINE.md`; translate the new strings and guide panel

## UI conventions

**A new control reuses an existing one. It never introduces a new look.** Call the factory of the closest existing control with the same tokens; copy nothing. A new size, color, width, spacing value, button shape or toggle idiom needs the user's agreement first: the panels share one tab stack, so a private look shows beside the shared one.

- **Controls come from a factory** in `styles/templates.py`, never a bare `QPushButton` + `setStyleSheet`.
  - `tool_toggle`: icon-only or icon+label toggle. `clearable_tool_toggle`: one whose tool leaves a saved result, with an inline × that clears it.
  - `labeled_toggle` (checkable) and `labeled_action`, its one-shot twin; `primary=True` for the panel's one call to action. A factory puts the gap between icon and label (`icon_text`), so a label never starts with a space.
  - `icon_button`: icon-only action. `header_button`: a section header's small flat icon.
  - `gradient_icon(name, sweep)`: for the one icon in a row that must stand out. Every sweep is a named constant beside `RGB_SWEEP` with one meaning; `clipping_sweep` takes the Color vision pair, as the Clipping Overlay does.
  - `field_label` / `field_row`: a label beside a combo or entry. `hint_label`: a line of help under a control. `section_subheader` and `header_row` for grouping.
  - `CollapsibleSection`: a panel section, and the only reset affordance.
  - `CompactSlider`: one slider per row, never two side by side.
  - `SliderGroup`: a rail under the control its members depend on (Toe Width under Toe, or the settings a toggle above turns on). The main control stays outside the rail; peers of equal rank get no rail.
  - Folds (`widgets/fold.py`) keep fine-tune controls out of the first view: `rail_fold` puts a rail behind a chevron on its parent's row, `group_fold` a group behind its sub-header. A fold starts closed (`default_open` for the few that start open) and remembers the user's state per key. Gate the fold's `content`, never the fold. A folded rail whose members do nothing while the parent is at neutral stays and grays out (Separation Damping at Dye Separation 1.0). An unfolded one shows only while the parent acts (`show_while`, Vignette Size at Burn 0).
  - `SegmentedChoice` (two to four short, fixed choices in view) and `StepCombo` (a longer list, a device-decided list, or a setting that need not stay in view) for every mutually exclusive choice, never a row of exclusive toggles and never radio buttons (`tests/test_no_radio_buttons.py`).
  - `StepCombo` for every dropdown (`StepCombo.of(labels, tooltip, data)`), never a bare `QComboBox`; an editable combo stays a `QComboBox`. A `MenuButton` opens a menu of actions, not a choice.
  - Every on/off choice is a toggle (`labeled_toggle`, or `tool_toggle` with `align_left` for a list of options), in a panel, a dialog or a list alike. The app has no checkbox: `templates.box_toggle` is the one `QCheckBox`, drawn as a toggle, because `QMessageBox.setCheckBox` takes nothing else (`tests/test_toggle_look.py`).
- **Type**: the `font_size_*` tokens in `styles/theme.py`, read in QSS as `@font_size_basepx`. Never a literal size in a stylesheet string or a `setPixelSize`.
- **Color**: `text_primary` body, `text_secondary` secondary copy, `text_hint` captions, `warn_amber` advisories, `error_text` error copy, `error` invalid input and error marks, `status_success` a good state. `text_muted` is the **disabled** gray, never on text a user has to read. Every color is a token (`tests/test_theme_tokens.py`). A view that tells marks apart by hue alone reads them from `styles/color_vision.py` (`palette_for(state.color_vision)`) and repaints on `session.color_vision_changed`.
- **Selection**: a chosen item (checked toggle, segment, tab, current row, checked menu item) takes the warm chip, `bg_checked` with a `border_checked` edge (`_hover` pair on hover). An icon that is on is `text_primary`, off `text_hint`, disabled `text_muted`. A selected frame on the Film Strip or the Light Table takes a `border_checked` mat over its border (`border_checked_hover` on the open frame). Hover, the arrow-key row and keyboard focus (`border_focus`) are neutral. Nothing is bright red but an error and the red channel (`tests/test_toggle_look.py`): a mark on the photo is `mark_photo` white on a `mark_shadow` edge or plate (`templates.with_alpha` for a translucent token), a primary action is `bg_primary` with a `border_checked_hover` edge (its hover a lighter fill), and an edited control's dot is `mark_edited`. A button at rest sits on the track surface (`bg_input`, `border_primary` edge), a sub-header's icons (`header_row`, `group_fold`) included; only a card header's own icons (`header_button`) and the canvas bar's buttons are flat (`setFlat`), the bar's in one track per group. A `QToolButton` outside the bar takes the `boxed` property, because the plain one is flat.
- **Geometry**: `ICON_BUTTON_WIDTH`, `FIELD_LABEL_WIDTH`, `default_button_height()` and the `THEME.space_*` scale; the canvas bar's controls take `BAR_BUTTON_HEIGHT` and `BAR_ICON_SIZE`. A row that needs a width already has one. A panel body has no side inset of its own; the section card insets.
- **Slider metadata**: unit in `unit=` (a space before a word, none before a symbol), never in the label; decimals from `step`/`precision`.
- **Dialogs**: a footer calls `templates.pin_dialog_default(default, *others)` (`pin_button_box` for a `QDialogButtonBox`). Verbs: **Cancel** before one action verb (OK for a plain form, else Apply, Save, Scan); **Close** alone on a view-only dialog. Delete confirmations go through `view/confirm.py`. A resizable dialog takes a keyword-only `repo` and ends `__init__` with `remember_dialog_geometry(self, repo, "<name>")`; a fixed-size one calls `setFixedWidth`/`setFixedSize`. `tests/test_dialog_footers.py` and `tests/test_dialog_geometry.py` walk every dialog.
- **Labels**: control names in Title Case ("Toe Width"); a label beside a combo/entry in sentence case ("Film stock"). The same concept takes the same words in every panel, so grep for the words before you write a new label.
- **Copy**: every user-visible string goes through `tr`. American spelling in strings and `docs/`. Buttons, menu items and dialog titles are Title Case; a message-box title is the feature name. An item that opens a dialog, picker or confirmation ends in `…` (the character; `tests/test_ui_copy.py`). The feature is "Flat Field". No emoji.
- **Tooltips**: every control gets one, within the budget under **Writing**. A shortcut-bearing widget is tooltipped in `controls_panel.apply_shortcut_tooltips()` only; a local `setToolTip` there is overwritten.
- **Hide or gray out**: hide a single control the frame's film or mode can never use. Gray out a group or a whole panel the mode cannot use, under one `hint_label` that says why. Gray out a control that waits on a toggle or resource on the same frame, or a folded rail that waits on its parent; an unfolded rail that waits on its parent hides instead (`show_while`).

If no existing control fits, say so and propose the addition. Do not ship a one-off.

## Writing

All prose (UI copy, docs, comments, commit messages) is **ASD-STE100 Simplified Technical English**:

- One topic per sentence: at most 20 words in an instruction, 25 in a description.
- Active voice, present tense. An instruction is a command ("Click Apply"), not a suggestion ("you may want to click Apply").
- Simple words with one meaning: "use", not "utilize"; "start", not "initiate". One thing keeps one word everywhere; grep before you name it.
- Keep the articles. No idioms, slang or contractions in UI copy and docs.

### Text the user reads

UI copy, tooltips, `docs/USER_GUIDE.md` and the tour (`widgets/tutorial_steps.py`) are for a photographer who does not know darkroom terms.

- Say what a control does in plain words first, the darkroom name second ("Print Density: how light or dark the print is, like the exposure time under an enlarger"). Internals (normalization, matrices, log densities, cache keys) belong in `docs/PIPELINE.md`.
- Budgets, each enforced by a test; over budget means cut, not reformat:
  - a tooltip is one sentence of at most 25 words, with no ranges (`tests/test_tooltip_budget.py`); the detail goes in the guide;
  - a guide panel stays under 350 words (`tests/test_section_help_dialog.py`): one sentence on what the card is for, then one bullet per control, and fine-tune controls in one fold may share a bullet;
  - a tour step stays under 40 words (`tests/test_tutorial_overlay.py`).
- Bold only the control name a bullet defines. Edge cases and rare states stay out of the guide unless a user acts on them.
- A label is a noun or a short verb phrase, never a sentence. A message says what happened, then what to do.
- No machine-sounding prose. Use a period, colon or comma, never an em-dash aside (`tests/test_ui_copy.py`). No "not X, but Y", no lists of three for rhythm, no filler adjectives ("seamless", "powerful", "robust"), no closing summary.

### Code comments and developer docs

- **Comments minimal.** Comment only non-obvious constraints the code cannot express (a cache contract, an ordering requirement, a rejected-alternative trap). Never narrate the next line, restate the diff or justify a change to a reviewer. One dense line beats a paragraph; docstrings short and factual.
- **Write the state, not the change.** Code, comments and docs describe how the thing works now. Never "used to", "no longer" or a war story about the bug: the symptom belongs in the commit message, and only there. Write a constraint that exists *because* of a past bug as the rule ("detection runs once, upstream of both engines"), not as its history.
- **No measurements in comments or docs.** No timings, frame rates, file sizes, speedups, error deltas or sample counts from a test run ("113→31 ms", "2.7% further right", "one 35mm night scan"): they are true of one machine and one file, and rot silently. State the constraint instead ("this scan is cached because it is the slowest step in a preview"). Numbers belong in the commit message, a test assertion or a report. Exceptions: a value the code depends on (a threshold, a limit, a unit) and a documented control range.
- **Budget the prose.** An inline comment is 1 or 2 lines, a docstring 1 to 4. A stage change gets at most a short paragraph in `docs/PIPELINE.md`, which keeps the math, mirrored constants and ordering rules and leaves UI behavior to the guide. Over budget means the point is buried: cut, do not reformat. Skip emphasis and rhetoric (bold, "on purpose", "this is not an optimization", em-dash asides, punchlines).
- **Say it once.** A paragraph that states one constraint three ways buries it. A constraint shared by a function and its caller goes in the function that enforces it; the caller says nothing.
- **No notes to a reviewer.** A comment asking someone to confirm, verify or check something is stale the day it merges, and outlives the answer. Open questions and things still to test go in the PR description. `TODO` is for work the code needs, not a message to a person.
- **End on the constraint.** The last sentence of a comment, a docstring or a doc paragraph is a fact, not a conclusion about it ("the one question the pipeline cannot answer about itself", "this is what Esc is for").
