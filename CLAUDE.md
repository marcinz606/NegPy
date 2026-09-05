# CLAUDE.md

Guidance for Claude Code in this repository.

> **Keep this file current.** When a change alters something documented here — stage order, the feature pattern, commands — update it in the same change.

> **Keep the user docs current, in the same change.** `docs/USER_GUIDE.md` covers every panel and control; it is also **rendered in-app** — each panel's ⓘ opens the section marked `<!-- panel:<key> -->` — so a stale doc is a stale in-app guide. Any control added, renamed, retired or given a new range/default belongs there. `docs/PIPELINE.md` covers what each stage does to the pixels: update it when stage order, the math, a mirrored constant or a default changes. Retiring a control means deleting its prose, not leaving it to rot.

> **Leave `docs/CHANGELOG.md` alone unless asked.** It is written per release, not per change. Do not add, edit or move an entry unless the user asks for it.

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

All commands run through `uv run`; never invoke pytest/ruff/ty directly.
Before commiting always run `make format`. If any non-related files got re-formatted by ruff - also commit the changes as lint fixes.

## Architecture

NegPy is a film-negative processing desktop app (PyQt6 + WebGPU). Images flow through a multi-stage pipeline implemented twice — CPU (numpy/Numba) and GPU (WGSL via `wgpu`) — which must stay in numerical parity.

### Data model

`WorkspaceConfig` (`negpy/domain/models.py`) — frozen dataclass composed of per-feature configs; the single source of truth for an edit. Change via `dataclasses.replace`, never mutate. `to_dict`/`from_flat_dict` serialize to one **flat** key namespace — a duplicate field name across sub-configs silently clobbers.

Edits persist in SQLite (`edits.db`, keyed by content hash), optionally mirrored to `.negpy` JSON sidecars next to sources. DB wins; a loaded sidecar is promoted into the DB (`negpy/services/assets/sidecar.py`, `session.py`).

On Windows, `desktop.py` runs data-folder recovery before importing app configuration. An approved recovery copies persistent data, backs up SQLite databases, and commits a location record under Local AppData. The record takes priority over Documents; `NEGPY_USER_DIR` takes priority over the record.

**Migrations** (`negpy/domain/migrations.py`) — every legacy fixup for persisted configs lives here, not inline in `from_flat_dict`: `KEY_RENAMES` (renamed fields), `DROPPED_KEYS` (removed fields, dropped without the unknown-key warning), `RETIRED_EXPORT_FORMATS`, and `migrate_flat_config()` for value rewrites. Renaming/removing a config field or retiring an enum value means one entry here. Two exceptions stay in their dataclasses because they must run on *every* construction, not just on load: `ExposureConfig.__post_init__` (legacy grade → ISO R, `cast_removal` bool → strength) and the tuple-rehydrating `__post_init__`s. The module imports nothing from `models.py` (which imports it) — use string literals.

Migrations that rewrite *rows* rather than a config payload need a repository, so they stay out of that dependency-free module and live beside their feature: `services/assets/hash_migration.py` (edits saved under a superseded content hash — see the identity note below) and `services/assets/flatfield_migration.py` (the retired profile table).

**Composite membership** (`services/assets/composites.py`) — which files a stitch or an HDR merge is made of is a user decision that nothing in the files records, so it is stored per primary path and lives until the composite is dissolved, not until the file list changes. Every asset discovery re-attaches from it and drops the parts it consumed; `_persist_session` upserts into it, never rewrites it, because the open files are one folder and the store is all of them.

### Pipeline

- **CPU**: `DarkroomEngine.process()` (`negpy/services/rendering/engine.py`) — base (geometry + normalization) → exposure (incl. dodge/burn) → clahe → lab → alt process → toning → crop → finish. The first four stages are cached per config-hash via `_run_stage()`; the rest run unconditionally. The alt-process stage (lith or cyanotype, never both) is B&W-only and off by default — when off, both engines skip it rather than run an identity pass.
- **GPU**: `GPUEngine` (`negpy/services/rendering/gpu_engine.py`) — same logical stages as WGSL compute shaders from `negpy/features/<name>/shaders/`, with its own config-diff change detection.
- **Orchestration**: `ImageProcessor` (`image_processor.py`) tries GPU first, falls back to CPU; export always runs full-res. `PipelineContext` carries `scale_factor`, `process_mode`, `active_roi`, and a `metrics` dict between stages.
- **Source bakes** run before either engine, on the linear source: flat-field, sensor unmix, and every defect repair (IR, detected specks, painted heal strokes). Both engines re-upload that source per frame, so a bake reaches them parity-free and needs no shader. Each bake folds a token into `source_hash` to invalidate the engine cache.
- **Working space**: scene-linear internally; the working OETF (Adobe RGB 1998 TRC — a pure 563/256 power, no linear segment) is applied only as the final engine step. Lab/toning compute CIELAB directly from linear, D65. Adobe RGB rather than a wide gamut because ProPhoto's imaginary primaries inflate chroma in the saturation/toning stages.

`docs/PIPELINE.md` describes each stage's behaviour and controls in depth.

### Feature pattern

Every feature lives in `negpy/features/<name>/`:

- `models.py` — frozen dataclass config with defaults
- `logic.py` — pure functions on numpy arrays
- `processor.py` — thin wrapper with `process(img, context) -> ImageBuffer`
- `shaders/<name>.wgsl` — optional GPU compute shader

One exception: `features/altprocess/` holds only `models.py`. Lith and cyanotype share the
Alternative Processes panel and one `AltProcessConfig`, because they are mutually exclusive;
their logic and shaders stay in `features/lith/` and `features/cyanotype/`.

### Desktop (MVC)

- `AppState` (`negpy/desktop/session.py`) — mutable session state
- `AppController` (`negpy/desktop/controller.py`) — single controller; all UI interactions call it; emits `config_updated` / `image_updated`
- Workers (`negpy/desktop/workers/`) — heavy work in QThread-backed objects, Qt-signal communication
- Sidebars (`negpy/desktop/view/sidebar/<name>.py`) — one per feature, registered in `ControlsPanel`, synced on `config_updated`
- **Shortcuts** (`negpy/desktop/view/shortcut_registry.py`) — `REGISTRY` is the single source of truth for every binding: one `ShortcutEntry(default_key, description, category)` per action id. Dispatch is the matching entry in the action map in `keyboard_shortcuts.py`. The registry also feeds the shortcut editor, the `?` overlay and `tooltip_with_shortcut()`, so a binding added here shows up in all three for free.
  **Any new user-facing toggle, tool or action gets a registry entry** — leave `default_key` empty rather than inventing a conflicting one if no obvious key is free. Check for collisions before picking: the same key on two actions makes Qt fire `activatedAmbiguously` and both go dead.

## Adding a new feature

1. Create `negpy/features/<name>/` with `models.py`, `logic.py`, `processor.py`
2. Add a field to `WorkspaceConfig`; update `to_dict`/`from_flat_dict` (watch flat-namespace collisions)
3. Insert a `_run_stage(...)` call in `DarkroomEngine.process()`
4. For GPU: add a WGSL shader, wire it into `GPUEngine` (shader path + stage index + change detection), and add the feature's `shaders/` dir to `build.py` (`--add-data`)
5. Add a sidebar and register it in `ControlsPanel`, building every control from the factories in **UI conventions** below; mark its `docs/USER_GUIDE.md` section with `<!-- panel:<key> -->` above the heading (`<key>` = the `_make_section` key) — that marker is what puts the ⓘ guide on the header
6. If it adds a toggle/tool/action, add a `REGISTRY` entry in `shortcut_registry.py` plus its action-map entry in `keyboard_shortcuts.py`
7. Add unit tests; if the feature has both CPU and GPU paths, add a parity test (pattern: `test_gpu_curve_parity.py`)
8. Document it: the panel and its controls in `docs/USER_GUIDE.md`, the stage's behaviour and math in `docs/PIPELINE.md`

## UI conventions

**A new control reuses an existing one. It never introduces a new look.** Find the closest
control already in the app, call the same factory with the same tokens, and copy nothing. A
new size, colour, width, spacing value, button shape or toggle idiom needs the user's
agreement first — the panels sit in one tab stack, so a private look is visible beside the
shared one.

- **Controls come from a factory**, never from a bare `QPushButton` + `setStyleSheet`:
  `BaseSidebar._tool_toggle` (icon-only or icon+label toggle), `_labeled_toggle` (checkable,
  carries an `edited_dot`), `_labeled_action` (its one-shot twin), `templates.icon_button` /
  `_icon_action` (icon-only action), `templates.field_label` (label beside a combo/entry),
  `templates.hint_label` (a line of help under a control), `section_subheader` (grouping),
  `CollapsibleSection` (a panel section, and the only reset affordance), `CompactSlider`
  (slider with a hidden spin readout). Booleans in a panel are toggle buttons; `QCheckBox` is
  for a list of options in a form.
- **Type**: four size tokens in `styles/theme.py` — `font_size_small` (12, caption/hint),
  `font_size_base` (13, body and the QSS global), `font_size_header` (14, section),
  `font_size_title` (16, dialog title), plus `font_size_display` for the wordmark. All in px;
  the sheet reads them as `@font_size_basepx`. Never a literal size in a stylesheet string.
- **Colour**: `text_primary` body, `text_secondary` secondary copy, `text_hint` captions and
  hints, `warn_amber` advisories, `channel_red` errors. `text_muted` is the **disabled** grey
  — 2.6:1 on the panel, so never on text a user has to read. Every other colour is a token in
  `theme.py` too; a literal hex in a widget is a bug.
- **Geometry**: `ICON_BUTTON_WIDTH`, `FIELD_LABEL_WIDTH`, `default_button_height()` and the
  `THEME.space_*` scale. A row that needs a width already has one.
- **Slider metadata**: unit in `unit=` (`"%"`, `" st"`, `" px"` — space before a word, none
  before a symbol), never in the label; decimals from `step`/`precision`.
- **Dialogs**: a hand-rolled footer calls `templates.pin_dialog_default(default, *others)` —
  it pins Enter, opts the rest out of `autoDefault` and marks the one filled button. Cancel
  sits before the action. Do not re-declare the dialog background; the sheet paints it.
- **Labels**: control names Title Case ("Toe Width", "Paper White"); a label beside a
  combo/entry sentence case ("Film stock", "Input gamma"). Same concept, same words in every
  panel — grep for the words before writing a new label.
- **Tooltips**: every control gets one, through `wrap_tooltip()` so it wraps. A shortcut-bearing
  widget is tooltipped in `controls_panel.apply_shortcut_tooltips()` only; a local `setToolTip`
  there is overwritten.

If no existing control fits, say so and propose the addition — do not ship a one-off.

## Style

- Use **ASD-STE100 Simplified Technical English**
- **Comments minimal.** Comment only non-obvious constraints the code can't express (a cache contract, an ordering requirement, a rejected-alternative trap). No comments that narrate what the next line does, restate the diff, or justify a change to a reviewer. Prefer one dense line over a paragraph; docstrings short and factual.
- **Write the state, not the change.** Code, comments and docs describe how the thing works now. Never "used to", "no longer", "this used to re-detect per render", or a war story about the bug. The symptom belongs in the commit message, and only there. A constraint that exists *because* of a past bug is written as the rule ("detection runs once, upstream of both engines"), not as its history.
- **No measurements in comments or docs.** Keep out timings, frame rates, file sizes, speedups, error deltas and sample counts from a test run ("113→31 ms", "2.7% further right", "one 35mm night scan"). They are true of one machine and one file, and they rot silently. State the constraint instead ("this scan is cached because it is the slowest step in a preview"). Numbers belong in the commit message, a test assertion or a report. Exceptions: a value the code depends on (a threshold, a limit, a unit) and a documented control range.
- **Budget the prose.** An inline comment is 1–2 lines, a docstring 1–4. A new control gets **one** `docs/USER_GUIDE.md` bullet; a stage change gets at most a short paragraph in `docs/PIPELINE.md`. Over budget means the point is buried: cut, don't reformat. Skip emphasis and rhetoric (bold, "on purpose", "this is not an optimization", em-dash asides, punchlines).
