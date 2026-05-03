# NegPy UX/Design Polish Plan

## Context

Audit of `negpy/desktop/view/` surfaced consistent gaps between an otherwise solid foundation (centralized `ThemeConfig`, custom `CompactSlider`/`CollapsibleSection`, dock-based shell) and shipped polish:

- **61 inline `setStyleSheet()` calls** across 20+ files leak past the theme system.
- **Out-of-palette colors** (`#2e7d32` green at `session_panel.py:54`, hardcoded RGB hexes in `widgets/charts.py:131-133`, `#2a2a2a` in `widgets/sliders.py:315-327`).
- **Missing token scales** for spacing, radius, opacity surfaces, and font weight; `font_size_base` and `font_size_small` are duplicates; toast `15px` is a scale outlier.
- **Hardcoded font-size strings** (`11px`/`12px`/`13px`) scattered through canvas + sidebars.
- **Sidebar inconsistencies**: Lab/Toning have no section headers; abbreviated labels (`S-Hue`/`S-Str`); Process roll layout cramps 8 controls without grouping; Retouch row is awkward; CollapsibleSection emits 4 redundant "modified" signals; double-click reset is undiscoverable.
- **Canvas/toolbar friction**: 16 controls in one row with no responsive overflow; weak active-tool indicator; thin 1px progress bar; 24px metadata panel cramped; canvas context menu shortcut strings hardcoded (drift vs `shortcut_registry`); zoom slider/wheel desync at `widget.py:234`.

Decided constraints (user): keep current dark + Safelight Red identity. Scope = theme, sidebar, canvas — slider/widget behaviours untouched per explicit ask. Appetite = polish + token consolidation, ~3 days. No widget rebuilds, no light mode, no architecture changes — view layer only.

## Plan

### Phase 1 — Token system extension (Day 1 AM)

Edit `negpy/desktop/view/styles/theme.py`. Extend `ThemeConfig` with:

- **Spacing scale** (px ints): `space_xs=2`, `space_sm=4`, `space_md=6`, `space_lg=8`, `space_xl=12`, `space_2xl=16`.
- **Radius scale**: `radius_sm=3`, `radius_md=4`, `radius_lg=6`.
- **Surface overlay tokens** (rgba strings, derived from `bg_dark`): `surface_overlay="rgba(13,13,13,0.88)"`, `surface_overlay_strong="rgba(26,26,26,0.82)"`, `surface_overlay_hover="rgba(34,34,34,0.88)"`.
- **Font-weight scale**: `weight_regular=400`, `weight_medium=500`, `weight_semibold=600`, `weight_bold=700`.
- **Font sizes deduped + extended**: keep `font_size_base=12` (alias `font_size_small`), add `font_size_xs=11` (status/captions), `font_size_lg=15` (toast).
- **Channel colors** (domain): `channel_red="#D32F2F"`, `channel_green="#388E3C"`, `channel_blue="#1976D2"` — replaces hex literals in `widgets/charts.py:131-133`.
- **Canvas swatch tokens**: `canvas_bg_black="#050505"`, `canvas_bg_dark_grey="#1C1C1C"`, `canvas_bg_mid_grey="#404040"` — `CANVAS_COLORS` in `canvas/toolbar.py:19-23` reads via `THEME`.
- **Status semantic** (resolves `#2e7d32` at `session_panel.py:54`): if "saved OK" semantic → add `status_success="#558B2F"`; if "modified" → reuse existing `accent_edited`. Decide at edit time.

**Verify**: `make all`; `make run` — pixel-identical to baseline.

### Phase 2 — Inline-style migration (Day 1 PM + Day 2 AM)

Three buckets, migrated as coherent batches (no half-state ships):

**A. Move into `modern_dark.qss`** (~25 sites). Selector-stable structural styles — collapsible content frame (`widgets/collapsible.py:133-141`), reset_btn hover (`:94-97`), toolbar container (`canvas/toolbar.py:54-61`), separator lines (`:43-44`), plain label styles in retouch/process/session_panel. Add `objectName` where missing.

**B. Theme f-string templates** (~25 sites). Per-instance dynamic styles → centralize in a new tiny helper `negpy/desktop/view/styles/templates.py` (functions: `slider_label_qss(color, edited)`, `hue_handle_qss(color)`, `swatch_qss(hex)`). Captures: `CompactSlider._update_edited_state` (sliders.py:268-269), `HueSlider._apply_hue` (sliders.py:336-337), `modified_dot` (collapsible.py:102), swatch buttons (canvas/toolbar.py:126-138). Replaces `#2a2a2a` hardcoded at sliders.py:315-327 with `THEME.bg_header`-derived value. Collapsible bg literals at `:46-47` swap to `surface_overlay_strong`/`_hover` tokens.

**C. Stay inline, justified** (~11 sites). `setStyleSheet("")` clears, GPU widget dynamic background swap (`canvas/gpu_widget.py:22,98`), `widgets/charts.py` per-channel pen colors (now token-sourced via THEME).

**Verify**: `make run` walk-through every sidebar (expand/collapse, drag every slider, hover toolbar, toggle swatches, open heal). Visual parity required. `git grep -c setStyleSheet` drops from 61 → ~10.

### Phase 3 — Sidebar polish (Day 2 PM + Day 3 AM)

- **Section subheaders** (small all-caps QLabel: `font_size_xs`, `text_muted`, `weight_semibold`, `space_xl` top margin). Helper function, not new class.
  - `sidebar/lab.py`: "COLOR" / "DETAIL" / "EFFECTS" — 3 subheaders for the 4 paired-slider rows.
  - `sidebar/toning.py`: "TONERS" / "SPLIT TONE" / "PAPER". Drop manual `paper_label` styling at `:45-46`.
  - `sidebar/process.py:51-91`: "AUTO" above Normalize, "BATCH" above Batch Analysis row, "ROLL" above roll combo. Drop fixed 35px height on `analyze_roll_btn`/`use_roll_avg_btn` — match default ~28px.
- **Label casing parity** (`sidebar/toning.py:28,30` etc): `S-Hue/S-Str/H-Hue/H-Str` → `Shadow Hue / Shadow Strength / Highlight Hue / Highlight Strength`.
- **Retouch row fix** (`sidebar/retouch.py:47-61`): replace right-aligned floating label with two rows — top: `HEALS · N` left-aligned subheader-style; bottom: `[Undo Last] [Clear All]` equal-width buttons.
- **Trim redundant modified indicators** (`widgets/collapsible.py:160-169`). 4 signals (dot + count-in-title + 1px border + reset button) → 2 (count-in-title + reset button). Drop `modified_dot` and `_modified_border`.
- **Double-click reset discoverability** (`widgets/sliders.py`): append " (double-click to reset)" suffix to the slider label tooltip in `CompactSlider.__init__`. Tooltip text only — no behaviour change.
- **Combo placeholders** (`sidebar/geometry.py`, `sidebar/icc.py`): match Process Roll's `setPlaceholderText(...)` pattern.
- **Retouch dust_size precision** (`sidebar/retouch.py:22`): leave `int()` cast at `:71`, add a `# TODO` comment. No widget variant.

**Verify**: `make run`. Click through every sidebar. Confirm subheaders render, casing consistent, modified state shows only count + reset button, double-click reset tooltip visible. Before/after screenshot of lab, toning, process, retouch.

### Phase 4 — Canvas / toolbar polish (Day 3 PM + Day 4 AM)

- **Toolbar responsive overflow** (`canvas/toolbar.py:196-213`): in `resizeEvent`, hide groups in priority order into existing `btn_overflow` menu. Thresholds (from sum of child sizeHints): `<720px` hide swatches+HQ; `<580px` also hide flip+rotate. Keep prev/next/zoom/save/export always visible. No two-row layout.
- **Active-tool indicator** (`main_window.py`): (a) title prefix `[Heal Tool] filename.dng` when `state.active_tool != ToolMode.NONE`; (b) status bar left-side icon+name label when active. Reuse existing QLabel; no new widget.
- **Status bar density** (`main_window.py`): drop UPPERCASE on messages, bump progress bar 1px → 3px, apply `font_size_xs`. Right cluster: zoom% / dimensions / active tool, separated by `space_md`.
- **Metadata panel breathing room** (`canvas/overlays.py:20`): 24px → 28px height; padding via QSS using `space_sm` vertical / `space_md` horizontal. Pixel-readout `rgba(13,13,13,0.88)` literal → `THEME.surface_overlay`.
- **Dirty-state reinforcement**: in addition to existing title `●`, add 1px `accent_primary` underline on active file's roll/strip entry when `state.dirty` (locate file roll widget — likely under `sidebar/session_panel.py` or sibling).
- **Context-menu shortcut sourcing** (`canvas/widget.py:262-264`): replace hardcoded `"Pick WB Shift+W"` strings with `tooltip_with_shortcut(...)` (already used at `sidebar/retouch.py:37`). Single source = `shortcut_registry`.
- **Zoom slider/wheel desync** (`canvas/widget.py:234`): in `ActionToolbar._on_zoom_changed`, ensure `zoom_label` updates after signal restore; route wheel-zoom through `controller.zoom_changed.emit` rather than direct slider mutation.
- **Hardcoded `font-size: 11px`** (`canvas/toolbar.py:110` and similar) → `font_size_xs` (most caught by Phase 2 buckets A/B).

**Verify**: `make run` at widths 1920/1280/1024/800 — toolbar collapses cleanly. Wheel-zoom + slider stay synced. Activate heal tool → title prefix + status indicator. Right-click canvas → menu strings reflect a temporarily edited shortcut.

## Cuts (intentional)

- **No slider behaviour changes** (per user). No value tooltip on drag, no drag-to-scrub spinbox. `CompactSlider`/`HueSlider` interaction model untouched. Phase 3's double-click reset hint is tooltip-text-only — no behaviour change.
- No light mode (out of scope by user choice).
- No two-row toolbar (overflow menu pattern is cheaper).
- No widget restructure (`CompactSlider`/`HueSlider`/`CollapsibleSection` internals stay; only additive changes — modified-indicator trim is property-level, not interaction-level).
- No icon/typography rework beyond `font_size_xs`/`font_size_lg` additions and Title Case label fixes.
- No `WorkspaceConfig` / controller / pipeline changes — view layer only.
- No retouch dust_size typed-int slider variant.
- No keyboard shortcut overlay redesign.
- No new histogram/curves UI — channel tokens added but only swap existing colors.
- No SectionLabel class — small helper function only.

## Critical files

- `negpy/desktop/view/styles/theme.py` — token extensions (Phase 1)
- `negpy/desktop/view/styles/modern_dark.qss` — Bucket A migrations (Phase 2)
- `negpy/desktop/view/styles/templates.py` — NEW, Bucket B helpers (Phase 2)
- `negpy/desktop/view/widgets/collapsible.py` — modified-state trim, overlay tokens (Phase 2, 3)
- `negpy/desktop/view/widgets/sliders.py` — Bucket B template adoption, double-click reset tooltip-text only (Phase 2, 3)
- `negpy/desktop/view/widgets/charts.py` — channel color tokens (Phase 1, 2)
- `negpy/desktop/view/widgets/toast.py` — `font_size_lg` (Phase 1)
- `negpy/desktop/view/sidebar/lab.py`, `toning.py`, `process.py`, `retouch.py`, `geometry.py`, `icc.py`, `session_panel.py` — Phase 3
- `negpy/desktop/view/canvas/toolbar.py` — overflow, swatch tokens, font-size tokens (Phase 1, 4)
- `negpy/desktop/view/canvas/widget.py` — context menu shortcut sourcing, zoom sync (Phase 4)
- `negpy/desktop/view/canvas/overlays.py` — metadata panel padding/height, surface_overlay token (Phase 4)
- `negpy/desktop/view/main_window.py` — title tool prefix, status bar density (Phase 4)

## Reused existing utilities

- `tooltip_with_shortcut(...)` (used at `sidebar/retouch.py:37`) — reused in `canvas/widget.py` context menu (Phase 4)
- `THEME` import (`styles/theme.py`) — single import target; all new tokens flow through it
- Existing `btn_overflow` QMenu (`canvas/toolbar.py`) — reused as overflow target for responsive toolbar (Phase 4)
- `CollapsibleSection` reset button (`widgets/collapsible.py:87-99`) — kept as the modified-state affordance after trimming dot+border

## End-to-end verification

After Phase 4 complete:
1. `make all` — lint + types + tests clean.
2. `make run` — load a real `.dng`. Adjust exposure (every CMY slider, density, grade, toe, shoulder). Toggle heal tool, perform 3 heals, undo, clear. Switch process mode. Save preset. Export. Watch for: visual parity in unchanged areas, new subheaders/casing, toolbar overflow at 1024px window, title prefix during heal mode, modified count + reset button (no dot/border), context menu shortcut strings live-tracked from registry. Slider drag/scrub/double-click behaviour identical to baseline.
3. `git grep -c setStyleSheet negpy/desktop/view` < 15.
4. Before/after screenshot pair: full window, sidebar close-up, toolbar at 1920px and 800px widths.
