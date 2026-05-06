# Interactive Tutorial Overlay

## Context

NegPy has no onboarding. New users land on the empty canvas with a packed sidebar (10 panels, ~80 widgets) and no in-app guidance — they have to find `docs/USER_GUIDE.md` themselves. Worse, several behaviours are non-obvious and unique to NegPy:

- **Bounds analysis** (not hardcoded orange masks) — D-Range Clip, white/black point offsets, analysis buffer
- **Roll average** — batch normalization for consistent looks across a roll
- **Sigmoid Toe/Shoulder/Width** — paper-grade analog metaphor, not a generic tone curve
- **Density direction** — lower density = brighter (analog darkroom)
- **Linear RAW vs camera WB** — affects starting point of every edit

Goal: a step-by-step in-app tour that walks new users through the pipeline (Process → Exposure → Lab → Retouch → Export) with a static highlight + popup per panel, surfacing those NegPy-specific twists.

## Approach

Build a single `TutorialOverlay` QWidget that parents to `MainWindow`, fills the window, and paints a dark scrim with a cutout around the current target widget. A popup card next to the cutout holds the title, body text, and Prev/Next/Skip buttons. Steps are declared as plain data (one per panel) in a separate module. Auto-expand collapsed `CollapsibleSection`s on entry; restore on exit. Persist `"tutorial_seen"` in `global_settings` (existing `repo.save_global_setting` API).

Keep it surgical: no refactors to controls, sidebars, or controller. The overlay is purely additive.

## File changes

### NEW: `negpy/desktop/view/widgets/tutorial_overlay.py`

`TutorialOverlay(QWidget)` — child of `MainWindow`, geometry tracks parent.

- `paintEvent`: fill window with `QColor(0, 0, 0, 160)` scrim. Compute target widget global rect → map to overlay coords → punch a rounded-rect hole (use `QPainterPath.subtracted`). Draw a 2px accent border around the hole. Pattern matches `CanvasOverlay._draw_ui` at `negpy/desktop/view/canvas/overlay.py:138`.
- Popup is a child `QFrame` with title `QLabel`, body `QLabel` (rich text, wraps), and a row of `QPushButton`s (Prev / Skip / Next). Position the popup adjacent to the target rect, flipping side if it would clip the window edge. Style via existing app stylesheet (Fusion + project QSS).
- Intercepts mouse/keyboard via `setAttribute(Qt.WA_TransparentForMouseEvents, False)` + `setFocusPolicy(StrongFocus)`. Eats clicks outside the popup. Handles `Escape` → dismiss. Arrow keys / Enter → Next, Backspace → Prev.
- Public API: `start(steps: list[TutorialStep])`, `dismiss()`, signal `finished(completed: bool)`.
- On step entry: if target widget is inside a `CollapsibleSection` that is collapsed, expand it (record original state for restore). Call `QScrollArea.ensureWidgetVisible(target)` so the right dock auto-scrolls. On step exit / overlay close: restore section states.
- On parent `resizeEvent`: re-layout overlay geometry and recompute target rect (connect via event filter on parent).

### NEW: `negpy/desktop/view/widgets/tutorial_steps.py`

Pure data module returning `list[TutorialStep]` given a `MainWindow`. Each step:

```python
@dataclass(frozen=True)
class TutorialStep:
    title: str
    body: str           # rich text; supports <b>, <br>, lists
    target: Callable[[MainWindow], QWidget | None]  # lazy lookup; None = centered card
```

Step list (~10 steps, per-panel high-level):

1. **Welcome** — no target. Centered card. Pipeline overview (Import → Process → Exposure → Lab → Export). Mention non-destructive, GPU-accelerated.
2. **Session panel** (left dock) — target `session_panel`. Loading files, multi-select, batch.
3. **Process panel** — target `process_sidebar`. **Bounds analysis vs hardcoded masks** is the headline. Mention D-Range Clip, white/black point offsets, Batch Analysis → Roll Average for consistent looks.
4. **Exposure: Density & Grade** — target `exposure_sidebar.density_slider`. Analog metaphor (lower density = brighter, grade = paper contrast).
5. **Exposure: Sigmoid (Toe/Shoulder)** — target `exposure_sidebar.toe_slider`. Highlight as unique — not a generic curve, models analog roll-off.
6. **Regional CMY + Pick WB** — target `exposure_sidebar.region_global_btn`. Three regions × CMY, plus eyedropper WB. Mention Linear RAW toggle.
7. **Lab panel** — target `lab_sidebar`. CLAHE, sharpening, Glow/Halation as film-emulation specifics.
8. **Retouch panel** — target `retouch_sidebar`. Auto Dust threshold + manual Heal tool.
9. **Export panel** — target `export_sidebar`. Formats, color spaces, batch export, original-resolution export through full GPU pipeline.
10. **Done** — no target. Centered card. Pointer to `docs/USER_GUIDE.md`, mention `override.toml` for backend troubleshooting, link Keyboard Shortcuts overlay.

### MODIFY: `negpy/desktop/view/main_window.py`

- Instantiate `self.tutorial_overlay = TutorialOverlay(self)` in `__init__`, hidden by default.
- Add `def show_tutorial(self) -> None`: build steps via `tutorial_steps.build(self)`, call `tutorial_overlay.start(steps)`. Connect `finished` signal → `repo.save_global_setting("tutorial_seen", True)` regardless of completed/skipped (don't nag again).
- After `MainWindow.__init__` completes, check `repo.get_global_setting("tutorial_seen", False)`; if False, `QTimer.singleShot(500, self.show_tutorial)` so first paint settles before overlay appears.
- Pass `repo` reference into MainWindow (already available via controller).

### MODIFY: `negpy/desktop/view/main_window.py` (`_EmptyStateOverlay`, line 51)

Add a "Take the tour" `QPushButton` below the existing label. Wire to `main_window.show_tutorial`. Always visible — harmless re-entry.

### MODIFY: `negpy/desktop/view/canvas/toolbar.py` (overflow menu, line 140-182)

Add `"Take the tour"` action above the existing `"Keyboard Shortcuts"` entry. Wire to `main_window.show_tutorial`.

### Critical files referenced (no changes needed)

- `negpy/desktop/view/sidebar/controls_panel.py:31` — sidebar attribute access pattern (`controls_panel.<name>_sidebar`, `controls_panel.<name>_section` for CollapsibleSection)
- `negpy/desktop/view/sidebar/collapsible_section.py` — `set_expanded(bool)` method to auto-expand
- `negpy/desktop/view/widgets/shortcuts_overlay.py:7` — reference for overlay styling/structure
- `negpy/desktop/view/canvas/overlay.py:138` — QPainter pattern for semi-transparent draw
- `negpy/infrastructure/storage/repository.py:176-189` — `save_global_setting`/`get_global_setting`

## Out of scope

- Animated control demos (deferred; static highlight per user choice)
- Per-control deep dives (per-panel granularity chosen)
- Localization
- Replaying a single step / "Why am I seeing this?" hint chips on individual controls

## Verification

1. **First-run**: clear `tutorial_seen` from settings DB (`sqlite3 ~/Documents/NegPy/settings.db "DELETE FROM global_settings WHERE key='tutorial_seen'"`), `make run`. Overlay appears ~500ms after window paints. Walk through all 10 steps; verify each highlight lands on the correct widget, popup readable, Prev/Next/Skip work, ESC dismisses.
2. **Persistence**: relaunch — overlay does NOT auto-show. Verify `tutorial_seen=true` in DB.
3. **Re-entry**: open overflow menu (⋯) → "Take the tour" — overlay reappears. Same from empty-state CTA when no image loaded.
4. **Collapsed sections**: collapse Process and Lab sections, run tour. Tutorial expands them on entry; verify they stay expanded if user had them open before, or restore to collapsed if they were closed before tutorial started.
5. **Resize**: during a step, resize the main window. Highlight cutout and popup track the target widget.
6. **Scroll**: collapse Session dock to force scroll; tutorial calls `ensureWidgetVisible` so target becomes visible in the scroll area.
7. **No image loaded**: tour works without any file imported (steps reference panels, not image state).
8. **`make all`**: lint + type + tests pass.
