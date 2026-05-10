# Film strip filter

## Context

User wants a search box at the top of the film strip (`negpy/desktop/view/sidebar/files.py`) to filter the visible file list by filename. Match is plain case-insensitive substring by default, with an optional toggle to switch into regex mode. Filter is purely UI-side — files stay in `state.uploaded_files`, but hidden files are dropped from `selected_indices` so sync edits never touch invisible files.

Decisions (from user):
- Regex via `.*` toggle button next to the input. Plain substring otherwise.
- Match against filename only (extension is part of the name, so `.cr2` works naturally).
- Hiding a file deselects it. Selection always reflects what's visible.

## Files to modify

- `negpy/desktop/session.py` — extend `AssetListModel` with filter state and gating logic.
- `negpy/desktop/view/sidebar/files.py` — add search QLineEdit + regex toggle, wire up to model and selection.
- `negpy/desktop/controller.py` — `request_batch_export` iterates over visible files only.

`DesktopSessionManager.update_selection`, `select_file`, and `state_changed` already handle the selection-pruning side cleanly.

## Design

### `AssetListModel` changes (`session.py:69`)

Add three fields in `__init__`:

```
self._filter_text: str = ""
self._filter_regex: bool = False
self._filter_pattern: re.Pattern | None = None  # cached compiled regex
```

New method:

```
def set_filter(self, text: str, regex: bool) -> bool:
    """Returns True if pattern compiled successfully (or filter is plain/empty)."""
```

- Plain mode: store `text.lower()`, set `_filter_pattern = None`, return `True`.
- Regex mode: try `re.compile(text, re.IGNORECASE)`; on `re.error`, leave previous filter unchanged and return `False`.
- On success, call `_rebuild_indices()` and emit `layoutChanged`.

Extend `_rebuild_indices` (currently `session.py:82-96`):

After sorting, apply filter:

```
if self._filter_text:
    if self._filter_pattern:
        indices = [i for i in indices if self._filter_pattern.search(files[i]["name"])]
    else:
        needle = self._filter_text  # already lowercased
        indices = [i for i in indices if needle in files[i]["name"].lower()]
self._sorted_indices = indices
```

Filter applied after sort so visible order is preserved. `display_to_actual` / `actual_to_display` work unchanged because they operate on `_sorted_indices`.

Also expose `visible_actual_indices() -> set[int]` for the view to compute selection intersection.

### `FileBrowser` changes (`files.py:44`)

UI additions in `_init_ui`, placed between the sort row and `self.list_view` (so it sits directly above the strip):

```
search_row = QHBoxLayout()
self.search_input = QLineEdit()
self.search_input.setPlaceholderText("Filter by filename...")
self.search_input.setClearButtonEnabled(True)
self.regex_btn = QPushButton(".*")
self.regex_btn.setCheckable(True)
self.regex_btn.setFixedWidth(32)
self.regex_btn.setToolTip("Regex mode")
search_row.addWidget(self.search_input)
search_row.addWidget(self.regex_btn)
action_layout.addLayout(search_row)
```

- Use `qta.icon("fa5s.search", color=THEME.text_secondary)` as a leading action via `QLineEdit.addAction(..., QLineEdit.ActionPosition.LeadingPosition)`.
- Style only via existing `THEME` colors when needed (border/bg already inherited from sidebar QSS).

Debounce: reuse the established `QTimer(singleShot=True, interval=200)` pattern from `selection_timer` (`files.py:60-63`). On `textChanged` or regex toggle, restart the timer; the timer slot runs `_apply_filter`.

```
def _apply_filter(self) -> None:
    text = self.search_input.text().strip()
    regex = self.regex_btn.isChecked()
    ok = self.session.asset_model.set_filter(text, regex)
    self._set_search_error(not ok)
    self._prune_selection_to_visible()
    self.sync_ui()  # refresh selection model + viewport
```

`_set_search_error(bad)`: when `True`, set `self.search_input.setStyleSheet(f"border: 1px solid {THEME.accent_primary};")`; on `False`, clear the stylesheet. Cheap visual cue for invalid regex.

`_prune_selection_to_visible()`:

```
visible = self.session.asset_model.visible_actual_indices()
state = self.session.state
new_selection = [i for i in state.selected_indices if i in visible]
new_active = state.selected_file_idx if state.selected_file_idx in visible else (new_selection[0] if new_selection else -1)
if new_selection != state.selected_indices:
    self.session.update_selection(new_selection)
if new_active != state.selected_file_idx and new_active >= 0:
    self.session.select_file(new_active, selection_override=new_selection)
elif new_active == -1 and state.selected_file_idx != -1:
    state.selected_file_idx = -1
    self.session.state_changed.emit()
```

Keeps the active file when it's still visible; otherwise picks the first visible selected file, else clears active. This matches "drop hidden from selection" without surprising re-selection of unrelated files.

### `request_batch_export` change (`controller.py:726`)

"Export All" must respect the active filter. Replace the iteration source at `controller.py:741`:

```
for f in self.state.uploaded_files:
```

with the model's currently-visible files, in display order:

```
model = self.session.asset_model
visible = [self.state.uploaded_files[i] for i in model.visible_actual_indices_ordered()]
for f in visible:
```

Add `visible_actual_indices_ordered()` on `AssetListModel` returning `list(self._sorted_indices)` — same data the list view shows. Empty filter → all files (current behavior preserved). Non-matching filter → empty `tasks`, `if tasks:` guard at `controller.py:775` cleanly no-ops.

The export sidebar "Export All" button label stays as is; the filter chip in the file panel makes the scope obvious. If the filter is empty the behavior matches today exactly.

### Hot folder + add files interaction

`DesktopSessionManager.add_files` calls `self.asset_model.refresh()` which runs `_rebuild_indices`. Filter is applied there automatically — newly added files that don't match stay hidden until the user clears the filter. No extra wiring needed.

### Persistence

Filter text / regex toggle are session-only by design (no `repo.save_global_setting` calls). Reopening the app starts with an empty filter. If the user later wants persistence, it slots in next to the existing `file_sort_order` saved setting.

## Tests

Three layers, mirroring the existing test layout under `tests/`. No `pytest-qt` — use the session-scoped `qapp` fixture from `tests/conftest.py` and `PyQt6.QtTest.QTest` for interaction (matches `test_slider_widgets.py`, `test_tutorial_overlay.py`).

### 1. Model tests — extend `tests/test_desktop_session.py`

New `TestAssetListModelFilter` class. Build the model directly against an `AppState` populated with synthetic `uploaded_files` dicts (`{"name", "path", "hash"}`). No repo needed.

Cases:
- `set_filter("")` → all files visible, `_sorted_indices` length equals input.
- Plain substring case-insensitive: `set_filter("img", regex=False)` → only files whose `name.lower()` contains `"img"`. Verify with mixed-case fixture (`IMG_01.cr2`, `image.NEF`, `note.txt`).
- Plain extension match: `set_filter(".cr2", regex=False)` → only `.cr2` files.
- Plain mode no-match → empty `_sorted_indices`, `rowCount() == 0`.
- Regex success: `set_filter(r"^IMG_\d{4}", regex=True)` returns `True`; only matching names visible.
- Regex invalid: `set_filter("[", regex=True)` returns `False`; previous filter state preserved (assert `_sorted_indices` unchanged from prior call).
- Filter + sort interaction: set sort to `date` descending, apply filter; assert visible order matches sorted-then-filtered expectation.
- `display_to_actual` / `actual_to_display` correctness with filter active (round-trip on every visible row).
- `visible_actual_indices_ordered()` returns the same list as `_sorted_indices`, in display order.
- `refresh()` after appending a non-matching file to `uploaded_files`: new file stays hidden until filter cleared.

### 2. Widget tests — new `tests/test_file_browser_widget.py`

Build a minimal `DesktopSessionManager` with a `MagicMock(spec=StorageRepository)` (mirror `test_desktop_session.py` setup). Stub `controller` enough for `FileBrowser.__init__` (it reads `controller.session`). Populate `state.uploaded_files` with three dicts.

Cases:
- Search input present: `assert browser.search_input is not None` and placeholder text set.
- Typing into the input updates the model after debounce: use `QTest.keyClicks(browser.search_input, "img")`, then trigger the debounce timer immediately by calling `browser._apply_filter()` directly (avoids real-time waits — same pattern used elsewhere for `selection_timer`).
- Toggling regex button switches mode: programmatically `setChecked(True)`, set text to `^img`, call `_apply_filter()`, assert `asset_model._filter_pattern is not None`.
- Invalid regex sets error stylesheet: type `[`, toggle regex on, call `_apply_filter()`; assert `search_input.styleSheet()` contains `THEME.accent_primary`.
- Selection pruning: pre-set `state.selected_indices = [0, 1, 2]` and `selected_file_idx = 1`; apply filter that hides index `1`; assert `state.selected_indices == [0, 2]` and `selected_file_idx in {0, 2}`.
- Selection cleared when no visible match: filter to empty result; assert `selected_indices == []` and `selected_file_idx == -1`.
- Active file preserved when still visible: selected_file_idx points at a still-visible row; assert it's untouched after filter.

### 3. Controller test — extend `tests/test_controller.py` (or new `tests/test_batch_export.py`)

Reuse the existing controller-mocking pattern (patch `RenderWorker`, `PreviewManager` before `__init__`, tear down threads). Patch `_run_export_tasks` to capture the `tasks` list without spinning up a worker thread.

Cases:
- `request_batch_export` with empty filter: tasks length equals `len(state.uploaded_files)` (regression guard).
- Filter narrows export: set `asset_model._filter_text = "img"` and call `asset_model._rebuild_indices()`; call `request_batch_export()`; assert `tasks` contains only `file_info` entries whose names match the filter.
- Filter to zero matches: assert `_run_export_tasks` is **not** called (the `if tasks:` guard at `controller.py:775` short-circuits).
- Task order matches model display order (sort + filter applied).
- `override_settings=True` still respected with filter active (one task with `params.export.export_path` set from current export config).

### Verification command

`uv run pytest tests/test_desktop_session.py tests/test_file_browser_widget.py tests/test_controller.py -v` then `make all` for the full lint + type + test gate.

## Reused utilities

- `THEME.accent_primary`, `THEME.text_secondary` — existing colors, no new theme attrs.
- `qta.icon("fa5s.search", ...)` — same import already used at `files.py:18`.
- `QTimer` debounce pattern — copied from `selection_timer` (`files.py:60`).
- `session.update_selection`, `session.select_file`, `session.state_changed` — existing controller surface (`session.py:375`, `session.py:335`).
- `AssetListModel._rebuild_indices` / `_sorted_indices` permutation — existing sort infrastructure (`session.py:82`).

## Verification

1. `make lint && make type && make test` — must pass.
2. `make run` and exercise:
   - Load a folder with mixed extensions (e.g., `.cr2`, `.nef`, `.jpg`, `.tif`).
   - Type `IMG` → only matching files visible. Backspace to empty → all return.
   - Type `.cr2` → extension filter works.
   - Toggle `.*`, type `^IMG_\d{4}$` → regex match. Type `[` (invalid) → input border turns red, list unchanged.
   - Select 5 files, type a filter that hides 3 of them → selection shrinks to the 2 visible; sync edits operate on those 2 only.
   - With filter active, hot-folder mode picks up a non-matching new file → it stays hidden until filter cleared.
   - Verify dirty underline still draws on the active file when matched, and the active file remains highlighted.
3. No regression in sort: change sort order/direction with filter active — visible files re-order; non-matching files stay hidden.
4. Export All with filter: load 10 files, filter to 3, click Export All → only those 3 appear in the export folder. Clear filter, Export All → all 10 export. Filter to zero matches → Export All no-ops (no worker dispatched).

## Notes

- User's `feedback_plan_doc_location` memory says save plan/spec docs to repo root with a short uppercase name (e.g. `FILTER.md`). Plan mode required this `~/.claude/plans/` path; once approved, copy to `darkroom-py/FILTER.md`.
