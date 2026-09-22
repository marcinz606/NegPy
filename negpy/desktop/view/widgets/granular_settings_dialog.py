import os
from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.settings_catalog import SettingRow, catalog_sections
from negpy.desktop.view.styles.templates import pin_dialog_default, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection


def _triplet(values) -> str:
    return " / ".join(f"{v:g}" for v in values)


class ScopeRadios:
    """The radios build_scope_row built, and the scope they resolve to."""

    __slots__ = ("current", "sel", "roll")

    def __init__(self, current: QRadioButton | None, sel: QRadioButton, roll: QRadioButton) -> None:
        self.current = current
        self.sel = sel
        self.roll = roll

    def value(self) -> str:
        if self.current is not None and self.current.isChecked():
            return "current"
        return "selection" if self.sel.isChecked() else "roll"


def build_scope_row(parent: QWidget, sel_count: int, roll_count: int, show_current: bool = False) -> tuple[QHBoxLayout, ScopeRadios]:
    """Current/Selected/Whole roll radio row, shared by every dialog that applies settings
    to more than the active frame."""
    row = QHBoxLayout()
    group = QButtonGroup(parent)
    current: QRadioButton | None = None
    if show_current:
        current = QRadioButton("Current frame")
        group.addButton(current)
        row.addWidget(current)
    sel = QRadioButton(f"Selected frames ({sel_count})")
    sel.setEnabled(sel_count > 0)
    roll = QRadioButton(f"Whole roll ({roll_count})")
    roll.setEnabled(roll_count > 0)
    group.addButton(sel)
    group.addButton(roll)
    if current is not None:
        current.setChecked(True)
    else:
        (sel if sel_count > 0 else roll).setChecked(True)
    row.addWidget(sel)
    row.addWidget(roll)
    row.addStretch()
    return row, ScopeRadios(current, sel, roll)


class GranularSettingsDialog(QDialog):
    """Per-setting picker for paste / apply-to-many. Lists one collapsible section
    per edit area, each setting with a checkbox and its value. Settings still at
    their default are built but hidden until "Show unchanged settings" — they must
    stay pickable so a roll can be reset back to a default value (#656). Reuses the
    shortcut-editor's CollapsibleSection look."""

    def __init__(
        self,
        parent,
        source_cfg,
        source_name: str,
        *,
        show_scope: bool = False,
        bounds_mode: str = "",
        sel_count: int = 0,
        roll_count: int = 0,
        ask_name: bool = False,
        exclude_sections: frozenset[str] = frozenset(),
        show_current: bool = False,
        show_apply_mode: bool = False,
        preselect_ids: frozenset[str] | None = None,
    ):
        # ponytail: at nine flags this class is at its ceiling. A tenth means splitting
        # pick mode into its own dialog.
        super().__init__(parent)
        self._checks: list[tuple[QCheckBox, SettingRow, bool, QWidget]] = []
        self._sections: list[tuple[QWidget, int]] = []
        self._section_ids: list[tuple[str, ...]] = []
        self._section_rows: list[tuple[CollapsibleSection, tuple[str, ...]]] = []
        self._scope_radios: ScopeRadios | None = None
        self._preselect_ids = preselect_ids
        self._bounds_luma: QCheckBox | None = None
        self._bounds_color: QCheckBox | None = None
        self._bounds_local: QCheckBox | None = None
        self._name_edit: QLineEdit | None = None
        if show_current:
            self._scope = "current"
        else:
            self._scope = "selection" if sel_count > 0 else "roll"

        if preselect_ids is not None:
            self.setWindowTitle("Persistent Settings")
        elif ask_name:
            self.setWindowTitle("Save Preset")
        else:
            self.setWindowTitle("Paste Settings" if not show_scope else "Apply Settings")
        self.resize(420, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        if preselect_ids is not None:
            header = QLabel("Settings that carry onto the next file you open")
            header.setToolTip("A file you have already edited keeps its own look; only export and metadata settings reach it.")
        else:
            header = QLabel(f'From "{source_name}"' if source_name else "Nothing to apply")
        header.setStyleSheet(f"color: {THEME.text_primary}; font-weight: bold;")
        root.addWidget(header)

        if ask_name:
            self._name_edit = QLineEdit()
            self._name_edit.setPlaceholderText("Preset Name")
            self._name_edit.textChanged.connect(self._update_apply_enabled)
            root.addWidget(self._name_edit)
        if show_scope:
            root.addLayout(self._build_scope_row(sel_count, roll_count, show_current))
        if show_apply_mode:
            root.addLayout(self._build_mode_row())
        root.addLayout(self._build_checks_row())
        root.addWidget(self._build_sections(source_cfg, bounds_mode, exclude_sections), 1)
        root.addLayout(self._build_footer(ask_name))

        self._apply_visibility()
        self._refresh_section_states()
        self._update_apply_enabled()

    def _build_scope_row(self, sel_count: int, roll_count: int, show_current: bool = False) -> QHBoxLayout:
        row, self._scope_radios = build_scope_row(self, sel_count, roll_count, show_current)
        return row

    def _build_mode_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.overlay_radio = QRadioButton("Apply on top")
        self.overlay_radio.setToolTip("Only the ticked settings change; the rest of each frame's edit stays")
        self.replace_radio = QRadioButton("Replace look")
        self.replace_radio.setToolTip("Reset look settings to defaults first — crop, rotation, metadata, export and retouch marks stay")
        self.mode_group.addButton(self.overlay_radio)
        self.mode_group.addButton(self.replace_radio)
        self.overlay_radio.setChecked(True)
        row.addWidget(self.overlay_radio)
        row.addWidget(self.replace_radio)
        row.addStretch()
        return row

    def _build_checks_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        check_all = QPushButton("Check All")
        check_all.clicked.connect(lambda: self._set_all_checked(True))
        check_none = QPushButton("Check None")
        check_none.clicked.connect(lambda: self._set_all_checked(False))
        self._show_unchanged = QCheckBox("Show unchanged settings")
        self._show_unchanged.setToolTip("List settings still at their default, so they can be applied too")
        self._show_unchanged.toggled.connect(self._apply_visibility)
        self._show_unchanged.setVisible(self._preselect_ids is None)
        row.addWidget(check_all)
        row.addWidget(check_none)
        row.addStretch()
        row.addWidget(self._show_unchanged)
        return row

    def _build_sections(self, source_cfg, bounds_mode: str, exclude_sections: frozenset[str] = frozenset()) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_sm)

        pick_mode = self._preselect_ids is not None
        for title, rows in catalog_sections(source_cfg):
            if title in exclude_sections:
                continue
            edited_count = sum(1 for _r, _v, edited in rows if edited)
            section = CollapsibleSection(title, expanded=not pick_mode, select=pick_mode)
            if pick_mode:
                boxes = self._build_rows(rows)
                section.set_content(boxes)
                section.selection_toggled.connect(partial(self._on_section_toggled, section, tuple(r for r, _v, _e in rows)))
                self._section_rows.append((section, tuple(r.id for r, _v, _e in rows)))
            else:
                section.set_modified(edited_count)
                section.set_content(self._build_rows(rows))
            col.addWidget(section)
            self._sections.append((section, edited_count))
            self._section_ids.append(tuple(r.id for r, _v, _e in rows))

        if bounds_mode == "axes":
            section = CollapsibleSection("Roll baseline", expanded=True)
            section.set_content(self._build_bounds_rows())
            col.addWidget(section)
        elif bounds_mode == "local":
            section = CollapsibleSection("Normalization bounds", expanded=True)
            section.set_content(self._build_local_bounds_row(source_cfg.process))
            col.addWidget(section)

        col.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_rows(self, rows: list[tuple[SettingRow, str, bool]]) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        for row, value, edited in rows:
            # One widget per row (not a grid) so an unchanged row hides as a unit.
            line = QWidget()
            line_layout = QHBoxLayout(line)
            line_layout.setContentsMargins(0, 0, 0, 0)
            box = QCheckBox(row.label)
            box.setChecked(row.id in self._preselect_ids if self._preselect_ids is not None else edited)
            box.stateChanged.connect(self._update_apply_enabled)
            box.stateChanged.connect(self._refresh_section_states)
            self._checks.append((box, row, edited, line))
            val = QLabel(value)
            val.setStyleSheet(f"color: {THEME.text_hint};")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            line_layout.addWidget(box)
            line_layout.addStretch()
            line_layout.addWidget(val)
            col.addWidget(line)
        return body

    def _build_bounds_rows(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        self._bounds_luma = QCheckBox("Tonal span")
        self._bounds_color = QCheckBox("Color balance")
        for box in (self._bounds_luma, self._bounds_color):
            box.stateChanged.connect(self._update_apply_enabled)
            col.addWidget(box)
        return body

    def _build_local_bounds_row(self, process) -> QWidget:
        """The source frame's own metered bounds, as one opt-out row: they are not
        catalog fields, so they cannot be listed with the rest."""
        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        label = "Copied frame's bounds" + (" (locked)" if process.lock_bounds else "")
        self._bounds_local = QCheckBox(label)
        self._bounds_local.setChecked(True)
        self._bounds_local.stateChanged.connect(self._update_apply_enabled)
        val = QLabel(f"{_triplet(process.local_floors)} → {_triplet(process.local_ceils)}")
        val.setStyleSheet(f"color: {THEME.text_hint};")
        row.addWidget(self._bounds_local)
        row.addStretch()
        row.addWidget(val)
        return body

    def _build_footer(self, ask_name: bool = False) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self.apply_btn = QPushButton("Save" if ask_name else "Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        row.addWidget(cancel_btn)
        row.addWidget(self.apply_btn)
        pin_dialog_default(self.apply_btn, cancel_btn)
        return row

    def _all_boxes(self) -> list[QCheckBox]:
        boxes = [box for box, _row, _edited, _line in self._checks]
        boxes += [b for b in (self._bounds_luma, self._bounds_color, self._bounds_local) if b is not None]
        return boxes

    def _on_section_toggled(self, _section, rows: tuple[SettingRow, ...], checked: bool) -> None:
        wanted = {r.id for r in rows}
        for box, row, _edited, _line in self._checks:
            if row.id in wanted:
                box.setChecked(checked)

    def _refresh_section_states(self) -> None:
        if self._preselect_ids is None:
            return
        state = {row.id: box.isChecked() for box, row, _edited, _line in self._checks}
        for section, row_ids in self._section_rows:
            section.set_selection_state(sum(1 for i in row_ids if state.get(i)), len(row_ids))

    def _apply_visibility(self) -> None:
        # Pick mode lists everything: a setting still at its default must stay selectable.
        if self._preselect_ids is not None:
            return
        show_all = self._show_unchanged.isChecked()
        for box, _row, edited, line in self._checks:
            if edited:
                continue
            line.setVisible(show_all)
            # A hidden-but-ticked row would keep Apply enabled with nothing on screen.
            if not show_all:
                box.setChecked(False)
        for section, edited_count in self._sections:
            section.setVisible(show_all or edited_count > 0)

    def _set_all_checked(self, checked: bool) -> None:
        show_all = self._show_unchanged.isChecked() or self._preselect_ids is not None
        for box, _row, edited, _line in self._checks:
            if edited or show_all:
                box.setChecked(checked)
        for box in (self._bounds_luma, self._bounds_color, self._bounds_local):
            if box is not None:
                box.setChecked(checked)

    def _update_apply_enabled(self) -> None:
        # Ticking nothing is a valid choice in pick mode: nothing carries over.
        if self._preselect_ids is not None:
            self.apply_btn.setEnabled(True)
            return
        enabled = any(box.isChecked() for box in self._all_boxes())
        if self._name_edit is not None:
            enabled = enabled and bool(self._name_edit.text().strip())
        self.apply_btn.setEnabled(enabled)

    def _on_apply(self) -> None:
        if self._scope_radios is not None:
            self._scope = self._scope_radios.value()
        self.accept()

    def selected(self) -> list[SettingRow]:
        return [row for box, row, _edited, _line in self._checks if box.isChecked()]

    def show_unchanged_settings(self, shown: bool = True) -> None:
        """List the rows still at their default, so they can be ticked without hunting."""
        self._show_unchanged.setChecked(shown)

    def set_checked_rows(self, row_ids) -> None:
        """Tick exactly these rows. A row sitting at its default is hidden until "show
        unchanged", and _apply_visibility unticks hidden rows, so reveal them first."""
        wanted = set(row_ids)
        if any(row.id in wanted and not edited for _box, row, edited, _line in self._checks):
            self._show_unchanged.setChecked(True)
        for box, row, _edited, _line in self._checks:
            box.setChecked(row.id in wanted)

    def limit_to_rows(self, row_ids) -> None:
        """Show only these rows, for a picker opened from one section's own header. The
        rest leave _checks entirely, so Check All and the Apply gate never see them."""
        wanted = set(row_ids)
        kept = []
        for box, row, edited, line in self._checks:
            if row.id in wanted:
                kept.append((box, row, edited, line))
            else:
                box.setChecked(False)
                line.setVisible(False)
                line.setParent(None)
        self._checks = kept
        # A section's own edited count drives whether it shows at all, so recount it over
        # what is left rather than over what it was built with.
        kept_edited = {row.id for _box, row, edited, _line in kept if edited}
        rebuilt = []
        for (section, _old), ids in zip(self._sections, self._section_ids):
            count = sum(1 for i in ids if i in kept_edited)
            section.set_modified(count)
            rebuilt.append((section, count))
        self._sections = rebuilt
        self._apply_visibility()
        self._update_apply_enabled()

    def selected_ids(self) -> list[str]:
        return [row.id for row in self.selected()]

    def name(self) -> str:
        return self._name_edit.text().strip() if self._name_edit is not None else ""

    def set_name(self, value: str) -> None:
        if self._name_edit is not None:
            self._name_edit.setText(value)

    def bounds_flags(self) -> tuple[bool, bool]:
        return (
            self._bounds_luma is not None and self._bounds_luma.isChecked(),
            self._bounds_color is not None and self._bounds_color.isChecked(),
        )

    def paste_bounds(self) -> bool:
        return self._bounds_local is not None and self._bounds_local.isChecked()

    def scope(self) -> str:
        return self._scope

    def apply_mode(self) -> str:
        if getattr(self, "replace_radio", None) is not None and self.replace_radio.isChecked():
            return "replace"
        return "overlay"


class SyncBoundsDialog(QDialog):
    """The active frame's metering bounds and nothing else. The Apply picker reaches the
    same two axes, but only once every edited row has been unticked by hand."""

    def __init__(self, parent, floors, ceils, source_name: str, sel_count: int, roll_count: int):
        super().__init__(parent)
        self.setWindowTitle("Sync Bounds")
        self.apply_btn = QPushButton("Apply")

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        header = QLabel(f'From "{source_name}"' if source_name else "From this frame")
        header.setStyleSheet(f"color: {THEME.text_primary}; font-weight: bold;")
        root.addWidget(header)
        value = QLabel(f"{_triplet(floors)} → {_triplet(ceils)}")
        value.setStyleSheet(f"color: {THEME.text_hint};")
        root.addWidget(value)

        row, self._scope_radios = build_scope_row(self, sel_count, roll_count)
        root.addLayout(row)

        self.luma_box = QCheckBox("Tonal span")
        self.luma_box.setToolTip(wrap_tooltip("Take the black/white-point span from this frame"))
        self.color_box = QCheckBox("Color balance")
        self.color_box.setToolTip(wrap_tooltip("Take the per-channel color balance from this frame"))
        for box in (self.luma_box, self.color_box):
            box.setChecked(True)
            box.stateChanged.connect(self._update_apply_enabled)
            root.addWidget(box)

        footer = QHBoxLayout()
        footer.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self.apply_btn.clicked.connect(self.accept)
        footer.addWidget(cancel_btn)
        footer.addWidget(self.apply_btn)
        pin_dialog_default(self.apply_btn, cancel_btn)
        root.addLayout(footer)

    def _update_apply_enabled(self) -> None:
        self.apply_btn.setEnabled(self.luma_box.isChecked() or self.color_box.isChecked())

    def bounds_flags(self) -> tuple[bool, bool]:
        return self.luma_box.isChecked(), self.color_box.isChecked()

    def scope(self) -> str:
        return self._scope_radios.value()


def _apply_targets(session) -> tuple[int, int, int] | None:
    """(source index, selected targets, roll targets) for a frame-to-frames apply, or
    None when there is nothing to apply to.

    "Whole roll" means the visible (filtered) frames, not every loaded file: a filename
    filter is a non-destructive view, so hidden files are not counted.
    """
    state = session.state
    src = state.selected_file_idx
    if src == -1:
        return None
    visible = session.asset_model.visible_actual_indices()
    sel_targets = len([i for i in set(state.selected_indices) if i != src and i in visible])
    roll_targets = len([i for i in visible if i != src])
    if not sel_targets and not roll_targets:
        session.settings_synced.emit("Only one frame here — nothing to apply to")
        return None
    return src, sel_targets, roll_targets


def _source_name(session, src: int) -> str:
    files = session.state.uploaded_files
    return os.path.basename(files[src]["path"]) if src < len(files) else ""


def open_sync_bounds_dialog(parent, session) -> None:
    """Push the active frame's metering bounds onto other frames, leaving every other
    setting where it is."""
    from negpy.desktop.session import _source_effective_bounds

    bounds = _source_effective_bounds(session.state.config.process)
    if bounds is None:
        session.settings_synced.emit("Render this frame before syncing its bounds")
        return
    targets = _apply_targets(session)
    if targets is None:
        return
    src, sel_targets, roll_targets = targets
    dlg = SyncBoundsDialog(parent, bounds[0], bounds[1], _source_name(session, src), sel_targets, roll_targets)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        session.sync_selected_settings([], dlg.bounds_flags(), dlg.scope())


def open_apply_dialog(parent, session, rows=None, title: str = "") -> tuple[list, str] | None:
    """Apply the active frame's settings to the selection or the whole roll.

    *rows* limits the picker to one section's own settings, for the Roll button on that
    section's header; None lists every section, which is the Film Strip's own button. The
    metered-bounds rows come only with the full list: they belong to Normalization, a
    Roll-tab card with a scope pair of its own.

    Returns what was applied, as (rows, scope), or None if nothing was.
    """
    from negpy.desktop.session import _source_effective_bounds

    targets = _apply_targets(session)
    if targets is None:
        return None
    src, sel_targets, roll_targets = targets

    source_cfg = session.state.config
    source_name = _source_name(session, src)
    bounds_mode = "axes" if rows is None and _source_effective_bounds(source_cfg.process) is not None else ""
    dlg = GranularSettingsDialog(
        parent,
        source_cfg,
        source_name,
        show_scope=True,
        bounds_mode=bounds_mode,
        sel_count=sel_targets,
        roll_count=roll_targets,
    )
    if rows is not None:
        dlg.limit_to_rows([r.id for r in rows])
    if title:
        dlg.setWindowTitle(title)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    applied = dlg.selected()
    session.sync_selected_settings(applied, dlg.bounds_flags(), dlg.scope())
    return applied, dlg.scope()


def open_paste_dialog(parent, controller) -> None:
    """Open the granular picker on the clipboard config and apply the chosen
    settings to the active frame. No-op when the clipboard is empty."""
    state = controller.session.state
    if state.clipboard is None or not state.current_file_hash:
        return
    bounds_mode = "local" if state.clipboard.process.is_local_initialized else ""
    dlg = GranularSettingsDialog(parent, state.clipboard, "clipboard", bounds_mode=bounds_mode)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        controller.session.apply_pasted_fields(dlg.selected(), include_bounds=dlg.paste_bounds())


def open_sticky_dialog(parent, controller) -> None:
    """Pick which settings carry onto a freshly-opened file. Values shown are the last
    saved edit's, so the list reads as what would actually carry."""
    from negpy.domain.models import WorkspaceConfig
    from negpy.desktop.sticky import load_sticky_config, load_sticky_rows, save_sticky_rows

    repo = controller.session.repo
    source = load_sticky_config(repo) or WorkspaceConfig()
    chosen = frozenset(r.id for r in load_sticky_rows(repo))
    dlg = GranularSettingsDialog(parent, source, "", preselect_ids=chosen)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        save_sticky_rows(repo, dlg.selected_ids())
