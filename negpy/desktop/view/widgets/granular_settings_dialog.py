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
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection


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
        show_bounds: bool = False,
        sel_count: int = 0,
        roll_count: int = 0,
        ask_name: bool = False,
        exclude_sections: frozenset[str] = frozenset(),
        show_current: bool = False,
        show_apply_mode: bool = False,
    ):
        super().__init__(parent)
        self._checks: list[tuple[QCheckBox, SettingRow, bool, QWidget]] = []
        self._sections: list[tuple[QWidget, int]] = []
        self._bounds_luma: QCheckBox | None = None
        self._bounds_colour: QCheckBox | None = None
        self._name_edit: QLineEdit | None = None
        if show_current:
            self._scope = "current"
        else:
            self._scope = "selection" if sel_count > 0 else "roll"

        if ask_name:
            self.setWindowTitle("Save Preset")
        else:
            self.setWindowTitle("Paste Settings" if not show_scope else "Apply Settings")
        self.setStyleSheet(f"QDialog {{ background: {THEME.bg_dark}; }}")
        self.resize(420, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        header = QLabel(f'From "{source_name}"' if source_name else "Nothing to apply")
        header.setStyleSheet(f"color: {THEME.text_primary}; font-weight: bold;")
        root.addWidget(header)

        if ask_name:
            self._name_edit = QLineEdit()
            self._name_edit.setPlaceholderText("Preset name")
            self._name_edit.textChanged.connect(self._update_apply_enabled)
            root.addWidget(self._name_edit)
        if show_scope:
            root.addLayout(self._build_scope_row(sel_count, roll_count, show_current))
        if show_apply_mode:
            root.addLayout(self._build_mode_row())
        root.addLayout(self._build_checks_row())
        root.addWidget(self._build_sections(source_cfg, show_bounds, exclude_sections), 1)
        root.addLayout(self._build_footer(ask_name))

        self._apply_visibility()
        self._update_apply_enabled()

    def _build_scope_row(self, sel_count: int, roll_count: int, show_current: bool = False) -> QHBoxLayout:
        row = QHBoxLayout()
        self.scope_group = QButtonGroup(self)
        if show_current:
            self.current_radio = QRadioButton("Current frame")
            self.scope_group.addButton(self.current_radio)
            row.addWidget(self.current_radio)
        self.sel_radio = QRadioButton(f"Selected frames ({sel_count})")
        self.sel_radio.setEnabled(sel_count > 0)
        self.roll_radio = QRadioButton(f"Whole roll ({roll_count})")
        self.roll_radio.setEnabled(roll_count > 0)
        self.scope_group.addButton(self.sel_radio)
        self.scope_group.addButton(self.roll_radio)
        if show_current:
            self.current_radio.setChecked(True)
        else:
            (self.sel_radio if sel_count > 0 else self.roll_radio).setChecked(True)
        row.addWidget(self.sel_radio)
        row.addWidget(self.roll_radio)
        row.addStretch()
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
        row.addWidget(check_all)
        row.addWidget(check_none)
        row.addStretch()
        row.addWidget(self._show_unchanged)
        return row

    def _build_sections(self, source_cfg, show_bounds: bool, exclude_sections: frozenset[str] = frozenset()) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_sm)

        for title, rows in catalog_sections(source_cfg):
            if title in exclude_sections:
                continue
            edited_count = sum(1 for _r, _v, edited in rows if edited)
            section = CollapsibleSection(title, expanded=True)
            section.set_modified(edited_count)
            section.set_content(self._build_rows(rows))
            col.addWidget(section)
            self._sections.append((section, edited_count))

        if show_bounds:
            section = CollapsibleSection("Roll baseline", expanded=True)
            section.set_content(self._build_bounds_rows())
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
            box.setChecked(edited)
            box.stateChanged.connect(self._update_apply_enabled)
            self._checks.append((box, row, edited, line))
            val = QLabel(value)
            val.setStyleSheet(f"color: {THEME.text_muted};")
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
        self._bounds_colour = QCheckBox("Colour balance")
        for box in (self._bounds_luma, self._bounds_colour):
            box.stateChanged.connect(self._update_apply_enabled)
            col.addWidget(box)
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
        return row

    def _all_boxes(self) -> list[QCheckBox]:
        boxes = [box for box, _row, _edited, _line in self._checks]
        boxes += [b for b in (self._bounds_luma, self._bounds_colour) if b is not None]
        return boxes

    def _apply_visibility(self) -> None:
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
        show_all = self._show_unchanged.isChecked()
        for box, _row, edited, _line in self._checks:
            if edited or show_all:
                box.setChecked(checked)
        for box in (self._bounds_luma, self._bounds_colour):
            if box is not None:
                box.setChecked(checked)

    def _update_apply_enabled(self) -> None:
        enabled = any(box.isChecked() for box in self._all_boxes())
        if self._name_edit is not None:
            enabled = enabled and bool(self._name_edit.text().strip())
        self.apply_btn.setEnabled(enabled)

    def _on_apply(self) -> None:
        if hasattr(self, "sel_radio"):
            if getattr(self, "current_radio", None) is not None and self.current_radio.isChecked():
                self._scope = "current"
            else:
                self._scope = "selection" if self.sel_radio.isChecked() else "roll"
        self.accept()

    def selected(self) -> list[SettingRow]:
        return [row for box, row, _edited, _line in self._checks if box.isChecked()]

    def name(self) -> str:
        return self._name_edit.text().strip() if self._name_edit is not None else ""

    def set_name(self, value: str) -> None:
        if self._name_edit is not None:
            self._name_edit.setText(value)

    def bounds_flags(self) -> tuple[bool, bool]:
        return (
            self._bounds_luma is not None and self._bounds_luma.isChecked(),
            self._bounds_colour is not None and self._bounds_colour.isChecked(),
        )

    def scope(self) -> str:
        return self._scope

    def apply_mode(self) -> str:
        if getattr(self, "replace_radio", None) is not None and self.replace_radio.isChecked():
            return "replace"
        return "overlay"


def open_paste_dialog(parent, controller) -> None:
    """Open the granular picker on the clipboard config and apply the chosen
    settings to the active frame. No-op when the clipboard is empty."""
    state = controller.session.state
    if state.clipboard is None or not state.current_file_hash:
        return
    dlg = GranularSettingsDialog(parent, state.clipboard, "clipboard", show_scope=False)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        controller.session.apply_pasted_fields(dlg.selected())
