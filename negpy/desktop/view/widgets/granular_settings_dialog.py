from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.settings_catalog import SettingRow, edited_sections
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection


class GranularSettingsDialog(QDialog):
    """Per-setting picker for paste / apply-to-many. Lists one collapsible section
    per edit area, showing only the settings that differ from default, each with a
    checkbox and its value. Reuses the shortcut-editor's CollapsibleSection look."""

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
    ):
        super().__init__(parent)
        self._checks: list[tuple[QCheckBox, SettingRow]] = []
        self._bounds_luma: QCheckBox | None = None
        self._bounds_colour: QCheckBox | None = None
        self._scope = "selection" if sel_count > 0 else "roll"

        self.setWindowTitle("Paste Settings" if not show_scope else "Apply Settings")
        self.setStyleSheet(f"QDialog {{ background: {THEME.bg_dark}; }}")
        self.resize(420, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        header = QLabel(f'From "{source_name}"' if source_name else "Nothing to apply")
        header.setStyleSheet(f"color: {THEME.text_primary}; font-weight: bold;")
        root.addWidget(header)

        if show_scope:
            root.addLayout(self._build_scope_row(sel_count, roll_count))
        root.addLayout(self._build_checks_row())
        root.addWidget(self._build_sections(source_cfg, show_bounds), 1)
        root.addLayout(self._build_footer())

        self._update_apply_enabled()

    def _build_scope_row(self, sel_count: int, roll_count: int) -> QHBoxLayout:
        row = QHBoxLayout()
        self.scope_group = QButtonGroup(self)
        self.sel_radio = QRadioButton(f"Selected frames ({sel_count})")
        self.sel_radio.setEnabled(sel_count > 0)
        self.roll_radio = QRadioButton(f"Whole roll ({roll_count})")
        self.roll_radio.setEnabled(roll_count > 0)
        self.scope_group.addButton(self.sel_radio)
        self.scope_group.addButton(self.roll_radio)
        (self.sel_radio if sel_count > 0 else self.roll_radio).setChecked(True)
        row.addWidget(self.sel_radio)
        row.addWidget(self.roll_radio)
        row.addStretch()
        return row

    def _build_checks_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        check_all = QPushButton("Check All")
        check_all.clicked.connect(lambda: self._set_all_checked(True))
        check_none = QPushButton("Check None")
        check_none.clicked.connect(lambda: self._set_all_checked(False))
        row.addWidget(check_all)
        row.addWidget(check_none)
        row.addStretch()
        return row

    def _build_sections(self, source_cfg, show_bounds: bool) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_sm)

        for title, rows in edited_sections(source_cfg):
            section = CollapsibleSection(title, expanded=True)
            section.set_modified(len(rows))
            section.set_content(self._build_rows(rows))
            col.addWidget(section)

        if show_bounds:
            section = CollapsibleSection("Roll baseline", expanded=True)
            section.set_content(self._build_bounds_rows())
            col.addWidget(section)

        col.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_rows(self, rows: list[tuple[SettingRow, str]]) -> QWidget:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(0, 1)
        for r, (row, value) in enumerate(rows):
            box = QCheckBox(row.label)
            box.setChecked(True)
            box.stateChanged.connect(self._update_apply_enabled)
            self._checks.append((box, row))
            val = QLabel(value)
            val.setStyleSheet(f"color: {THEME.text_muted};")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(box, r, 0)
            grid.addWidget(val, r, 1)
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

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        row.addWidget(cancel_btn)
        row.addWidget(self.apply_btn)
        return row

    def _all_boxes(self) -> list[QCheckBox]:
        boxes = [box for box, _ in self._checks]
        boxes += [b for b in (self._bounds_luma, self._bounds_colour) if b is not None]
        return boxes

    def _set_all_checked(self, checked: bool) -> None:
        for box in self._all_boxes():
            box.setChecked(checked)

    def _update_apply_enabled(self) -> None:
        self.apply_btn.setEnabled(any(box.isChecked() for box in self._all_boxes()))

    def _on_apply(self) -> None:
        if hasattr(self, "sel_radio"):
            self._scope = "selection" if self.sel_radio.isChecked() else "roll"
        self.accept()

    def selected(self) -> list[SettingRow]:
        return [row for box, row in self._checks if box.isChecked()]

    def bounds_flags(self) -> tuple[bool, bool]:
        return (
            self._bounds_luma is not None and self._bounds_luma.isChecked(),
            self._bounds_colour is not None and self._bounds_colour.isChecked(),
        )

    def scope(self) -> str:
        return self._scope


def open_paste_dialog(parent, controller) -> None:
    """Open the granular picker on the clipboard config and apply the chosen
    settings to the active frame. No-op when the clipboard is empty."""
    state = controller.session.state
    if state.clipboard is None or not state.current_file_hash:
        return
    dlg = GranularSettingsDialog(parent, state.clipboard, "clipboard", show_scope=False)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        controller.session.apply_pasted_fields(dlg.selected())
