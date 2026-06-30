from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from negpy.desktop.view.styles.theme import THEME

_GROUPS = (
    ("Setup", (("process", "Process"), ("crop", "Crop"), ("rotation", "Rotation"))),
    ("Exposure", (("exposure", "Exposure"),)),
    ("Color", (("color", "Color"),)),
    ("Finish", (("finish", "Finish"),)),
    ("Bounds", (("bounds_luma", "Tonal span"), ("bounds_colour", "Colour balance"))),
)


class SyncSettingsDialog(QDialog):
    """Lightroom-style "Apply Settings" dialog: independent checkboxes + Apply."""

    def __init__(self, parent, source_name: str, sel_count: int, roll_count: int):
        super().__init__(parent)
        self._aspects: frozenset = frozenset()
        self._scope = "selection" if sel_count > 0 else "roll"

        self.setWindowTitle("Apply Settings")
        self.setStyleSheet(f"QDialog {{ background: {THEME.bg_dark}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        header = QLabel(f'From "{source_name}"' if source_name else "No frame loaded")
        header.setStyleSheet(f"color: {THEME.text_primary}; font-weight: bold;")
        root.addWidget(header)

        root.addLayout(self._build_scope_row(sel_count, roll_count))
        root.addLayout(self._build_checks_row())
        root.addLayout(self._build_groups(), 1)
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
        if sel_count > 0:
            self.sel_radio.setChecked(True)
        else:
            self.roll_radio.setChecked(True)

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

    def _build_groups(self) -> QVBoxLayout:
        col = QVBoxLayout()
        self._checkboxes: dict[str, QCheckBox] = {}
        for group_label, items in _GROUPS:
            label = QLabel(group_label.upper())
            label.setStyleSheet(f"color: {THEME.text_muted}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
            col.addWidget(label)
            for key, text in items:
                box = QCheckBox(text)
                box.stateChanged.connect(self._update_apply_enabled)
                self._checkboxes[key] = box
                col.addWidget(box)
        return col

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

    def _set_all_checked(self, checked: bool) -> None:
        for box in self._checkboxes.values():
            box.setChecked(checked)

    def _update_apply_enabled(self) -> None:
        self.apply_btn.setEnabled(any(box.isChecked() for box in self._checkboxes.values()))

    def _on_apply(self) -> None:
        self._aspects = frozenset(key for key, box in self._checkboxes.items() if box.isChecked())
        self._scope = "selection" if self.sel_radio.isChecked() else "roll"
        self.accept()

    def aspects(self) -> frozenset:
        return self._aspects

    def scope(self) -> str:
        return self._scope
