from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QKeySequenceEdit,
)

from negpy.desktop.view.shortcut_registry import REGISTRY, default_bindings
from negpy.desktop.view.styles.theme import THEME


class ShortcutEditorDialog(QDialog):
    def __init__(self, bindings: dict[str, str], parent=None, session=None):
        super().__init__(parent)
        self._initial_bindings = dict(bindings)
        self._session = session
        self._edits: dict[str, QKeySequenceEdit] = {}
        self.setWindowTitle("Customize Shortcuts")
        self.resize(760, 720)
        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        self.setStyleSheet(f"""
            QDialog {{ background-color: {THEME.bg_panel}; }}
            QLabel {{ color: {THEME.text_primary}; font-size: 12px; }}
            QPushButton {{ padding: 6px 14px; }}
        """)

        intro = QLabel("Set a shortcut for any action. Duplicate bindings are rejected. Reset All restores the defaults.")
        intro.setWordWrap(True)
        root.addWidget(intro)

        # Mouse / general viewer preferences, kept up top for easy access.
        self._invert_zoom_chk = QCheckBox("Reverse scroll-to-zoom direction (scroll up zooms out)")
        self._invert_zoom_chk.setToolTip(
            "Flip the mouse-wheel zoom direction on the image viewer: scroll up to zoom out, scroll down to zoom in."
        )
        if self._session is not None:
            self._invert_zoom_chk.setChecked(bool(getattr(self._session.state, "invert_zoom_scroll", False)))
        else:
            self._invert_zoom_chk.setEnabled(False)
        root.addWidget(self._invert_zoom_chk)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {THEME.border_color};")
        root.addWidget(divider)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        row = 0
        last_category = None
        for action_id, entry in REGISTRY.items():
            if entry.category != last_category:
                category = QLabel(entry.category)
                category.setStyleSheet(f"color: {THEME.text_secondary}; font-weight: bold; padding-top: 8px;")
                grid.addWidget(category, row, 0, 1, 3)
                row += 1
                last_category = entry.category

            desc = QLabel(entry.description)
            default_lbl = QLabel(entry.default_key)
            default_lbl.setStyleSheet(f"color: {THEME.text_secondary}; font-family: Consolas, monospace;")
            edit = QKeySequenceEdit(QKeySequence(self._initial_bindings.get(action_id, entry.default_key)))
            edit.setClearButtonEnabled(True)
            self._edits[action_id] = edit

            grid.addWidget(desc, row, 0)
            grid.addWidget(default_lbl, row, 1)
            grid.addWidget(edit, row, 2)
            row += 1

        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

        buttons = QHBoxLayout()
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self._reset_all)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(reset_btn)
        buttons.addStretch()
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)
        root.addLayout(buttons)

    def _reset_all(self) -> None:
        for action_id, key in default_bindings().items():
            self._edits[action_id].setKeySequence(QKeySequence(key))

    def _portable(self, edit: QKeySequenceEdit) -> str:
        return edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText)

    def bindings(self) -> dict[str, str]:
        return {action_id: self._portable(edit) for action_id, edit in self._edits.items()}

    def _save(self) -> None:
        seen: dict[str, str] = {}
        for action_id, edit in self._edits.items():
            key = self._portable(edit)
            if not key:
                continue
            other = seen.get(key)
            if other is not None:
                QMessageBox.warning(
                    self,
                    "Duplicate Shortcut",
                    f'"{key}" is assigned to both "{REGISTRY[other].description}" and "{REGISTRY[action_id].description}".',
                )
                return
            seen[key] = action_id

        # Persist the viewer preferences alongside the shortcut bindings. The state is
        # shared with the canvas, so the change takes effect immediately (no restart).
        if self._session is not None:
            invert = self._invert_zoom_chk.isChecked()
            self._session.state.invert_zoom_scroll = invert
            self._session.repo.save_global_setting("invert_zoom_scroll", invert)

        self.accept()
