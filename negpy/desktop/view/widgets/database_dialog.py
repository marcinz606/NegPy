from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.theme import THEME

# (stat key, display label). Order = display order; a separator sits between the
# per-image group and the reusable-tooling group.
_EDIT_ROWS = (
    ("file_settings", "Saved edits (images)"),
    ("edit_history", "Undo-history steps"),
    ("file_marks", "Keep / reject marks"),
)
_TOOLING_ROWS = (
    ("normalization_rolls", "Normalization rolls"),
    ("flatfield_profiles", "Flat-field profiles"),
    ("export_presets", "Export presets"),
    ("app_preferences", "App preferences"),
)


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class DatabaseDialog(QDialog):
    """View what the app has stored in SQLite and clear it.

    Two clear actions: 'Clear Saved Edits' drops per-image looks/history/marks so a
    reloaded image starts from defaults; 'Reset Everything' wipes both databases
    (also presets, rig profiles, preferences). Both confirm first; the counts and
    sizes refresh in place after a clear.
    """

    def __init__(self, repo, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.setWindowTitle("Manage Database")
        self.setMinimumWidth(420)
        self.setStyleSheet(f"QDialog {{ background: {THEME.bg_dark}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        header = QLabel("Stored data")
        header.setStyleSheet(f"color: {THEME.text_primary}; font-size: {THEME.font_size_header}px; font-weight: {THEME.weight_semibold};")
        root.addWidget(header)

        self._grid = QGridLayout()
        self._grid.setColumnStretch(0, 1)
        self._grid.setHorizontalSpacing(THEME.space_2xl)
        self._grid.setVerticalSpacing(THEME.space_sm)
        self._value_labels: dict[str, QLabel] = {}
        root.addLayout(self._grid)

        self._size_label = QLabel("")
        self._size_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_xs}px;")
        root.addWidget(self._size_label)

        note = QLabel(
            "Clearing only affects this app's database. Source files are never touched. "
            "If you export .negpy sidecars, those still exist next to your images and can "
            "restore an edit when that image is reloaded."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_xs}px;")
        root.addWidget(note)

        root.addLayout(self._build_footer())
        self._populate()

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(THEME.space_lg)

        self.clear_edits_btn = QPushButton("Clear Saved Edits")
        self.clear_edits_btn.setToolTip("Drop saved per-image edits, undo history and keep/reject marks. Keeps presets and rig profiles.")
        self.clear_edits_btn.clicked.connect(self._on_clear_edits)

        self.reset_all_btn = QPushButton("Reset Everything")
        self.reset_all_btn.setToolTip(
            "Wipe the entire database: edits, history, marks, rig profiles, export presets and all app preferences."
        )
        self.reset_all_btn.setStyleSheet(
            f"QPushButton {{ background: {THEME.accent_primary}; color: #FFFFFF; border: none; "
            f"border-radius: {THEME.radius_sm}px; padding: {THEME.space_md}px {THEME.space_xl}px; }}"
            f"QPushButton:hover {{ background: {THEME.accent_secondary}; }}"
        )
        self.reset_all_btn.clicked.connect(self._on_reset_all)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        row.addWidget(self.clear_edits_btn)
        row.addWidget(self.reset_all_btn)
        row.addStretch(1)
        row.addWidget(close_btn)
        return row

    def _add_separator(self, grid_row: int) -> None:
        # A plain QFrame HLine draws from the palette and barely shows on the dark
        # background; a 1px background-filled frame renders reliably.
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {THEME.border_color};")
        self._grid.addWidget(line, grid_row, 0, 1, 2)

    def _stat_row(self, grid_row: int, key: str, label: str) -> None:
        name = QLabel(label)
        name.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_base}px;")
        value = QLabel("0")
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value.setStyleSheet(f"color: {THEME.text_primary}; font-size: {THEME.font_size_base}px; font-weight: {THEME.weight_medium};")
        self._grid.addWidget(name, grid_row, 0)
        self._grid.addWidget(value, grid_row, 1)
        self._value_labels[key] = value

    def _populate(self) -> None:
        # Build the grid once; refreshes only update the value labels.
        if not self._value_labels:
            r = 0
            for key, label in _EDIT_ROWS:
                self._stat_row(r, key, label)
                r += 1
            self._add_separator(r)
            r += 1
            for key, label in _TOOLING_ROWS:
                self._stat_row(r, key, label)
                r += 1
        self._refresh()

    def _refresh(self) -> None:
        try:
            stats = self.repo.database_stats()
        except Exception:
            for lbl in self._value_labels.values():
                lbl.setText("—")
            self._size_label.setText("Could not read the database.")
            return
        for key, lbl in self._value_labels.items():
            lbl.setText(f"{stats.get(key, 0):,}")
        total = stats.get("edits_db_bytes", 0) + stats.get("settings_db_bytes", 0)
        self._size_label.setText(f"On disk: {_human_bytes(total)}")
        self._update_enabled(stats)

    def _update_enabled(self, stats: dict) -> None:
        edits = stats.get("file_settings", 0) + stats.get("edit_history", 0) + stats.get("file_marks", 0)
        total = edits + sum(stats.get(k, 0) for k in ("normalization_rolls", "flatfield_profiles", "export_presets", "app_preferences"))
        self.clear_edits_btn.setEnabled(edits > 0)
        self.reset_all_btn.setEnabled(total > 0)

    def _confirm(self, title: str, text: str, ok_label: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        box.setInformativeText("This cannot be undone.")
        ok = box.addButton(ok_label, QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(box.buttons()[-1])  # default to Cancel
        box.exec()
        return box.clickedButton() is ok

    def _on_clear_edits(self) -> None:
        if not self._confirm(
            "Clear Saved Edits",
            "Delete all saved per-image edits, their undo history, and keep/reject marks?\n\n"
            "Reloading an image will start from defaults. Export presets and rig profiles are kept.",
            "Clear Saved Edits",
        ):
            return
        try:
            self.repo.clear_saved_edits()
        except Exception as exc:
            QMessageBox.critical(self, "Clear failed", f"Could not clear the database:\n{exc}")
        self._refresh()

    def _on_reset_all(self) -> None:
        if not self._confirm(
            "Reset Everything",
            "Wipe the entire database — every saved edit, undo history, keep/reject mark, "
            "normalization roll, flat-field profile, export preset, and all app preferences?\n\n"
            "The app returns to a first-run state.",
            "Reset Everything",
        ):
            return
        try:
            self.repo.reset_everything()
        except Exception as exc:
            QMessageBox.critical(self, "Reset failed", f"Could not reset the database:\n{exc}")
        self._refresh()
