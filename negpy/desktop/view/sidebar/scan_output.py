"""Where both scanners write: one folder, one roll, shared by the Film Scanner and Camera Scanning."""

import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QVBoxLayout, QWidget

from negpy.desktop.view.styles.templates import field_row, icon_button, labeled_toggle, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.services.assets.rolls import roll_folder_name

SETTINGS_KEY = "scan_output_settings"

_AS_ROLL_TIP = (
    "Make the folder the frames go to a roll in the Library and open it, so Half Frame, roll "
    "defaults and Roll Analysis apply to the frames as they are scanned."
)
_FOLDER_ROLL_TIP = "Scan straight into the output folder, which is the roll and gives it its name. Off scans into a Roll subfolder."


@dataclass(frozen=True)
class ScanOutputSettings:
    output_folder: str = ""
    roll_name: str = "Roll001"
    scan_as_roll: bool = True
    roll_is_folder: bool = True


def load_scan_output_settings(repo) -> ScanOutputSettings:
    """The saved output, or on first run the folder and roll each scanner kept before they shared one."""
    data = repo.get_global_setting(SETTINGS_KEY, default=None)
    if isinstance(data, dict):
        known = {f.name for f in fields(ScanOutputSettings)}
        return ScanOutputSettings(**{k: v for k, v in data.items() if k in known})
    camera = repo.get_global_setting("scanlight_settings", default={}) or {}
    scanner = repo.get_global_setting("scanner_settings", default={}) or {}
    seeded = ScanOutputSettings(
        output_folder=str(camera.get("output_folder") or scanner.get("output_folder") or ""),
        roll_name=str(camera.get("roll_name") or "Roll001"),
    )
    # Saved at once: the scanners' next save drops the keys this was read from.
    repo.save_global_setting(SETTINGS_KEY, asdict(seeded))
    return seeded


class ScanOutputPanel(QWidget):
    """Folder, roll and the two roll options; both scan panels read their target from here."""

    changed = pyqtSignal()

    def __init__(self, repo) -> None:
        super().__init__()
        self._repo = repo
        self._settings = load_scan_output_settings(repo)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_md)

        self.folder_edit = QLineEdit(self._settings.output_folder)
        self.folder_edit.setPlaceholderText("Output folder…")
        self.folder_edit.setToolTip(wrap_tooltip("Where scanned frames are written"))
        self.browse_btn = icon_button("fa5s.folder-open", "Choose the output folder")
        layout.addLayout(field_row("Folder", self.folder_edit, self.browse_btn))

        self.as_roll_btn = labeled_toggle("fa5s.film", " Scan as Roll", self._settings.scan_as_roll, _AS_ROLL_TIP)
        self.folder_roll_btn = labeled_toggle("fa5s.folder", " Folder as Roll", self._settings.roll_is_folder, _FOLDER_ROLL_TIP)
        toggles = QHBoxLayout()
        toggles.addWidget(self.as_roll_btn, 1)
        toggles.addWidget(self.folder_roll_btn, 1)
        layout.addLayout(toggles)

        self.roll_edit = QLineEdit(self._settings.roll_name)
        self.roll_edit.setToolTip(wrap_tooltip("Roll subfolder of the output folder; the name also starts each camera frame's file name"))
        layout.addLayout(field_row("Roll", self.roll_edit))
        self._sync_roll_edit()

        self.browse_btn.clicked.connect(self.browse)
        self.folder_edit.editingFinished.connect(self._save)
        self.roll_edit.editingFinished.connect(self._save)
        self.as_roll_btn.toggled.connect(self._save)
        self.folder_roll_btn.toggled.connect(self._on_folder_roll_toggled)

    def folder(self) -> str:
        return self.folder_edit.text().strip()

    def as_roll(self) -> bool:
        return self.as_roll_btn.isChecked()

    def folder_is_roll(self) -> bool:
        return self.folder_roll_btn.isChecked()

    def roll_name(self) -> Optional[str]:
        """The roll the frames belong to: the output folder's own name, or the Roll field.
        None when the Roll field holds no safe folder name."""
        if self.folder_is_roll():
            return roll_folder_name(os.path.basename(os.path.normpath(self.folder()))) or "Roll001"
        return roll_folder_name(self.roll_edit.text())

    def target_folder(self) -> Optional[str]:
        """The folder the next frames are written to, or None for an unsafe Roll name."""
        if self.folder_is_roll():
            return self.folder()
        roll = self.roll_name()
        return os.path.join(self.folder(), roll) if roll is not None else None

    def set_roll_text(self, text: str) -> None:
        if self.roll_edit.text() != text:
            self.roll_edit.setText(text)
            self._save()

    def browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.folder())
        if folder:
            self.folder_edit.setText(folder)
            self._save()

    def _on_folder_roll_toggled(self) -> None:
        self._sync_roll_edit()
        self._save()

    def _sync_roll_edit(self) -> None:
        self.roll_edit.setEnabled(not self.folder_is_roll())

    def _save(self) -> None:
        updated = replace(
            self._settings,
            output_folder=self.folder(),
            roll_name=self.roll_edit.text().strip() or "Roll001",
            scan_as_roll=self.as_roll(),
            roll_is_folder=self.folder_is_roll(),
        )
        if updated != self._settings:
            self._settings = updated
            self._repo.save_global_setting(SETTINGS_KEY, asdict(updated))
        self.changed.emit()
