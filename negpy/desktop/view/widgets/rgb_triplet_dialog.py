from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from negpy.desktop.view.styles.templates import labeled_action, pin_button_box
from negpy.desktop.view.widgets.file_dialogs import last_open_folder, pick_start_dir
from negpy.infrastructure.loaders.helpers import get_supported_raw_wildcards


class RgbTripletDialog(QDialog):
    """Manually assign the red/green/blue exposure files for one RGB-scan frame."""

    def __init__(self, parent, red: str, green: str, blue: str, align: bool = True, start_dir: str = "") -> None:
        super().__init__(parent)
        self._start_dir = start_dir
        self.setWindowTitle("Edit RGB Triplet")
        layout = QVBoxLayout(self)
        self._edits: dict[str, QLineEdit] = {}
        for label, path in (("Red", red), ("Green", green), ("Blue", blue)):
            row = QHBoxLayout()
            row.addWidget(QLabel(label, minimumWidth=48))
            edit = QLineEdit(path)
            row.addWidget(edit, 1)
            browse = labeled_action("", "Browse…", "Pick the file for this channel")
            browse.clicked.connect(lambda _=False, e=edit: self._browse(e))
            row.addWidget(browse)
            layout.addLayout(row)
            self._edits[label] = edit

        self._align = QCheckBox("Align channels (sub-pixel)")
        self._align.setChecked(align)
        self._align.setToolTip("Register green/blue to the red exposure to remove fringing from capture drift.")
        layout.addWidget(self._align)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        pin_button_box(buttons)
        layout.addWidget(buttons)

    def _browse(self, edit: QLineEdit) -> None:
        # An empty row starts where its siblings are: the three exposures of a triplet
        # are shot in one go and live together.
        siblings = [self._edits[label].text() for label in ("Red", "Green", "Blue")]
        start = pick_start_dir(edit.text(), *siblings, self._start_dir)
        path, _ = QFileDialog.getOpenFileName(self, "Select exposure", start, f"Supported Images ({get_supported_raw_wildcards()})")
        if path:
            edit.setText(path)

    def paths(self) -> tuple[str, str, str]:
        return (self._edits["Red"].text(), self._edits["Green"].text(), self._edits["Blue"].text())

    def align(self) -> bool:
        return self._align.isChecked()


def open_triplet_dialog(parent, session) -> None:
    """Assign the active frame's three exposures by hand — the Trichrome card and the
    film strip's right-click menu both open this."""
    idx = session.state.selected_file_idx
    files = session.state.uploaded_files
    if not (0 <= idx < len(files)):
        return
    info = files[idx]
    dlg = RgbTripletDialog(
        parent,
        info["path"],
        info.get("green_path", ""),
        info.get("blue_path", ""),
        info.get("align", True),
        start_dir=last_open_folder(session.repo),
    )
    if dlg.exec():
        red, green, blue = dlg.paths()
        if red and green and blue:
            session.set_triplet(idx, red, green, blue, dlg.align())
