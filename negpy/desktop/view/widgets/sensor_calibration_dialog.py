"""Calibrate the sensor-crosstalk matrix from three bare-light exposures.

Red-only / green-only / blue-only captures (no film in the holder, same light
and camera settings as scanning) measure the sensor's response to each band;
the inverted mix is saved as a named sensor profile the Process panel selects.
"""

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from negpy.kernel.system.text import plural
from negpy.desktop.view.styles.templates import hint_label
from negpy.desktop.view.styles.theme import THEME
from negpy.features.process.sensor import build_sensor_matrix, measure_capture
from negpy.infrastructure.loaders.helpers import get_supported_raw_wildcards
from negpy.services.assets.sensor import SensorProfiles

_BANDS = (("R", "Red exposure"), ("G", "Green exposure"), ("B", "Blue exposure"))
_CLIP_LEVEL = 0.98
_CLIP_FRACTION = 0.02


class SensorCalibrationDialog(QDialog):
    """Pick three bare-light exposures, compute the unmix matrix, save a profile."""

    profile_saved = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._paths = {"R": "", "G": "", "B": ""}
        self.setWindowTitle("Calibrate Sensor")
        self.resize(560, 320)
        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        intro = QLabel(
            "Pick three <b>bare-light</b> exposures — red-only, green-only and blue-only, with no film in the "
            "holder and the same light and camera settings you scan with. NegPy measures the sensor's response "
            "to each band and builds the correction. Expose just below clipping."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {THEME.text_secondary};")
        root.addWidget(intro)

        grid = QGridLayout()
        self._path_edits = {}
        for i, (band, label) in enumerate(_BANDS):
            grid.addWidget(QLabel(label), i, 0)
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText("Choose a capture…")
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _c=False, b=band: self._browse(b))
            grid.addWidget(edit, i, 1)
            grid.addWidget(browse, i, 2)
            self._path_edits[band] = edit
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. X-T30 + Scanlight v4")
        self.name_edit.textChanged.connect(self._refresh)
        name_row.addWidget(self.name_edit, 1)
        root.addLayout(name_row)

        self.result_label = hint_label()
        self.result_label.setWordWrap(True)
        self.result_label.setVisible(False)
        root.addWidget(self.result_label)
        root.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        self.compute_btn = QPushButton("Compute and Save")
        self.compute_btn.setDefault(True)
        self.compute_btn.clicked.connect(self._compute_and_save)
        btn_row.addWidget(close)
        btn_row.addWidget(self.compute_btn)
        root.addLayout(btn_row)
        self._refresh()

    def _browse(self, band: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select the {band} bare-light exposure", "", f"Supported Images ({get_supported_raw_wildcards()})"
        )
        if path:
            self._paths[band] = path
            self._path_edits[band].setText(path)
            self._refresh()

    def _refresh(self) -> None:
        name = self.name_edit.text().strip()
        self.compute_btn.setEnabled(all(self._paths.values()) and bool(name) and not SensorProfiles.is_bundled(name))

    def _decode(self, path: str) -> np.ndarray:
        # Sensor-native linear decode (neutral WB), same as the flat-field reference —
        # per-capture exposure scale cancels in build_sensor_matrix's normalization.
        from negpy.services.rendering.preview_manager import PreviewManager

        buf, _, _ = PreviewManager().load_linear_preview(path, use_camera_wb=False, full_resolution=False)
        return buf

    def _compute_and_save(self) -> None:
        name = self.name_edit.text().strip()
        if not name or not all(self._paths.values()):
            return
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            measured = {}
            clipped = []
            for band in ("R", "G", "B"):
                img = self._decode(self._paths[band])
                measured[band] = measure_capture(img)
                if float(np.mean(np.any(np.asarray(img[:, :, :3]) >= _CLIP_LEVEL, axis=2))) > _CLIP_FRACTION:
                    clipped.append(band)
            matrix = build_sensor_matrix(measured["R"], measured["G"], measured["B"])
        except Exception as exc:
            self._show_result(f"Could not build the matrix: {exc}")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        SensorProfiles.save(name, list(matrix))
        self._show_result(self._summary(name, measured, clipped))
        self.profile_saved.emit(name)

    def _summary(self, name: str, measured: dict, clipped: list) -> str:
        s = np.column_stack([measured["R"], measured["G"], measured["B"]])
        s_norm = s / np.diag(s)
        gb, bg = s_norm[2, 1], s_norm[1, 2]  # green->blue / blue->green, the usual dominant leaks
        lines = [
            f"<b>Saved “{name}”</b> — measured green↔blue leakage {gb * 100:.0f}% / {bg * 100:.0f}%.",
        ]
        if clipped:
            lines.append(
                f"⚠ {'/'.join(clipped)} {plural(len(clipped), 'exposure')} {plural(len(clipped), 'looks', 'look')} clipped — reshoot dimmer for an accurate matrix."
            )
        return "<br>".join(lines)

    def _show_result(self, text: str) -> None:
        self.result_label.setText(text)
        self.result_label.setVisible(True)
