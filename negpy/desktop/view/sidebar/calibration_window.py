"""Dedicated pop-up for creating a film-stock preset by ETTR calibration.

Opened by the "+" next to the preset dropdown (independent of the scan cockpit, so
you can calibrate the very first preset). The operator names the stock, clicks the
clear film base (crosshair), and presses Calibrate; on success the panel saves the
preset and closes this window automatically.
"""

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from negpy.desktop.view.sidebar.live_view_window import SettingStepper
from negpy.desktop.view.sidebar.roi_image import RoiImageLabel
from negpy.desktop.view.styles.theme import THEME


class CalibrationWindow(QDialog):
    """Live-view + crosshair + name, to calibrate a new film-stock preset."""

    calibrateRequested = pyqtSignal(str)  # preset name
    closed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New preset — calibrate on the film base")
        self.setModal(False)
        self.resize(820, 680)
        layout = QVBoxLayout(self)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Film stock"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Portra 400")
        name_row.addWidget(self.name_edit, 1)
        self.calibrate_btn = QPushButton(qta.icon("fa5s.crosshairs", color=THEME.text_primary), " Calibrate & Save")
        self.calibrate_btn.setToolTip("Meter the clicked film base and save the result as this preset")
        name_row.addWidget(self.calibrate_btn)
        layout.addLayout(name_row)

        self.image = RoiImageLabel()  # roi_mode=True → a click drops the small base-sampling patch
        self.image.setCursor(Qt.CursorShape.CrossCursor)  # crosshair cursor to place it precisely
        layout.addWidget(self.image, 1)

        # ── ISO + aperture (populated from the stream's settings JSON, like the live view) ──
        # The calibration meters the base at THESE settings and ties the preset to them, so set the
        # ones you'll scan with. The shutter isn't here: the calibration solves it.
        settings_row = QHBoxLayout()
        self.iso_stepper = SettingStepper()
        self.aperture_stepper = SettingStepper()
        for tag_text, stepper, tip in (
            ("ISO", self.iso_stepper, "ISO — use what you will scan with"),
            ("Aperture", self.aperture_stepper, "Aperture (needs an electronically controlled lens)"),
        ):
            tag = QLabel(tag_text)
            tag.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            tag.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
            stepper.setToolTip(tip)
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(tag)
            col.addWidget(stepper)
            settings_row.addLayout(col, 1)
        layout.addLayout(settings_row)

        self.consistency_hint = QLabel(
            "Set the ISO and aperture you'll scan with. Changing either afterwards throws off every scan made with this preset."
        )
        self.consistency_hint.setWordWrap(True)
        self.consistency_hint.setStyleSheet(f"color: #C8922E; font-size: {THEME.font_size_small}px;")
        layout.addWidget(self.consistency_hint)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("Click the clear film base (crosshair), name the stock, then Calibrate & Save.")
        self.status.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.calibrate_btn.clicked.connect(self._emit_calibrate)

    def _emit_calibrate(self) -> None:
        self.calibrateRequested.emit(self.name_edit.text().strip())

    def set_inputs_locked(self, locked: bool) -> None:
        """Freeze the calibration inputs while a run is in progress: the film-stock name, the base
        ROI (clicking the image must not move the patch being metered), and the ISO/aperture the
        base is metered at. Re-enabled at any terminal outcome so a failed run can be retried."""
        self.name_edit.setEnabled(not locked)
        self.iso_stepper.setEnabled(not locked)
        self.aperture_stepper.setEnabled(not locked)
        self.image.set_roi_locked(locked)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def set_progress(self, frac: float) -> None:
        self.progress.setVisible(True)
        self.progress.setValue(int(frac * 100))

    def start(self, default_name: str = "") -> None:
        """Reset and show the window for a fresh calibration."""
        self.set_inputs_locked(False)  # a fresh window always starts editable
        self.name_edit.setText(default_name)
        self.image.clear_roi()
        # Blank the previous session's frame before showing, so reopening goes straight to black +
        # the buffering spinner instead of briefly flashing the stale image.
        self.image.clear_frame()
        self.image.set_loading(True)
        self.progress.setVisible(False)
        self.set_status("Click the clear film base (crosshair), name the stock, then Calibrate & Save.")
        self.show()
        self.raise_()

    def closeEvent(self, ev) -> None:
        self.closed.emit()
        super().closeEvent(ev)
