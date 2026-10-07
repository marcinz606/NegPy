"""Dedicated pop-up for creating a film-stock preset by ETTR calibration.

Opened by the "+" next to the preset dropdown (independent of the scan cockpit, so
you can calibrate the very first preset). The operator names the stock, clicks the
clear film base (crosshair), and presses Calibrate; on success the panel saves the
preset and closes this window automatically.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QFontMetrics
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
from negpy.desktop.view.styles.templates import hint_label, labeled_action, labeled_toggle
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.choice_button import ChoiceButton
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.desktop.view.widgets.floating_panel import float_over_app

# The index is the preset's `single_capture` flag.
CAPTURE_MODES = (("", "Triplet"), ("", "Single Capture"))
CAPTURE_MODE_TOOLTIP = (
    "Triplet takes one exposure per LED and merges them. Single Capture takes one exposure with red, green and blue lit together."
)


SENSOR_PROFILE_TOOLTIP = (
    "Also save a sensor profile under the preset's name, measured from the same exposures. "
    "A roll scanned with the preset takes the profile. Single Capture only."
)


def _reserve_checked_width(btn: QPushButton) -> None:
    """Size a toggle for its checked label. The sheet draws a checked button semibold, and the
    size hint measures the regular weight."""
    btn.ensurePolished()
    semibold = QFont(btn.font())
    semibold.setWeight(QFont.Weight(THEME.weight_semibold))
    extra = QFontMetrics(semibold).horizontalAdvance(btn.text()) - QFontMetrics(btn.font()).horizontalAdvance(btn.text())
    btn.setMinimumWidth(btn.sizeHint().width() + max(0, extra))


class CalibrationWindow(QDialog):
    """Live-view + crosshair + name, to calibrate a new film-stock preset."""

    calibrateRequested = pyqtSignal(str)  # preset name
    closed = pyqtSignal()

    def __init__(self, parent=None, *, repo=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Preset — Calibrate on the Film Base")
        self.setModal(False)
        float_over_app(self)
        self.resize(820, 680)
        layout = QVBoxLayout(self)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Film stock"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Portra 400")
        name_row.addWidget(self.name_edit, 1)
        self._running = False
        self.capture_btn = ChoiceButton(CAPTURE_MODES, CAPTURE_MODE_TOOLTIP)
        name_row.addWidget(self.capture_btn)
        self._sensor_profile_wanted = True  # the operator's pick, kept while Triplet shows the toggle off
        self.sensor_profile_btn = labeled_toggle("fa5s.vials", " Sensor Profile", True, SENSOR_PROFILE_TOOLTIP)
        _reserve_checked_width(self.sensor_profile_btn)
        self.sensor_profile_btn.clicked.connect(self._on_sensor_profile_clicked)
        self.capture_btn.currentChanged.connect(lambda _i: self._sync_sensor_profile())
        name_row.addWidget(self.sensor_profile_btn)
        self._sync_sensor_profile()
        self.calibrate_btn = labeled_action(
            "fa5s.crosshairs", " Calibrate && Save", "Meter the clicked film base and save the result as this preset"
        )
        name_row.addWidget(self.calibrate_btn)
        layout.addLayout(name_row)

        self.image = RoiImageLabel()  # roi_mode=True → a click drops the small base-sampling patch
        self.image.setCursor(Qt.CursorShape.CrossCursor)  # crosshair cursor to place it precisely
        layout.addWidget(self.image, 1)

        # ISO and aperture, populated from the stream's settings JSON like the live view.
        # The calibration meters the base at THESE settings and ties the preset to them, so set
        # the ones you will scan with. The shutter is not here: the calibration solves it.
        settings_row = QHBoxLayout()
        self.iso_stepper = SettingStepper()
        self.aperture_stepper = SettingStepper()
        for tag_text, stepper, tip in (
            ("ISO", self.iso_stepper, "ISO — use what you will scan with"),
            ("Aperture", self.aperture_stepper, "Aperture (needs an electronically controlled lens)"),
        ):
            tag = hint_label(tag_text)
            tag.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            stepper.setToolTip(tip)
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(tag)
            col.addWidget(stepper)
            settings_row.addLayout(col, 1)
        layout.addLayout(settings_row)

        self.consistency_hint = hint_label(
            "Set the ISO and aperture you'll scan with. Changing either afterwards throws off every scan made with this preset.", "warning"
        )
        layout.addWidget(self.consistency_hint)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = hint_label("Click the clear film base (crosshair), name the stock, then Calibrate & Save.")
        layout.addWidget(self.status)

        self._running = False
        self.calibrate_btn.clicked.connect(self._emit_calibrate)
        remember_dialog_geometry(self, repo, "scanlight_calibration")

    def _emit_calibrate(self) -> None:
        self.calibrateRequested.emit(self.name_edit.text().strip())

    def _on_sensor_profile_clicked(self, checked: bool) -> None:
        self._sensor_profile_wanted = checked

    def _sync_sensor_profile(self) -> None:
        single = self.capture_btn.currentIndex() == 1
        self.sensor_profile_btn.setChecked(single and self._sensor_profile_wanted)
        self.sensor_profile_btn.setEnabled(single and not self._running)

    def wants_sensor_profile(self) -> bool:
        """Whether a finished run saves a sensor profile with the preset."""
        return self.capture_btn.currentIndex() == 1 and self._sensor_profile_wanted

    def set_inputs_locked(self, locked: bool) -> None:
        """Freeze the calibration inputs while a run is in progress: the film-stock name, the capture mode, the Sensor Profile toggle, the base
        ROI (clicking the image must not move the patch being metered), and the ISO/aperture the
        base is metered at. Re-enabled at any terminal outcome so a failed run can be retried."""
        self._running = locked
        self.name_edit.setEnabled(not locked)
        self.capture_btn.setEnabled(not locked)
        self._sync_sensor_profile()
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
        # Blank the previous session's frame before showing, so reopening goes straight to black
        # and the buffering spinner instead of flashing the stale image.
        self.image.clear_frame()
        self.image.set_loading(True)
        self.progress.setVisible(False)
        self.set_status("Click the clear film base (crosshair), name the stock, then Calibrate & Save.")
        self.show()
        self.raise_()

    def keyPressEvent(self, ev) -> None:
        # Esc must not cancel a running calibration; the close button still does.
        if not (self._running and ev.key() == Qt.Key.Key_Escape):
            super().keyPressEvent(ev)

    def done(self, result: int) -> None:
        # Esc reaches here without a closeEvent, so the session cleanup hangs off done().
        self.closed.emit()
        super().done(result)
