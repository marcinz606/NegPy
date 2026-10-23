"""Calibrate the sensor-crosstalk matrix from three bare-light exposures.

Red, green and blue captures with no film, at the scanning light and camera settings, measure
each band; the inverted mix is saved as a named sensor crosstalk profile. Files, or a tethered capture.
"""

from negpy.kernel.system.i18n import tr

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from negpy.kernel.system.text import plural
from negpy.desktop.view.confirm import confirm_sensor_capture
from negpy.desktop.view.styles.templates import field_row, hint_label, pin_dialog_default, wrap_tooltip
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.desktop.view.widgets.file_dialogs import pick_start_dir
from negpy.features.process.sensor import build_sensor_matrix, measure_capture
from negpy.infrastructure.loaders.helpers import get_supported_raw_wildcards
from negpy.services.assets.sensor import SensorProfiles

_BANDS = (("R", tr("Red")), ("G", tr("Green")), ("B", tr("Blue")))
_CLIP_LEVEL = 0.98
_CLIP_FRACTION = 0.02
_CAPTURE_LABEL = tr("Capture from Camera…")
_CAPTURE_TOOLTIP = tr(
    "Shoot the red, green and blue exposures with the camera and the Scanlight, then save the profile under the name above."
)
_PRESENCE_POLL_MS = 3000
_LED_SETTLE_S = 0.05  # the Scanlight panel's settle before each exposure


class SensorCalibrationDialog(QDialog):
    """Pick three bare-light exposures, compute the unmix matrix, save a profile."""

    profile_saved = pyqtSignal(str)

    def __init__(self, parent=None, start_dir: str = "", *, repo=None, controller=None) -> None:
        super().__init__(parent)
        self._start_dir = start_dir
        self._paths = {"R": "", "G": "", "B": ""}
        self._repo = repo
        self._controller = controller  # None = no tethered capture; the files are the only source
        self._capture = None  # the running SensorResponseRequest
        self._presence = None  # (camera, Scanlight) from the last poll; None until one answers
        self._presence_pending = False
        self.setWindowTitle(tr("Calibrate Sensor"))
        self.resize(560, 320)
        self._init_ui()
        if controller is not None:
            controller.capture_sensor_response_progress.connect(self._on_capture_progress)
            controller.capture_sensor_response_measured.connect(self._on_capture_measured)
            controller.capture_sensor_response_failed.connect(self._on_capture_failed)
            controller.capture_presence_polled.connect(self._on_presence)
            self._presence_timer = QTimer(self)
            self._presence_timer.setInterval(_PRESENCE_POLL_MS)
            self._presence_timer.timeout.connect(self._poll_presence)
        remember_dialog_geometry(self, repo, "sensor_calibration")

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.addWidget(
            hint_label(
                tr(
                    "Pick three <b>bare-light</b> exposures, red only, green only and blue only, with no film in the holder and the light and camera settings you scan with. Expose just below clipping. With a tethered camera and a Scanlight, <b>Capture from Camera…</b> shoots them for you."
                )
            )
        )

        self._path_edits = {}
        for band, label in _BANDS:
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText(tr("Choose a capture…"))
            edit.setToolTip(tr("The exposure shot with this color of light alone and no film in the holder."))
            browse = QPushButton(tr("Browse…"))
            browse.setToolTip(tr("Choose the exposure file for this color."))
            browse.clicked.connect(lambda _c=False, b=band: self._browse(b))
            root.addLayout(field_row(label, edit, browse))
            self._path_edits[band] = edit

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(tr("e.g. X-T30 + Scanlight v4"))
        self.name_edit.setToolTip(tr("The profile's name in the sensor Profile dropdown: name the camera and the light."))
        self.name_edit.textChanged.connect(self._refresh)
        root.addLayout(field_row(tr("Name"), self.name_edit))

        self.result_label = hint_label()
        self.result_label.setWordWrap(True)
        self.result_label.setVisible(False)
        root.addWidget(self.result_label)
        root.addStretch()

        btn_row = QHBoxLayout()
        self.capture_btn = QPushButton(_CAPTURE_LABEL)
        self.capture_btn.clicked.connect(self._capture_and_save)
        self.capture_btn.setVisible(self._controller is not None)
        btn_row.addWidget(self.capture_btn)
        btn_row.addStretch()
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        self.compute_btn = QPushButton(tr("Compute and Save"))
        self.compute_btn.setToolTip(tr("Build the matrix from the three exposures and save it as a sensor profile."))
        self.compute_btn.clicked.connect(self._compute_and_save)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.compute_btn)
        pin_dialog_default(self.compute_btn, self.cancel_btn, self.capture_btn)
        root.addLayout(btn_row)
        self._refresh()

    def _browse(self, band: str) -> None:
        # The three bare-light exposures are shot in one go, so an empty band starts
        # where the bands already picked are.
        start = pick_start_dir(self._paths[band], *self._paths.values(), self._start_dir)
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select the {band} bare-light exposure", band=band),
            start,
            tr("Supported Images ({value})", value=get_supported_raw_wildcards()),
        )
        if path:
            self._paths[band] = path
            self._path_edits[band].setText(path)
            self._refresh()

    def _refresh(self) -> None:
        name = self.name_edit.text().strip()
        nameable = bool(name) and not SensorProfiles.is_bundled(name)
        capturing = self._capture is not None
        self.compute_btn.setEnabled(all(self._paths.values()) and nameable and not capturing)
        missing = self._missing_hardware()
        self.capture_btn.setEnabled(nameable and not capturing and not missing)
        self.capture_btn.setToolTip(wrap_tooltip(missing or _CAPTURE_TOOLTIP))
        self.capture_btn.setText(tr("Capturing…") if capturing else _CAPTURE_LABEL)
        self.name_edit.setEnabled(not capturing)

    def _decode(self, path: str) -> np.ndarray:
        # Sensor-native linear decode with neutral WB, the same as the flat-field reference. The
        # per-capture exposure scale cancels in build_sensor_matrix's normalization.
        from negpy.services.rendering.preview_manager import PreviewManager

        buf, _, _ = PreviewManager().load_linear_preview(path, use_camera_wb=False, full_resolution=False)
        return buf

    def _compute_and_save(self) -> None:
        name = self.name_edit.text().strip()
        if not name or not all(self._paths.values()):
            return
        # An override cursor set inside this modal dialog can crash Qt on macOS while it
        # builds the cursor image, so the button shows the busy state instead.
        self.compute_btn.setEnabled(False)
        self.compute_btn.setText(tr("Computing…"))
        self.compute_btn.repaint()
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
            self._show_result(tr("Could not build the matrix: {exc}", exc=exc))
            return
        finally:
            self.compute_btn.setText(tr("Compute and Save"))
            self._refresh()

        SensorProfiles.save(name, list(matrix))
        self._show_result(self._summary(name, measured, clipped))
        self._pin_close()
        self.profile_saved.emit(name)

    def _light_port(self) -> str:
        settings = self._repo.get_global_setting("scanlight_settings", default={}) if self._repo is not None else {}
        return str(settings.get("port", "")) if isinstance(settings, dict) else ""

    def _missing_hardware(self) -> str:
        """Why Capture from Camera cannot run, "" when it can."""
        if self._presence is None:
            return tr("Looking for the camera and the Scanlight.")
        camera, light = self._presence
        if camera and light:
            return ""
        if not camera and not light:
            return tr("Connect the camera and the Scanlight to use it.")
        return tr("Connect the camera to use it.") if not camera else tr("Connect the Scanlight to use it.")

    def _poll_presence(self) -> None:
        # The worker runs one job at a time, so a poll sent during a capture only queues.
        if self._capture is None and not self._presence_pending:
            self._presence_pending = True
            self._controller.poll_capture_presence(self._light_port())

    def _on_presence(self, camera: bool, light: bool) -> None:
        self._presence_pending = False
        self._presence = (camera, light)
        self._refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._controller is not None:
            self._poll_presence()
            self._presence_timer.start()

    def _capture_and_save(self) -> None:
        from negpy.desktop.workers.capture_worker import SensorResponseRequest

        if not confirm_sensor_capture(self):
            return
        self._capture = SensorResponseRequest(port=self._light_port(), settle_s=_LED_SETTLE_S)
        self._refresh()
        self._show_result(tr("Waiting for the camera…"))
        self._controller.start_sensor_response(self._capture)

    def _on_capture_progress(self, _frac: float, msg: str) -> None:
        if self._capture is not None:
            self._show_result(msg)

    def _on_capture_measured(self, response) -> None:
        if self._capture is None:
            return
        self._capture = None
        self._refresh()
        name = self.name_edit.text().strip()
        measured = {band: tuple(float(v) for v in np.asarray(response)[:, j]) for j, band in enumerate("RGB")}
        try:
            matrix = build_sensor_matrix(measured["R"], measured["G"], measured["B"])
        except ValueError as exc:
            self._show_result(tr("Could not build the matrix: {exc}", exc=exc))
            return
        SensorProfiles.save(name, list(matrix))
        self._show_result(self._summary(name, measured, []))
        self._pin_close()
        self.profile_saved.emit(name)

    def _on_capture_failed(self, msg: str) -> None:
        if self._capture is None:
            return
        self._capture = None
        self._refresh()
        self._show_result(tr("Could not capture: {msg}", msg=msg))

    def done(self, result: int) -> None:
        # Esc and the close box reach here without a button click.
        if self._capture is not None:
            self._capture.cancel.set()
            self._capture = None
        if self._controller is not None:
            self._presence_timer.stop()
        super().done(result)

    def _pin_close(self) -> None:
        # The dialog stays open to show the measured leakage; with the profile saved there
        # is nothing left to cancel.
        self.cancel_btn.setText(tr("Close"))
        self.compute_btn.setDefault(False)
        self.compute_btn.setProperty("primary", False)
        pin_dialog_default(self.cancel_btn, self.compute_btn, self.capture_btn)
        for btn in (self.cancel_btn, self.compute_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _summary(self, name: str, measured: dict, clipped: list) -> str:
        s = np.column_stack([measured["R"], measured["G"], measured["B"]])
        s_norm = s / np.diag(s)
        gb, bg = s_norm[2, 1], s_norm[1, 2]  # green->blue / blue->green, the usual dominant leaks
        lines = [tr("<b>Saved “{name}”</b>: measured green↔blue leakage {gb:.0f}% / {bg:.0f}%.", name=name, gb=gb * 100, bg=bg * 100)]
        if clipped:
            lines.append(
                tr(
                    "The {bands} {value} clipped. Shoot again, dimmer, for an accurate matrix.",
                    value=plural(len(clipped), "exposure"),
                    bands="/".join(clipped),
                )
            )
        return "<br>".join(lines)

    def _show_result(self, text: str) -> None:
        self.result_label.setText(text)
        self.result_label.setVisible(True)
