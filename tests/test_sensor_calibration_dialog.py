"""Sensor calibration dialog: Compute and Save shows its busy state on the button, never
through an override cursor. Capture from Camera runs on the controller's capture worker."""

import numpy as np
import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QGuiApplication

from negpy.desktop.view.widgets import sensor_calibration_dialog as mod
from negpy.desktop.view.widgets.sensor_calibration_dialog import SensorCalibrationDialog

_CAPTURES = {"r.dng": (0.9, 0.1, 0.03), "g.dng": (0.05, 0.5, 0.15), "b.dng": (0.04, 0.3, 1.0)}


def _dialog(monkeypatch, seen: list, fail: bool = False) -> SensorCalibrationDialog:
    dlg = SensorCalibrationDialog()

    def _decode(path):
        seen.append((dlg.compute_btn.text(), dlg.compute_btn.isEnabled(), QGuiApplication.overrideCursor()))
        if fail:
            raise ValueError("unreadable")
        return np.full((32, 32, 3), _CAPTURES[path], dtype=np.float32)

    monkeypatch.setattr(dlg, "_decode", _decode)
    monkeypatch.setattr(mod.SensorProfiles, "save", lambda name, matrix: None)
    dlg.name_edit.setText("Test Rig")
    for band, path in zip(("R", "G", "B"), _CAPTURES):
        dlg._paths[band] = path
    dlg._refresh()
    return dlg


def test_compute_shows_busy_on_the_button(qapp, monkeypatch):
    seen: list = []
    dlg = _dialog(monkeypatch, seen)
    dlg._compute_and_save()
    assert seen and all(s == ("Computing…", False, None) for s in seen)
    assert dlg.compute_btn.text() == "Compute and Save"
    assert dlg.compute_btn.isEnabled()


def test_a_failed_compute_restores_the_button(qapp, monkeypatch):
    seen: list = []
    dlg = _dialog(monkeypatch, seen, fail=True)
    dlg._compute_and_save()
    assert dlg.compute_btn.text() == "Compute and Save"
    assert dlg.compute_btn.isEnabled()
    assert "Could not build the matrix" in dlg.result_label.text()


def test_a_saved_profile_turns_cancel_into_the_default_close(qapp, monkeypatch):
    dlg = _dialog(monkeypatch, [])
    assert dlg.cancel_btn.text() == "Cancel" and dlg.compute_btn.isDefault()
    dlg._compute_and_save()
    assert dlg.cancel_btn.text() == "Close"
    assert dlg.cancel_btn.isDefault() and dlg.cancel_btn.property("primary")
    assert not dlg.compute_btn.isDefault() and not dlg.compute_btn.property("primary")
    assert dlg.compute_btn.isEnabled()


def test_a_failed_compute_keeps_cancel(qapp, monkeypatch):
    dlg = _dialog(monkeypatch, [], fail=True)
    dlg._compute_and_save()
    assert dlg.cancel_btn.text() == "Cancel" and dlg.compute_btn.isDefault()


class _Controller(QObject):
    capture_sensor_response_progress = pyqtSignal(float, str)
    capture_sensor_response_measured = pyqtSignal(object)
    capture_sensor_response_failed = pyqtSignal(str)
    capture_presence_polled = pyqtSignal(bool, bool)

    def __init__(self) -> None:
        super().__init__()
        self.requests: list = []
        self.polls = 0

    def poll_capture_presence(self, _port: str) -> None:
        self.polls += 1

    def start_sensor_response(self, req) -> None:
        self.requests.append(req)


def _tethered(monkeypatch, saved: list, *, confirm: bool = True):
    controller = _Controller()
    monkeypatch.setattr(mod, "confirm_sensor_capture", lambda _parent: confirm)
    dlg = SensorCalibrationDialog(controller=controller)
    monkeypatch.setattr(mod.SensorProfiles, "save", lambda name, matrix: saved.append((name, matrix)))
    dlg.name_edit.setText("Test Rig")
    controller.capture_presence_polled.emit(True, True)
    return dlg, controller


def test_capture_is_offered_only_with_a_controller(qapp):
    assert SensorCalibrationDialog().capture_btn.isHidden()
    assert not SensorCalibrationDialog(controller=_Controller()).capture_btn.isHidden()


def test_capture_needs_a_name_and_no_files(qapp, monkeypatch):
    dlg, _controller = _tethered(monkeypatch, [])
    assert dlg.capture_btn.isEnabled() and not dlg.compute_btn.isEnabled()
    dlg.name_edit.setText("")
    assert not dlg.capture_btn.isEnabled()


def test_a_measured_response_saves_the_profile(qapp, monkeypatch):
    saved: list = []
    dlg, controller = _tethered(monkeypatch, saved)
    names: list = []
    dlg.profile_saved.connect(names.append)
    dlg.capture_btn.click()
    assert len(controller.requests) == 1
    assert dlg.capture_btn.text() == "Capturing…" and not dlg.capture_btn.isEnabled() and not dlg.name_edit.isEnabled()
    controller.capture_sensor_response_progress.emit(0.3, "Measuring G…")
    assert dlg.result_label.text() == "Measuring G…"
    response = np.column_stack(list(_CAPTURES.values()))
    controller.capture_sensor_response_measured.emit(response)
    assert names == ["Test Rig"] and saved[0][0] == "Test Rig"
    assert saved[0][1] == pytest.approx(list(mod.build_sensor_matrix(*_CAPTURES.values())))
    assert dlg.cancel_btn.text() == "Close" and dlg.capture_btn.text() == "Capture from Camera…"
    assert dlg.capture_btn.isEnabled() and not dlg.capture_btn.isDefault()


def test_a_declined_confirmation_shoots_nothing(qapp, monkeypatch):
    dlg, controller = _tethered(monkeypatch, [], confirm=False)
    dlg.capture_btn.click()
    assert not controller.requests and dlg.capture_btn.isEnabled() and dlg.result_label.isHidden()


def test_a_failed_capture_reports_and_allows_a_retry(qapp, monkeypatch):
    saved: list = []
    dlg, controller = _tethered(monkeypatch, saved)
    dlg.capture_btn.click()
    controller.capture_sensor_response_failed.emit("No serial ports found. Is the Scanlight plugged in?")
    assert not saved and "Is the Scanlight plugged in?" in dlg.result_label.text()
    assert dlg.capture_btn.isEnabled() and dlg.cancel_btn.text() == "Cancel"


def test_closing_abandons_a_running_capture(qapp, monkeypatch):
    saved: list = []
    dlg, controller = _tethered(monkeypatch, saved)
    dlg.capture_btn.click()
    dlg.reject()
    assert controller.requests[0].cancel.is_set()
    controller.capture_sensor_response_measured.emit(np.column_stack(list(_CAPTURES.values())))
    assert not saved


@pytest.mark.parametrize(
    "camera, light, missing",
    [(False, True, "Connect the camera to"), (True, False, "Connect the Scanlight to"), (False, False, "the camera and the Scanlight")],
)
def test_capture_is_disabled_without_the_hardware(qapp, monkeypatch, camera, light, missing):
    dlg, controller = _tethered(monkeypatch, [])
    controller.capture_presence_polled.emit(camera, light)
    assert not dlg.capture_btn.isEnabled() and missing in dlg.capture_btn.toolTip()
    controller.capture_presence_polled.emit(True, True)
    assert dlg.capture_btn.isEnabled() and "Connect" not in dlg.capture_btn.toolTip()


def test_capture_waits_for_the_first_presence_answer(qapp):
    controller = _Controller()
    dlg = SensorCalibrationDialog(controller=controller)
    dlg.name_edit.setText("Test Rig")
    dlg.show()
    assert controller.polls == 1 and not dlg.capture_btn.isEnabled()
    dlg._poll_presence()
    assert controller.polls == 1  # one poll in flight at a time
    dlg.close()
    assert not dlg._presence_timer.isActive()
