"""Sensor calibration dialog: Compute and Save shows its busy state on the button, never
through an override cursor."""

import numpy as np
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
