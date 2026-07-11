"""Smoke tests that the Scanlight capture sidebar actually builds (not just imports).

These instantiate the widget with a mock controller so a typo in `_init_ui` /
`_update_settings_from_ui` (e.g. a bad THEME attribute) fails CI instead of only
at app launch.
"""

import sys
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.scanlight import ScanlightSidebar

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _sidebar():
    ctrl = MagicMock()
    ctrl.session.repo.get_global_setting.return_value = {}
    return ScanlightSidebar(ctrl)


def _poll(usb_ok=False, usb_model="", light_ok=True, light_detail="fw"):
    return {
        "usb_ok": usb_ok,
        "usb_model": usb_model,
        "light_ok": light_ok,
        "light_detail": light_detail,
    }


def test_sidebar_builds_with_all_controls():
    w = _sidebar()
    for attr in (
        "r_slider",
        "g_slider",
        "b_slider",
        "w_slider",
        "shutter_edit",
        "lv_btn",
        "lv_image",
        "cam_status",
        "light_status",
        "preset_new_btn",
        "calib_window",
        "gate_hint",
    ):
        assert hasattr(w, attr), attr


def test_normal_mode_relaxes_gate_to_camera_and_folder(tmp_path):
    w = _sidebar()
    w.folder_edit.setText(str(tmp_path))
    w._camera_verified = True
    # RGB mode (default): the Scanlight + a preset are required → still blocked.
    assert "connect the Scanlight" in " ".join(w._missing_requirements())
    # No Scanlight → normal white-light mode: only camera + output are needed.
    w._set_rgb_mode(False)
    assert not w._rgb_mode
    assert w._missing_requirements() == []


def test_normal_mode_clears_stale_light_status():
    w = _sidebar()
    w._on_light_set(255, 118, 86, 0)  # RGB framing set "Light: R255 G118 B86"
    assert "Light:" in w.status_label.text()
    w._set_rgb_mode(False)  # Scanlight unplugged → normal mode
    assert w.status_label.text() == ""  # the stale light status is dropped


def test_normal_mode_capture_request_is_single_no_triplet(tmp_path):
    w = _sidebar()
    w.folder_edit.setText(str(tmp_path))
    w._camera_verified = True
    w._set_rgb_mode(False)
    w._start_capture(retake=False)
    req = w.controller.start_capture.call_args[0][0]
    assert not req.rgb_mode and not req.white_mode  # one plain white-light frame, not the RGB triplet


def test_update_settings_from_ui_reads_every_widget():
    w = _sidebar()
    w.r_slider.setValue(123)
    w.w_slider.setValue(77)
    w._update_settings_from_ui()
    assert w._settings.r_level == 123
    assert w._settings.w_level == 77


def test_live_view_popup_has_capture_toolbar():
    w = _sidebar()
    for attr in ("scan_btn", "retake_btn", "status"):
        assert hasattr(w.lv_window, attr), attr
    assert not hasattr(w.lv_window, "zoom_btn")  # digital zoom removed
    assert not hasattr(w.lv_window, "mag_btn")  # magnifier button removed (click-to-magnify)


def test_magnifier_click_aims_camera_and_maps_to_grid():
    w = _sidebar()
    w.lv_btn.blockSignals(True)
    w.lv_btn.setChecked(True)  # pretend live view is streaming
    w.lv_btn.blockSignals(False)
    w._on_magnifier_click(0.5, 0.5)  # centre → (320, 240) on the 640×480 grid
    w.controller.set_focus_magnifier_pos.assert_called_with(320, 240)
    assert w._magnifier_on  # click turns the magnifier on
    w._magnifier_on = False  # pretend it's off again, to check the corner maps too
    w.controller.set_focus_magnifier_pos.reset_mock()
    w._on_magnifier_click(0.0, 0.0)  # top-left corner → (0, 0)
    w.controller.set_focus_magnifier_pos.assert_called_with(0, 0)


def test_magnifier_click_ignored_when_not_streaming():
    w = _sidebar()
    w._on_magnifier_click(0.5, 0.5)  # live view off → no-op
    assert not w.controller.set_focus_magnifier_pos.called


def test_builtin_white_preset_sets_white_mode():
    w = _sidebar()
    idx = w.preset_combo.findData("White Light (B&W or Slide Film)")
    w.preset_combo.setCurrentIndex(idx)
    w._on_preset_selected(idx)
    assert w._settings.white_mode is True
    assert w._settings.white_process_mode == "auto"  # B&W/slide merged → NegPy autodetects


def test_white_preset_hint_shows_then_clears():
    w = _sidebar()
    idx = w.preset_combo.findData("White Light (B&W or Slide Film)")
    w.preset_combo.setCurrentIndex(idx)
    w._on_preset_selected(idx)
    assert not w.preset_hint.isHidden()  # note appears under the preset row (not in camera status)
    assert "white-light" in w.preset_hint.text().lower()
    # Deselecting the preset must clear the note (it used to linger in the camera status line).
    w.preset_combo.setCurrentIndex(0)  # "— Select preset —" (data=None)
    w._on_preset_selected(0)
    assert w.preset_hint.text() == ""
    assert w.preset_hint.isHidden()


def test_set_scanning_mirrors_to_popup_button():
    w = _sidebar()
    w.set_scanning(True)
    assert "Stop" in w.lv_window.scan_btn.text()
    w.set_scanning(False)
    assert "Scan" in w.lv_window.scan_btn.text()


def test_cancelled_scan_returns_sidebar_to_a_terminal_idle_state():
    w = _sidebar()
    w.set_scanning(True)
    w.progress_bar.setValue(67)
    w.lv_btn.blockSignals(True)
    w.lv_btn.setChecked(True)
    w.lv_btn.blockSignals(False)
    w.controller.set_scanlight_color.reset_mock()

    w._on_cancelled()

    assert not w._scanning
    assert w.progress_bar.isHidden()
    assert "cancelled" in w.status_label.text().lower()
    assert w.lv_btn.isChecked()  # capture cancellation preserves the live-view session
    assert w.controller.set_scanlight_color.called  # restore the framing light


def test_cancelled_calibration_restores_scan_target_and_gates():
    w = _sidebar()
    w._camera_verified = True
    w._light_verified = True
    w._calibrating_preset = "Portra 400"
    w._lv_target = w.calib_window.image
    w.calib_window.set_progress(0.67)
    w._lv_timer.start()
    w._apply_gating()
    assert not w.preset_new_btn.isEnabled()

    w._on_cancelled()

    assert w._calibrating_preset == ""
    assert w._lv_target is w.lv_image
    assert w.calib_window.progress.isHidden()
    assert "cancelled" in w.calib_window.status.text().lower()
    assert not w._lv_timer.isActive()
    w.controller.stop_live_view.assert_called_once_with()
    assert w.preset_new_btn.isEnabled()
    assert w.calib_window.calibrate_btn.isEnabled()


def test_closing_running_calibration_waits_for_worker_terminal_signal():
    w = _sidebar()
    w._calibrating_preset = "Portra 400"
    w._lv_target = w.calib_window.image

    w._on_calib_window_closed()

    w.controller.cancel_capture.assert_called_once_with()
    assert w._calibrating_preset == "Portra 400"  # worker still owns the capture session

    w._on_cancelled()

    assert w._calibrating_preset == ""
    assert w._lv_target is w.lv_image


def test_capture_error_closes_the_stale_live_view_session():
    w = _sidebar()
    w.set_scanning(True)
    w.lv_btn.blockSignals(True)
    w.lv_btn.setChecked(True)
    w.lv_btn.blockSignals(False)
    w.lv_window.show()
    w._lv_timer.start()
    w.controller.stop_live_view.reset_mock()

    w._on_error("camera disconnected")

    assert not w._scanning
    assert w.progress_bar.isHidden()
    assert not w.lv_btn.isChecked()
    assert w.lv_window.isHidden()
    assert not w._lv_timer.isActive()
    w.controller.stop_live_view.assert_called_once_with()
    assert "camera disconnected" in w.status_label.text().lower()


def test_status_is_mirrored_into_popup():
    w = _sidebar()
    w._set_status("metering base…")
    assert w.lv_window.status.text() == "metering base…"


def test_popup_scan_signal_triggers_capture(tmp_path):
    w = _sidebar()
    w.folder_edit.setText(str(tmp_path))
    w.lv_window.scanRequested.emit()  # the pop-up's Scan button
    assert w.controller.start_capture.called


def test_scan_gated_until_all_requirements(tmp_path):
    w = _sidebar()
    # Nothing ready → "Live View & Scan" and the pop-up Scan are disabled.
    assert not w.lv_btn.isEnabled()
    assert not w.lv_window.scan_btn.isEnabled()
    # Satisfy folder + preset + camera + light → enabled.
    w.folder_edit.setText(str(tmp_path))
    w.preset_combo.setCurrentIndex(w.preset_combo.findData("White Light (B&W or Slide Film)"))
    w._on_poll_status(_poll(usb_ok=True, usb_model="FAKE-1"))
    assert w.lv_btn.isEnabled()
    assert w.lv_window.scan_btn.isEnabled()
    assert w.gate_hint.text() == ""


def test_gate_hint_lists_missing_requirements():
    w = _sidebar()
    w._on_poll_status(_poll(usb_ok=True))  # camera + light ok; folder + preset still missing
    assert "output folder" in w.gate_hint.text()
    assert "preset" in w.gate_hint.text()
    assert not w.lv_btn.isEnabled()


def test_new_preset_button_needs_only_camera_and_light():
    w = _sidebar()
    assert not w.preset_new_btn.isEnabled()
    w._on_poll_status(_poll(usb_ok=True))  # no folder / preset yet
    assert w.preset_new_btn.isEnabled()  # calibration can create the first preset


def test_magnifier_click_toggles_back_to_full_frame():
    w = _sidebar()
    w.lv_btn.blockSignals(True)
    w.lv_btn.setChecked(True)
    w.lv_btn.blockSignals(False)
    w._on_magnifier_click(0.5, 0.5)  # first click → magnify at that spot
    assert w._magnifier_on
    w.controller.set_focus_magnifier_pos.reset_mock()
    w._on_magnifier_click(0.2, 0.8)  # second click anywhere → back to the full frame
    w.controller.set_focus_magnifier.assert_called_with(False)
    assert not w._magnifier_on
    assert not w.controller.set_focus_magnifier_pos.called  # it does not re-aim on the way out


def test_reset_magnifier_clears_state():
    w = _sidebar()
    w._magnifier_on = True
    w._reset_magnifier()
    assert not w._magnifier_on


def test_camera_settings_populate_and_set(tmp_path, monkeypatch):
    import json

    import negpy.desktop.view.sidebar.scanlight as sl

    p = tmp_path / "settings.json"
    p.write_text(
        json.dumps(
            {
                "iso": {
                    "cur": 200,
                    "label": "ISO 200",
                    "writable": True,
                    "options": [{"raw": 100, "label": "ISO 100"}, {"raw": 200, "label": "ISO 200"}],
                },
                "aperture": None,  # unavailable (manual lens)
            }
        )
    )
    monkeypatch.setattr(sl, "default_settings_path", lambda: str(p))
    w = _sidebar()
    w._refresh_camera_settings()
    assert w.lv_window.iso_stepper.count() == 2
    assert w.lv_window.iso_stepper.currentData() == 200  # reflects the camera's current value
    assert not w.lv_window.aperture_stepper.isEnabled()  # unavailable → disabled
    # user steps ISO to 100 → controller gets the raw value
    w.lv_window.iso_stepper.setCurrentIndex(w.lv_window.iso_stepper.findData(100))
    w._on_camera_setting("iso", w.lv_window.iso_stepper)
    w.controller.set_camera_setting.assert_called_with("iso", 100)


def test_setting_stepper_steps_and_clamps():
    from negpy.desktop.view.sidebar.live_view_window import SettingStepper

    s = SettingStepper()
    for raw, label in ((100, "ISO 100"), (200, "ISO 200"), (400, "ISO 400")):
        s.addItem(label, raw)
    s.setCurrentIndex(s.findData(200))
    seen = []
    s.activated.connect(lambda _i: seen.append(s.currentData()))
    s._step(1)  # → 400
    assert s.currentData() == 400
    s._step(1)  # already at the top → clamped, no change, no emit
    assert s.currentData() == 400
    s._step(-1)  # → 200
    assert s.currentData() == 200
    assert seen == [400, 200]  # only real moves emit an `activated`


def test_scan_button_is_bold_scan():
    w = _sidebar()
    assert w.lv_btn.text().strip() == "Scan"  # renamed from "Live View & Scan"
    assert w.lv_btn.font().bold()


def test_poll_clears_stale_searching_status_on_connect():
    w = _sidebar()  # USB mode
    w._camera_verified = False
    w._set_status("Camera disconnected.")
    w._on_poll_status(_poll(usb_ok=True, usb_model="ZV-E1"))  # USB body appears → connected
    assert w._camera_verified
    assert w.status_label.text() == ""  # the stale failure line is dropped on connect


def test_poll_finds_usb_camera_marks_green():
    w = _sidebar()  # USB mode by default
    w._conn_poll_inflight = True
    w._on_poll_status(_poll(usb_ok=True, usb_model="ZV-E1"))
    assert w._camera_verified and w._light_verified
    assert not w._conn_poll_inflight  # cleared so the next tick can run
    assert "USB" in w.cam_status.text()  # transport shown in the label


def test_disconnect_during_live_view_closes_the_preview():
    # Enumerating the bus keeps running through live view, so an unplug is caught there
    # too — and the last frame must not be left on screen looking live.
    w = _sidebar()
    w._camera_verified = True
    w.lv_btn.blockSignals(True)
    w.lv_btn.setChecked(True)
    w.lv_btn.blockSignals(False)
    w._on_poll_status(_poll(usb_ok=False))
    assert not w._camera_verified
    assert not w.lv_btn.isChecked()
    assert "disconnected" in w.status_label.text().lower()


def test_poll_no_usb_in_usb_mode_marks_not_connected():
    w = _sidebar()  # USB mode
    w._camera_verified = True
    w._on_poll_status(_poll(usb_ok=False))  # nothing plugged in
    assert not w._camera_verified
    assert w._light_verified


def test_frame_number_auto_derived_from_roll_subfolder(tmp_path):
    w = _sidebar()
    w.folder_edit.setText(str(tmp_path))
    w.roll_edit.setText("Roll007")
    roll_dir = tmp_path / "Roll007"

    def captured_req():
        return w.controller.start_capture.call_args[0][0]

    # Empty output → the first fresh scan is frame 1, targeting a per-roll subfolder.
    w._start_capture(retake=False)
    w.set_scanning(False)  # the finished signal would do this
    assert captured_req().frame_number == 1
    assert captured_req().output_folder == str(roll_dir)  # Output/Roll007/
    # Two triplets already in the roll's subfolder → the next fresh scan is frame 3.
    roll_dir.mkdir(exist_ok=True)
    for n in (1, 2):
        for ch in ("R", "G", "B"):
            (roll_dir / f"Roll007_Frame{n:03d}_{ch}.ARW").write_bytes(b"x")
    w.controller.start_capture.reset_mock()
    w._start_capture(retake=False)
    w.set_scanning(False)  # the finished signal would do this
    assert captured_req().frame_number == 3
    # A retake re-shoots the last frame (2), not a new number.
    w.controller.start_capture.reset_mock()
    w._start_capture(retake=True)
    w.set_scanning(False)  # the finished signal would do this
    assert captured_req().frame_number == 2


@pytest.mark.parametrize(
    "roll_name",
    [
        pytest.param("2026/07", id="posix-separator"),
        pytest.param("/Volumes/scans/Roll007", id="posix-absolute"),
        pytest.param(r"2026\07", id="windows-separator"),
        pytest.param(r"C:\Scans\Roll007", id="windows-absolute"),
        pytest.param(".", id="current-directory"),
        pytest.param("..", id="parent-directory"),
        pytest.param("Roll\0escape", id="nul"),
    ],
)
def test_capture_rejects_unsafe_roll_name(tmp_path, roll_name):
    w = _sidebar()
    w.folder_edit.setText(str(tmp_path))
    w.roll_edit.setText(roll_name)

    w._start_capture(retake=False)

    assert not w.controller.start_capture.called
    assert "single safe name" in w.status_label.text().lower()


def test_blank_roll_name_falls_back_consistently(tmp_path):
    w = _sidebar()
    w.folder_edit.setText(str(tmp_path))
    w.roll_edit.clear()

    w._start_capture(retake=False)

    req = w.controller.start_capture.call_args.args[0]
    assert req.roll_name == "Roll001"
    assert req.output_folder == str(tmp_path / "Roll001")
    assert w.roll_edit.text() == "Roll001"
    assert w._settings.roll_name == "Roll001"
    key, saved = w.controller.session.repo.save_global_setting.call_args.args
    assert key == "scanlight_settings"
    assert saved["roll_name"] == "Roll001"


def test_safe_roll_name_is_trimmed_consistently(tmp_path):
    w = _sidebar()
    w.folder_edit.setText(str(tmp_path))
    w.roll_edit.setText("  Summer 2026  ")

    w._start_capture(retake=False)

    req = w.controller.start_capture.call_args.args[0]
    assert req.roll_name == "Summer 2026"
    assert req.output_folder == str(tmp_path / "Summer 2026")
    assert w.roll_edit.text() == "Summer 2026"
    assert w._settings.roll_name == "Summer 2026"
    key, saved = w.controller.session.repo.save_global_setting.call_args.args
    assert key == "scanlight_settings"
    assert saved["roll_name"] == "Summer 2026"


def test_temp_label_hides_when_no_reading():
    w = _sidebar()
    assert w.light_temp.isHidden()  # starts hidden — an empty label paints a dark #0D0D0D box
    w._on_light_temp(42.0)  # a reading arrives
    assert not w.light_temp.isHidden() and "42" in w.light_temp.text()
    w._on_light_temp(None)  # light unplugged / no telemetry
    assert w.light_temp.isHidden() and w.light_temp.text() == ""  # no lingering placeholder box


def test_setup_hint_toggles_with_gphoto2_availability(monkeypatch):
    w = _sidebar()
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    w._refresh_setup_hint()
    assert not w._setup_hint.isHidden()  # python-gphoto2 missing → the setup hint shows
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
    w._refresh_setup_hint()
    assert w._setup_hint.isHidden()  # installed → hint hidden (never nags an equipped user)


def test_scan_is_blocked_while_a_calibration_runs():
    """The worker runs one job at a time. A scan clicked mid-calibration would only queue,
    then fire with the exposure the calibration was about to replace — and cancelling the
    calibration would not stop it."""
    w = _sidebar()
    w._calibrating_preset = "Portra 400"
    assert "wait for the calibration to finish" in w._missing_requirements()
    w._apply_gating()
    assert not w.lv_window.scan_btn.isEnabled()
    assert not w.lv_window.retake_btn.isEnabled()
    assert "calibration" in w.gate_hint.text().lower()

    w._calibrating_preset = ""
    assert "wait for the calibration to finish" not in w._missing_requirements()


def test_calibration_is_blocked_while_a_scan_runs():
    w = _sidebar()
    w._on_poll_status(_poll(usb_ok=True))
    assert w.preset_new_btn.isEnabled()  # camera + light present
    w.set_scanning(True)
    assert not w.preset_new_btn.isEnabled()
    w.set_scanning(False)
    assert w.preset_new_btn.isEnabled()


def test_calibrate_button_in_an_open_popup_is_refused_during_a_scan():
    """The '+' button only *opens* the pop-up. If it was already open, its own
    "Calibrate & Save" button is the real trigger — and it must refuse mid-scan."""
    w = _sidebar()
    w.set_scanning(True)
    w._on_calibrate_new_preset("Portra 400")
    assert not w.controller.start_calibration.called
    assert "scan is running" in w.calib_window.status.text().lower()
    assert not w.calib_window.calibrate_btn.isEnabled()


def test_scan_is_refused_while_a_calibration_runs(tmp_path):
    w = _sidebar()
    w.folder_edit.setText(str(tmp_path))
    w._calibrating_preset = "Portra 400"
    w._start_capture(retake=False)
    assert not w.controller.start_capture.called
    assert "calibration is running" in w.status_label.text().lower()


def test_a_second_scan_click_does_not_queue_another_frame(tmp_path):
    w = _sidebar()
    w.folder_edit.setText(str(tmp_path))
    w.set_scanning(True)
    w._start_capture(retake=False)
    assert not w.controller.start_capture.called
