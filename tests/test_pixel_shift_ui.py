"""Pixel Shift in the camera scan panel, the Live View window and the capture worker."""

import os

from negpy.services.capture.pixel_shift import FRAMES_FOLDER
import sys
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest
import tifffile
from PyQt6.QtWidgets import QApplication

import negpy.desktop.workers.capture_worker as capture_worker_module
from negpy.desktop.view.sidebar.scanlight import ScanlightSidebar
from negpy.desktop.workers.capture_worker import CaptureRequest, CaptureWorker
from negpy.services.capture.service import CaptureError

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _sidebar(monkeypatch, counts="missing"):
    ctrl = MagicMock()
    ctrl.session.repo.get_global_setting.return_value = {}
    w = ScanlightSidebar(ctrl)
    w._exposure_chosen = True
    w._camera_verified = True
    data = {"iso": {"options": [{"label": "100", "raw": 0}], "cur": 0}}
    if counts != "missing":
        data["pixel_shift"] = counts
    monkeypatch.setattr(w, "_settings_json", lambda: data)
    w._refresh_camera_settings()
    return w


def _saved(w):
    return w.controller.session.repo.save_global_setting.call_args_list[-1].args[1]


@pytest.mark.parametrize("counts", ["missing", None])
def test_the_row_hides_on_a_body_without_pixel_shift(monkeypatch, counts):
    w = _sidebar(monkeypatch, counts)
    assert w.pixel_shift_row.isHidden()
    assert w.lv_window.pixel_shift_col.isHidden()


def test_the_row_grays_and_says_why_while_the_body_has_it_off(monkeypatch):
    w = _sidebar(monkeypatch, [])
    assert not w.pixel_shift_row.isHidden()
    for choice in (w.pixel_shift_choice, w.lv_window.pixel_shift_choice):
        assert not choice.isEnabled()
        assert "on the camera" in choice.toolTip()
    w._set_pixel_shift_counts([4, 16])  # the next poll after the user turns it on at the body
    assert w.pixel_shift_choice.isEnabled() and "on the camera" not in w.pixel_shift_choice.toolTip()


def test_the_choice_locks_while_a_scan_runs(monkeypatch):
    w = _sidebar(monkeypatch, [4, 16])
    w.set_scanning(True)
    assert not w.pixel_shift_choice.isEnabled() and not w.lv_window.pixel_shift_choice.isEnabled()
    w.set_scanning(False)
    assert w.pixel_shift_choice.isEnabled()


def test_only_the_listed_counts_are_offered(monkeypatch):
    w = _sidebar(monkeypatch, [4])
    for choice in (w.pixel_shift_choice, w.lv_window.pixel_shift_choice):
        assert choice.isEnabled()
        assert [choice.is_choice_enabled(i) for i in range(choice.count())] == [True, True, False]


def test_the_choice_lands_in_the_request_and_the_saved_settings(monkeypatch, tmp_path):
    w = _sidebar(monkeypatch, [4, 16])
    assert w.pixel_shift_rail.isHidden()
    w.pixel_shift_choice.setCurrentIndex(w.pixel_shift_choice.findData(16))
    w.keep_shift_btn.setChecked(True)
    assert not w.pixel_shift_rail.isHidden()
    assert _saved(w)["pixel_shift"] == 16 and _saved(w)["keep_shift_frames"] is True

    w.output.folder_edit.setText(str(tmp_path))
    w._set_rgb_mode(False)
    w._start_capture(retake=False)
    req = w.controller.start_capture.call_args[0][0]
    assert (req.pixel_shift, req.keep_shift_frames) == (16, True)


def test_a_body_that_stops_offering_the_saved_count_shoots_one(monkeypatch, tmp_path):
    w = _sidebar(monkeypatch, [4])
    w._settings = replace(w._settings, pixel_shift=16)
    w.output.folder_edit.setText(str(tmp_path))
    w._set_rgb_mode(False)
    w._start_capture(retake=False)
    assert w.controller.start_capture.call_args[0][0].pixel_shift == 1


def test_blend_grays_the_row_and_shoots_one(monkeypatch):
    w = _sidebar(monkeypatch, [4, 16])
    w._settings = replace(w._settings, pixel_shift=4)
    monkeypatch.setattr(w, "_blend_active", lambda: True)
    w._refresh_pixel_shift()
    assert not w.pixel_shift_choice.isEnabled()
    assert w._pixel_shift() == 1


def test_the_live_view_choice_follows_the_panel_and_back(monkeypatch):
    w = _sidebar(monkeypatch, [4, 16])
    lv = w.lv_window.pixel_shift_choice
    assert not w.lv_window.pixel_shift_col.isHidden()
    w.pixel_shift_choice.setCurrentIndex(w.pixel_shift_choice.findData(4))
    assert lv.currentData() == 4
    lv.setCurrentIndex(lv.findData(16))
    assert w.pixel_shift_choice.currentData() == 16
    assert w._settings.pixel_shift == 16


class _Camera:
    pixel_shift_maker = "sony"

    def capture(self, out_path, shutter=None, iso=None, aperture=None):
        path = os.path.splitext(out_path)[0] + ".ARW"
        with open(path, "wb") as raw:
            raw.truncate(8 * 1024 * 1024)
        return path


class _Light:
    def set_color(self, **_channels):
        pass

    def off(self):
        pass


@pytest.mark.parametrize(
    ("pixel_shift", "blend", "wrapped"),
    [(1, False, False), (4, False, True), (16, True, False)],
)
def test_the_worker_wraps_the_camera_only_for_a_set(tmp_path, monkeypatch, pixel_shift, blend, wrapped):
    made = []

    class FakeShift:
        def __init__(self, camera, positions, *, frames_folder, keep_frames, fallback, report=None, part=None, cancel=None):
            made.append((len(positions), frames_folder, keep_frames, fallback, cancel))
            self._camera = camera

        def capture(self, *args, **kwargs):
            return self._camera.capture(*args, **kwargs)

    monkeypatch.setattr(capture_worker_module, "PixelShiftCamera", FakeShift)
    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: _Camera())
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: _Light())
    monkeypatch.setattr(worker, "_report_clipping", lambda _path: None)
    monkeypatch.setattr(capture_worker_module, "_blend_seconds", lambda _camera, _shutter: 1.0)
    monkeypatch.setattr(capture_worker_module.CaptureService, "capture_blend", lambda self, *a, **k: str(tmp_path / "blend.ARW"))
    finished = []
    worker.finished.connect(finished.append)

    worker.run_capture(
        CaptureRequest(
            roll_name="Roll01",
            frame_number=1,
            output_folder=str(tmp_path),
            levels=(200, 180, 255),
            settle_s=0,
            single_capture=not blend,
            blend_capture=blend,
            pixel_shift=pixel_shift,
            keep_shift_frames=True,
        )
    )

    assert finished
    assert made == ([(pixel_shift, os.path.join(str(tmp_path), FRAMES_FOLDER), True, True, worker._cancel)] if wrapped else [])


@pytest.mark.parametrize(
    ("sibling_shots", "pixel_shift", "refused"),
    [
        (None, 4, True),  # a plain ARW beside a set
        (4, 1, True),
        (4, 4, False),
        (4, 16, True),  # both merged, of two sizes
        (0, 1, False),  # a body's own DNG is a plain frame
        (None, 1, False),
    ],
)
def test_one_exposure_is_refused_under_another_setting_than_its_siblings(tmp_path, sibling_shots, pixel_shift, refused):
    """`sibling_shots`: None for an ARW, 0 for a DNG the body wrote, else the set a merged DNG came from."""
    if sibling_shots is None:
        (tmp_path / "Roll01_Frame001_G.ARW").write_text("x")
    else:
        software = f"NegPy Pixel Shift {sibling_shots}" if sibling_shots else "a camera"
        tifffile.imwrite(tmp_path / "Roll01_Frame001_G.dng", np.zeros((2, 2, 3), np.uint16), software=software)
    req = CaptureRequest(
        roll_name="Roll01", frame_number=1, output_folder=str(tmp_path), levels=(1, 1, 1), channels=("R",), pixel_shift=pixel_shift
    )
    if refused:
        with pytest.raises(CaptureError, match="Retake the whole frame"):
            CaptureWorker()._check_sibling_formats(req, pixel_shift > 1)
    else:
        CaptureWorker()._check_sibling_formats(req, pixel_shift > 1)
