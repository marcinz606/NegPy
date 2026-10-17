"""Capture Delay holds Scan and Retake before the first exposure; Stop ends the wait."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from unittest.mock import MagicMock

from negpy.desktop.workers import capture_worker as module
from negpy.desktop.workers.capture_worker import CaptureRequest, CaptureWorker
from negpy.infrastructure.capture.settings import ScanlightSettings


class _HeldCamera:
    def is_open(self) -> bool:
        return True

    def close(self) -> None:
        pass


def _worker(monkeypatch) -> CaptureWorker:
    worker = CaptureWorker()
    worker._camera = _HeldCamera()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: worker._camera)
    return worker


def _request(tmp_path, delay: float) -> CaptureRequest:
    return CaptureRequest(
        roll_name="Roll001", frame_number=1, output_folder=str(tmp_path), levels=(255, 255, 255), rgb_mode=False, capture_delay_s=delay
    )


def test_the_shot_waits_for_the_delay_and_counts_down(monkeypatch, tmp_path):
    worker = _worker(monkeypatch)
    shots: list[float] = []
    countdown: list[int] = []
    worker.countdown.connect(countdown.append)
    monkeypatch.setattr(module, "capture_single", lambda *a, **k: shots.append(time.monotonic()) or str(tmp_path / "f.ARW"))

    start = time.monotonic()
    worker.run_capture(_request(tmp_path, 0.6))

    assert shots and shots[0] - start >= 0.6
    assert countdown[0] == 1 and countdown[-1] == 0 and countdown == sorted(countdown, reverse=True)


def test_no_delay_means_no_countdown(monkeypatch, tmp_path):
    worker = _worker(monkeypatch)
    countdown: list[int] = []
    worker.countdown.connect(countdown.append)
    monkeypatch.setattr(module, "capture_single", lambda *a, **k: str(tmp_path / "f.ARW"))

    worker.run_capture(_request(tmp_path, 0.0))

    assert countdown == []


def test_a_missing_scanlight_fails_before_the_wait(monkeypatch, tmp_path):
    worker = _worker(monkeypatch)
    errors: list[str] = []
    worker.error.connect(errors.append)

    def _no_light(_port):
        raise RuntimeError("no Scanlight")

    monkeypatch.setattr(worker, "_ensure_light", _no_light)
    req = CaptureRequest(
        roll_name="Roll001", frame_number=1, output_folder=str(tmp_path), levels=(255, 255, 255), rgb_mode=True, capture_delay_s=5.0
    )

    start = time.monotonic()
    worker.run_capture(req)

    assert time.monotonic() - start < 1.0
    assert errors == ["no Scanlight"]


def test_stop_during_the_wait_cancels_without_a_shot(monkeypatch, tmp_path):
    worker = _worker(monkeypatch)
    shots: list[int] = []
    cancelled: list[bool] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    monkeypatch.setattr(module, "capture_single", lambda *a, **k: shots.append(1))
    worker.arm()
    threading.Timer(0.1, worker.cancel).start()

    start = time.monotonic()
    worker.run_capture(_request(tmp_path, 5.0))

    assert time.monotonic() - start < 2.0
    assert shots == []
    assert cancelled == [True]


def test_the_slider_feeds_the_setting_and_survives_a_reload(qapp):
    from negpy.desktop.view.sidebar.scanlight import ScanlightSidebar

    ctrl = MagicMock()
    ctrl.session.repo.get_global_setting.return_value = {}
    sidebar = ScanlightSidebar(ctrl)
    assert sidebar.capture_delay_slider.parentWidget() is sidebar.camera_body  # inside the Camera card

    sidebar.capture_delay_slider.setValue(3)
    sidebar._update_settings_from_ui()
    sidebar._save_settings()

    assert sidebar._settings.capture_delay_s == 3.0
    ctrl.session.repo.get_global_setting.return_value = ctrl.session.repo.save_global_setting.call_args[0][1]
    reloaded = ScanlightSidebar(ctrl)
    assert reloaded._settings.capture_delay_s == 3.0
    assert reloaded.capture_delay_slider.value() == 3


def test_the_countdown_shows_in_the_strip_while_a_scan_runs(qapp):
    from negpy.desktop.view.sidebar.scanlight import ScanlightSidebar

    ctrl = MagicMock()
    ctrl.session.repo.get_global_setting.return_value = {}
    sidebar = ScanlightSidebar(ctrl)
    sidebar.set_scanning(True)

    sidebar._on_countdown(3)
    role, shown = sidebar.status_strip.showing(), sidebar.status_strip._bar.format()
    sidebar._on_countdown(0)

    assert role == "progress" and "Capturing in 3 s" in shown
    assert "Capturing in" not in sidebar.status_strip._bar.format()
    sidebar.set_scanning(False)


def test_the_default_is_no_delay():
    assert ScanlightSettings().capture_delay_s == 0.0
    assert asdict(ScanlightSettings())["capture_delay_s"] == 0.0
