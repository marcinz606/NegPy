"""Cancellation is a terminal outcome for camera capture work."""

from types import SimpleNamespace

from negpy.desktop.workers import capture_worker as capture_worker_module
from negpy.desktop.workers.capture_worker import CalibrationRequest, CaptureRequest, CaptureWorker
from negpy.services.capture.service import CaptureError
from negpy.services.capture.calibration import Roi


def test_capture_worker_emits_cancelled_once_when_capture_aborts(monkeypatch, tmp_path):
    worker = CaptureWorker()
    cancelled: list[bool] = []
    finished: list[list[str]] = []
    errors: list[str] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    class HeldCamera:
        closed = False

        def is_open(self):
            return not self.closed

        def close(self):
            self.closed = True

    camera = HeldCamera()
    worker._camera = camera
    monkeypatch.setattr(worker, "_acquire_camera", lambda: camera)

    def abort_capture(*_args, **_kwargs):
        worker.cancel()
        raise CaptureError("capture cancelled")

    monkeypatch.setattr(capture_worker_module, "capture_single", abort_capture)

    worker.run_capture(
        CaptureRequest(
            roll_name="Roll001",
            frame_number=1,
            output_folder=str(tmp_path),
            levels=(255, 255, 255),
            rgb_mode=False,
        )
    )

    assert cancelled == [True]
    assert finished == []
    assert errors == []
    assert not camera.closed  # a deliberate cancel keeps the healthy live-view session
    assert worker._camera is camera


def test_capture_worker_finishes_a_committed_capture_if_cancel_arrives_afterward(monkeypatch, tmp_path):
    worker = CaptureWorker()
    cancelled: list[bool] = []
    finished: list[list[str]] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.finished.connect(finished.append)
    monkeypatch.setattr(worker, "_acquire_camera", lambda: object())
    captured_path = str(tmp_path / "Roll001_Frame001.ARW")

    def complete_then_cancel(*_args, **_kwargs):
        worker.cancel()
        return captured_path

    monkeypatch.setattr(capture_worker_module, "capture_single", complete_then_cancel)

    worker.run_capture(
        CaptureRequest(
            roll_name="Roll001",
            frame_number=1,
            output_folder=str(tmp_path),
            levels=(255, 255, 255),
            rgb_mode=False,
        )
    )

    assert finished == [[captured_path]]
    assert cancelled == []


def test_capture_worker_finishes_a_committed_triplet_if_cancel_arrives_afterward(monkeypatch, tmp_path):
    worker = CaptureWorker()
    cancelled: list[bool] = []
    finished: list[list[str]] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.finished.connect(finished.append)
    monkeypatch.setattr(worker, "_acquire_camera", lambda: object())
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: object())
    paths = [str(tmp_path / f"Roll001_Frame001_{channel}.ARW") for channel in "RGB"]

    class CompletingService:
        def __init__(self, *_args):
            pass

        def capture_triplet(self, *_args, **_kwargs):
            worker.cancel()
            return SimpleNamespace(paths=paths)

    monkeypatch.setattr(capture_worker_module, "CaptureService", CompletingService)

    worker.run_capture(
        CaptureRequest(
            roll_name="Roll001",
            frame_number=1,
            output_folder=str(tmp_path),
            levels=(255, 255, 255),
        )
    )

    assert finished == [paths]
    assert cancelled == []


def test_capture_worker_finishes_a_committed_white_capture_if_cancel_arrives_afterward(monkeypatch, tmp_path):
    worker = CaptureWorker()
    cancelled: list[bool] = []
    finished: list[list[str]] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.finished.connect(finished.append)
    monkeypatch.setattr(worker, "_acquire_camera", lambda: object())
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: object())
    captured_path = str(tmp_path / "Roll001_Frame001.ARW")

    class CompletingService:
        def __init__(self, *_args):
            pass

        def capture_white(self, **_kwargs):
            worker.cancel()
            return captured_path

    monkeypatch.setattr(capture_worker_module, "CaptureService", CompletingService)

    worker.run_capture(
        CaptureRequest(
            roll_name="Roll001",
            frame_number=1,
            output_folder=str(tmp_path),
            levels=(255, 255, 255),
            white_mode=True,
        )
    )

    assert finished == [[captured_path]]
    assert cancelled == []


def test_capture_worker_emits_cancelled_once_when_calibration_aborts(monkeypatch, tmp_path):
    worker = CaptureWorker()
    cancelled: list[bool] = []
    finished: list[object] = []
    errors: list[str] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.calibration_finished.connect(finished.append)
    worker.error.connect(errors.append)
    monkeypatch.setattr(worker, "_acquire_camera", lambda: object())
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: object())

    class AbortingCalibration:
        def __init__(self, *_args, **_kwargs):
            pass

        def calibrate(self, *_args, **_kwargs):
            worker.cancel()
            raise RuntimeError("calibration cancelled")

    monkeypatch.setattr(capture_worker_module, "CalibrationService", AbortingCalibration)

    worker.run_calibration(CalibrationRequest(roi=Roi(0, 0, 1, 1), output_folder=str(tmp_path)))

    assert cancelled == [True]
    assert finished == []
    assert errors == []


def test_capture_worker_finishes_calibration_if_cancel_arrives_after_result(monkeypatch, tmp_path):
    worker = CaptureWorker()
    cancelled: list[bool] = []
    finished: list[object] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.calibration_finished.connect(finished.append)
    monkeypatch.setattr(worker, "_acquire_camera", lambda: object())
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: object())
    result = object()

    class CompletingCalibration:
        def __init__(self, *_args, **_kwargs):
            pass

        def calibrate(self, *_args, **_kwargs):
            worker.cancel()
            return result

    monkeypatch.setattr(capture_worker_module, "CalibrationService", CompletingCalibration)

    worker.run_calibration(CalibrationRequest(roi=Roi(0, 0, 1, 1), output_folder=str(tmp_path)))

    assert finished == [result]
    assert cancelled == []
