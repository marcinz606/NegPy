"""CaptureWorker cancellation behavior at the camera/service boundary."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from negpy.desktop.workers.capture_worker import CalibrationRequest, CaptureRequest, CaptureWorker
from negpy.services.capture.calibration import Roi


class CancellingCamera:
    def __init__(self, worker: CaptureWorker) -> None:
        self.worker = worker

    def capture(self, out_path: str, shutter=None, iso=None, aperture=None) -> str:
        path = os.path.splitext(out_path)[0] + ".ARW"
        with open(path, "wb") as raw:
            raw.truncate(8 * 1024 * 1024)
        self.worker.cancel()  # Stop pressed while the camera is downloading the RAW.
        return path


class FakeLight:
    def set_color(self, **_channels) -> None:
        pass

    def off(self) -> None:
        pass


class FailingControlCamera:
    def __init__(self) -> None:
        self.closed = False

    def is_open(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True

    def set_focus_magnifier(self, _on: bool) -> None:
        raise RuntimeError("USB disconnected")

    def set_focus_magnifier_at(self, _x: int, _y: int) -> None:
        raise RuntimeError("USB disconnected")

    def set_iso(self, _raw: int) -> None:
        raise RuntimeError("USB disconnected")

    def set_shutter(self, _raw: int) -> None:
        pass

    def set_aperture(self, _raw: int) -> None:
        pass


@pytest.mark.parametrize(
    ("slot", "args"),
    [
        ("set_focus_magnifier", (True,)),
        ("set_focus_magnifier_pos", (320, 240)),
        ("set_camera_setting", ("iso", 1)),
    ],
)
def test_camera_control_slot_failure_is_recoverable(slot, args, caplog):
    worker = CaptureWorker()
    camera = FailingControlCamera()
    worker._camera = camera
    errors = []
    worker.error.connect(errors.append)

    getattr(worker, slot)(*args)

    assert camera.closed
    assert errors and "USB disconnected" in errors[-1]
    assert "Reconnect" in errors[-1]
    assert f"{slot} failed" in caplog.text


def test_normal_capture_cancel_before_promotion_preserves_retake(tmp_path, monkeypatch):
    existing = tmp_path / "Roll01_Frame007.ARW"
    existing.write_bytes(b"existing-good-raw")
    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: CancellingCamera(worker))
    finished = []
    worker.finished.connect(finished.append)

    worker.run_capture(
        CaptureRequest(
            roll_name="Roll01",
            frame_number=7,
            output_folder=str(tmp_path),
            levels=(200, 180, 255),
            rgb_mode=False,
            is_retake=True,
        )
    )

    assert existing.read_bytes() == b"existing-good-raw"
    assert finished == []


def test_scanlight_white_cancel_before_promotion_preserves_retake(tmp_path, monkeypatch):
    existing = tmp_path / "Slide01_Frame003.ARW"
    existing.write_bytes(b"existing-good-raw")
    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: CancellingCamera(worker))
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: FakeLight())
    finished = []
    worker.finished.connect(finished.append)

    worker.run_capture(
        CaptureRequest(
            roll_name="Slide01",
            frame_number=3,
            output_folder=str(tmp_path),
            levels=(200, 180, 255),
            settle_s=0,
            white_mode=True,
            is_retake=True,
        )
    )

    assert existing.read_bytes() == b"existing-good-raw"
    assert finished == []


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_calibration_uses_disposable_scratch_without_touching_roll(tmp_path, monkeypatch, outcome):
    import negpy.desktop.workers.capture_worker as capture_worker_module

    user_file = tmp_path / "_negpy_calibration.ARW"
    user_file.write_bytes(b"user-owned-raw")
    worker = CaptureWorker()

    class FakeCalibrationService:
        written_path: Path | None = None

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def calibrate(self, _roi, scratch_path, **_kwargs):
            written = Path(scratch_path).with_suffix(".ARW")
            written.write_bytes(b"temporary-calibration-raw")
            FakeCalibrationService.written_path = written
            if outcome == "cancel":
                worker.cancel()
                raise RuntimeError("calibration cancelled")
            if outcome == "error":
                raise RuntimeError("decode failed")
            return SimpleNamespace(levels=(1, 2, 3), shutters=("1/15",) * 3)

    monkeypatch.setattr(capture_worker_module, "CalibrationService", FakeCalibrationService)
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: FakeLight())
    monkeypatch.setattr(worker, "_acquire_camera", lambda: object())

    worker.run_calibration(CalibrationRequest(roi=Roi(0, 0, 1, 1), output_folder=str(tmp_path), settle_s=0))

    assert user_file.read_bytes() == b"user-owned-raw"
    assert FakeCalibrationService.written_path is not None
    assert not FakeCalibrationService.written_path.exists()
    assert not FakeCalibrationService.written_path.parent.exists()


class ClaimedCamera:
    """A body on the bus whose USB claim another program holds (gphoto -53)."""

    def __init__(self) -> None:
        self.open_calls = 0

    def is_open(self) -> bool:
        return False

    def open(self) -> None:
        self.open_calls += 1
        from negpy.infrastructure.capture.gphoto import CameraClaimedError

        raise CameraClaimedError(
            "could not open the camera: [-53] Could not claim the USB device. Close Preview, Photos and Image Capture, then retry."
        )

    def close(self) -> None:
        pass


def test_poll_reports_a_camera_claimed_by_another_app(monkeypatch):
    # macOS hands the body to Preview/Photos/Image Capture the moment one of them opens.
    # Enumeration still succeeds, so the camera dot showed a healthy "connected" while every
    # open failed with -53 — the poll must surface the claim so the UI can say what to do.
    import negpy.desktop.workers.capture_worker as capture_worker_module

    worker = CaptureWorker()
    worker._camera = ClaimedCamera()
    with pytest.raises(Exception):
        worker._acquire_camera()  # as any live-view/scan/calibration attempt would

    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [{"model": "USB PTP Class Camera"}])
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: (_ for _ in ()).throw(RuntimeError("no light")))
    seen: list[dict] = []
    worker.poll_status.connect(seen.append)
    worker.poll_connection("")
    assert seen and seen[0]["usb_ok"] is True
    assert seen[0]["usb_claimed_elsewhere"] is True

    # The body leaving the bus clears the verdict — the situation has changed.
    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [])
    worker.poll_connection("")
    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [{"model": "USB PTP Class Camera"}])
    worker.poll_connection("")
    assert seen[-1]["usb_claimed_elsewhere"] is False


def test_successful_open_clears_the_claimed_state(monkeypatch):
    class OpensFine:
        model = "ILCE-7CM2"

        def is_open(self) -> bool:
            return False

        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

    worker = CaptureWorker()
    worker._claimed_elsewhere = True  # left over from a failed attempt
    worker._camera = OpensFine()
    worker._acquire_camera()
    assert worker._claimed_elsewhere is False
