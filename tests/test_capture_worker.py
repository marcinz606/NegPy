"""CaptureWorker cancellation behavior at the camera/service boundary."""

import os

from negpy.desktop.workers.capture_worker import CaptureRequest, CaptureWorker


class CancellingCamera:
    def __init__(self, worker: CaptureWorker) -> None:
        self.worker = worker

    def capture(self, out_path: str, shutter=None) -> str:
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
