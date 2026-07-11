"""White-light single capture and RAW size-guard checks."""

import os
import threading

import pytest

from negpy.services.capture.service import (
    CaptureError,
    CaptureService,
    capture_single,
    verify_raw_size,
)


class FakeLight:
    def __init__(self):
        self.colors = []
        self.off_called = False

    def set_color(self, r=0, g=0, b=0, w=0, save=False):
        self.colors.append((r, g, b, w))

    def off(self):
        self.off_called = True

    def close(self):
        pass


class FakeCamera:
    def __init__(self, size=21 * 1024 * 1024):
        self.size = size
        self.captured = []

    def capture(self, out_path, shutter=None):
        out_path = os.path.splitext(out_path)[0] + ".ARW"  # the camera picks the suffix
        self.captured.append((out_path, shutter))
        with open(out_path, "wb") as f:
            f.write(b"\0" * self.size)
        return out_path

    def close(self):
        pass


def test_capture_white_single_file(tmp_path):
    light, cam = FakeLight(), FakeCamera()
    svc = CaptureService(light, cam, sleep=lambda _s: None)

    path = svc.capture_white(roll_name="Slide01", frame_number=3, output_folder=str(tmp_path), w_level=200, shutter="1/30")
    assert os.path.basename(path) == "Slide01_Frame003.ARW"  # no _R/_G/_B suffix
    assert os.path.exists(path)
    assert cam.captured[0][1] == "1/30"
    # White channel only, then light off.
    assert any(c[3] == 200 and (c[0], c[1], c[2]) == (0, 0, 0) for c in light.colors)
    assert light.off_called


def test_failed_white_retake_preserves_existing_file(tmp_path):
    existing = tmp_path / "Slide01_Frame003.ARW"
    existing.write_bytes(b"existing-good-raw")
    light = FakeLight()
    svc = CaptureService(light, FakeCamera(size=0), sleep=lambda _s: None)

    with pytest.raises(CaptureError):
        svc.capture_white(
            roll_name="Slide01",
            frame_number=3,
            output_folder=str(tmp_path),
            w_level=200,
            min_raw_bytes=1,
            max_raw_bytes=100,
        )

    assert existing.read_bytes() == b"existing-good-raw"
    assert light.off_called


def test_cancelled_white_retake_preserves_existing_file(tmp_path):
    existing = tmp_path / "Slide01_Frame003.ARW"
    existing.write_bytes(b"existing-good-raw")
    cancel = threading.Event()

    class CancellingCamera(FakeCamera):
        def capture(self, out_path, shutter=None):
            path = super().capture(out_path, shutter=shutter)
            cancel.set()  # Stop pressed while the camera is downloading the RAW.
            return path

    light = FakeLight()
    svc = CaptureService(light, CancellingCamera(size=10), sleep=lambda _s: None)

    with pytest.raises(CaptureError, match="capture cancelled"):
        svc.capture_white(
            roll_name="Slide01",
            frame_number=3,
            output_folder=str(tmp_path),
            w_level=200,
            min_raw_bytes=1,
            max_raw_bytes=100,
            cancel=cancel,
        )

    assert existing.read_bytes() == b"existing-good-raw"
    assert light.off_called


def test_capture_single_no_light(tmp_path):
    # Normal white-light scanning: camera-only, one file, no Scanlight involved.
    cam = FakeCamera()
    path = capture_single(cam, roll_name="Roll01", frame_number=7, output_folder=str(tmp_path), shutter="1/60")
    assert os.path.basename(path) == "Roll01_Frame007.ARW"  # single file, no _R/_G/_B suffix
    assert os.path.exists(path)
    assert cam.captured[0][1] == "1/60"


def test_failed_single_retake_preserves_existing_file(tmp_path):
    existing = tmp_path / "Roll01_Frame007.ARW"
    existing.write_bytes(b"existing-good-raw")

    with pytest.raises(CaptureError):
        capture_single(
            FakeCamera(size=0),
            roll_name="Roll01",
            frame_number=7,
            output_folder=str(tmp_path),
            min_raw_bytes=1,
            max_raw_bytes=100,
        )

    assert existing.read_bytes() == b"existing-good-raw"


def test_cancelled_single_retake_preserves_existing_file(tmp_path):
    existing = tmp_path / "Roll01_Frame007.ARW"
    existing.write_bytes(b"existing-good-raw")
    cancel = threading.Event()

    class CancellingCamera(FakeCamera):
        def capture(self, out_path, shutter=None):
            path = super().capture(out_path, shutter=shutter)
            cancel.set()  # Stop pressed while the camera is downloading the RAW.
            return path

    with pytest.raises(CaptureError, match="capture cancelled"):
        capture_single(
            CancellingCamera(size=10),
            roll_name="Roll01",
            frame_number=7,
            output_folder=str(tmp_path),
            min_raw_bytes=1,
            max_raw_bytes=100,
            cancel=cancel,
        )

    assert existing.read_bytes() == b"existing-good-raw"


def test_verify_raw_size_rejects_small(tmp_path):
    p = tmp_path / "x.ARW"
    p.write_bytes(b"\0" * 100)
    with pytest.raises(CaptureError):
        verify_raw_size(str(p), 20 * 1024 * 1024, 200 * 1024 * 1024)


def test_verify_raw_size_missing_file():
    with pytest.raises(CaptureError):
        verify_raw_size("/no/such/file.ARW", 0, 1)
