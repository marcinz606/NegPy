"""CaptureService triplet-sequence unit tests (fake light + camera)."""

import os
import threading

import pytest

from negpy.infrastructure.capture.base import CaptureSettings
from negpy.services.capture.service import CaptureError, CaptureService


class FakeLight:
    """Records every set_color call and whether it was turned off."""

    def __init__(self):
        self.colors: list[tuple[int, int, int, int]] = []
        self.off_called = False
        self.closed = False

    def set_color(self, r=0, g=0, b=0, w=0, save=False):
        self.colors.append((r, g, b, w))

    def off(self):
        self.off_called = True
        self.colors.append((0, 0, 0, 0))

    def close(self):
        self.closed = True


class FakeCamera:
    """Writes a dummy RAW of `file_size` bytes on each capture."""

    def __init__(self, file_size=21 * 1024 * 1024):
        self.file_size = file_size
        self.captured: list[str] = []
        self.shutters: list = []

    def capture(self, out_path, shutter=None):
        out_path = os.path.splitext(out_path)[0] + ".ARW"  # the camera picks the suffix
        self.captured.append(out_path)
        self.shutters.append(shutter)
        with open(out_path, "wb") as f:
            f.write(b"\0" * self.file_size)
        return out_path

    def close(self):
        pass


def _settings(tmp_path, **kw):
    base = dict(
        roll_name="Roll001",
        frame_number=1,
        output_folder=str(tmp_path),
        levels=(200, 180, 255),
    )
    base.update(kw)
    return CaptureSettings(**base)


def test_capture_triplet_sequence_and_filenames(tmp_path):
    light, cam = FakeLight(), FakeCamera()
    svc = CaptureService(light, cam, sleep=lambda _s: None)
    seen = []

    result = svc.capture_triplet(_settings(tmp_path), progress=seen.append)

    # One file per channel, correctly named, red is primary.
    assert [os.path.basename(p) for p in result.paths] == [
        "Roll001_Frame001_R.ARW",
        "Roll001_Frame001_G.ARW",
        "Roll001_Frame001_B.ARW",
    ]
    assert result.red_path.endswith("_R.ARW")
    for p in result.paths:
        assert os.path.exists(p)

    # Each channel was lit alone, in order, at its slider level.
    lit = [c for c in light.colors if c != (0, 0, 0, 0)]
    assert lit == [(200, 0, 0, 0), (0, 180, 0, 0), (0, 0, 255, 0)]

    # Light is off at the end; progress reaches 1.0.
    assert light.off_called
    assert light.colors[-1] == (0, 0, 0, 0)
    assert seen[-1] == pytest.approx(1.0)


def test_capture_triplet_reports_each_channel(tmp_path):
    svc = CaptureService(FakeLight(), FakeCamera(), sleep=lambda _s: None)
    channels = []
    svc.capture_triplet(_settings(tmp_path), on_channel=channels.append)
    assert channels == ["R", "G", "B"]


def test_capture_triplet_turns_light_off_on_error(tmp_path):
    light = FakeLight()
    cam = FakeCamera(file_size=100)  # implausibly small → size check fails
    svc = CaptureService(light, cam, sleep=lambda _s: None)

    with pytest.raises(CaptureError):
        svc.capture_triplet(_settings(tmp_path))
    assert light.off_called  # finally-block always kills the light


def test_failed_retake_preserves_complete_existing_triplet(tmp_path):
    """A bad late channel must not leave a new R/G paired with the previous B."""
    existing = {}
    for channel in "RGB":
        path = tmp_path / f"Roll001_Frame001_{channel}.ARW"
        path.write_bytes(f"old-{channel}".encode())
        existing[channel] = path

    class FailingBlueCamera(FakeCamera):
        def capture(self, out_path, shutter=None):
            out_path = os.path.splitext(out_path)[0] + ".ARW"
            channel = os.path.splitext(out_path)[0].rsplit("_", 1)[-1]
            with open(out_path, "wb") as f:
                f.write(b"new-channel" if channel != "B" else b"")
            return out_path

    svc = CaptureService(FakeLight(), FailingBlueCamera(), sleep=lambda _s: None)

    with pytest.raises(CaptureError):
        svc.capture_triplet(_settings(tmp_path, min_raw_bytes=1, max_raw_bytes=100))

    assert {channel: path.read_bytes() for channel, path in existing.items()} == {
        "R": b"old-R",
        "G": b"old-G",
        "B": b"old-B",
    }


def test_capture_triplet_cancel_before_first_channel(tmp_path):
    light, cam = FakeLight(), FakeCamera()
    svc = CaptureService(light, cam, sleep=lambda _s: None)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(CaptureError):
        svc.capture_triplet(_settings(tmp_path), cancel=cancel)
    assert cam.captured == []  # nothing was shot
    assert light.off_called


def test_per_channel_shutter_passed_through(tmp_path):
    light, cam = FakeLight(), FakeCamera()
    svc = CaptureService(light, cam, sleep=lambda _s: None)

    svc.capture_triplet(_settings(tmp_path, shutters=("1/100", "1/100", "1/4")))
    assert cam.shutters == ["1/100", "1/100", "1/4"]
