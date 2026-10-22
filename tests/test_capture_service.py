"""CaptureService triplet-sequence unit tests (fake light + camera)."""

import os
import threading

import pytest

from negpy.infrastructure.capture.base import CaptureSettings, Channel
from negpy.infrastructure.capture.settings import ScanlightSettings
from negpy.services.capture.service import CaptureError, CaptureService


class FakeLight:
    """Records every set_color call and whether it was turned off."""

    def __init__(self):
        self.colors: list[tuple[int, int, int, int]] = []
        self.ir_levels: list[int] = []
        self.off_called = False
        self.closed = False

    def set_color(self, r=0, g=0, b=0, w=0, ir=0, save=False):
        self.colors.append((r, g, b, w))
        self.ir_levels.append(ir)

    def off(self):
        self.off_called = True
        self.colors.append((0, 0, 0, 0))
        self.ir_levels.append(0)

    def close(self):
        self.closed = True


class FakeCamera:
    """Writes a dummy RAW of `file_size` bytes on each capture."""

    def __init__(self, file_size=21 * 1024 * 1024):
        self.file_size = file_size
        self.captured: list[str] = []
        self.shutters: list = []
        self.isos: list = []
        self.apertures: list = []

    def capture(self, out_path, shutter=None, iso=None, aperture=None):
        out_path = os.path.splitext(out_path)[0] + ".ARW"  # the camera picks the suffix
        self.captured.append(out_path)
        self.shutters.append(shutter)
        self.isos.append(iso)
        self.apertures.append(aperture)
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


def test_capture_defaults_to_zero_inter_exposure_delay(tmp_path):
    settings = _settings(tmp_path)
    assert settings.inter_exposure_delay_s == 0.0
    assert ScanlightSettings().inter_exposure_delay_ms == 0


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


def test_triplet_forces_the_preset_iso_and_aperture(tmp_path):
    cam = FakeCamera()
    svc = CaptureService(FakeLight(), cam, sleep=lambda _s: None)
    svc.capture_triplet(_settings(tmp_path, iso="100", aperture="f/8"))
    # The frame's first shot asserts the preset's exposure, so a drifted body can't falsify
    # the scan; the body cannot drift between two shots of one frame.
    assert cam.isos == ["100", None, None]
    assert cam.apertures == ["f/8", None, None]


def test_triplet_leaves_exposure_free_when_the_preset_bakes_none(tmp_path):
    cam = FakeCamera()
    svc = CaptureService(FakeLight(), cam, sleep=lambda _s: None)
    svc.capture_triplet(_settings(tmp_path))  # no iso/aperture → camera keeps its current values
    assert cam.isos == [None, None, None]
    assert cam.apertures == [None, None, None]


def test_capture_triplet_reports_each_channel(tmp_path):
    svc = CaptureService(FakeLight(), FakeCamera(), sleep=lambda _s: None)
    channels = []
    svc.capture_triplet(_settings(tmp_path), on_channel=channels.append)
    assert channels == ["R", "G", "B"]


def test_capture_triplet_waits_between_channels(tmp_path):
    light, cam = FakeLight(), FakeCamera()
    sleeps = []
    svc = CaptureService(light, cam, sleep=lambda s: sleeps.append(s))

    svc.capture_triplet(_settings(tmp_path, settle_s=0.2, inter_exposure_delay_s=2.5))

    assert sleeps == [0.2, 2.5, 0.2, 2.5, 0.2]


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
        def capture(self, out_path, shutter=None, iso=None, aperture=None):
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
    assert cam.shutters == ["1/100", None, "1/4"]  # written when it changes between channels


def test_single_rgb_capture_lights_all_three_leds_for_one_file(tmp_path):
    light, cam = FakeLight(), FakeCamera()
    svc = CaptureService(light, cam, sleep=lambda _s: None)
    path = svc.capture_rgb_single(_settings(tmp_path, shutters=("1/5", "1/5", "1/5"), iso="100", aperture="f/8"))
    assert os.path.basename(path) == "Roll001_Frame001.ARW"  # no _R/_G/_B suffix
    assert os.listdir(tmp_path) == ["Roll001_Frame001.ARW"]
    assert light.colors == [(200, 180, 255, 0), (0, 0, 0, 0)]  # R, G and B together, then off
    assert (cam.shutters, cam.isos, cam.apertures) == (["1/5"], ["100"], ["f/8"])  # the preset's exposure is forced


def test_single_rgb_capture_turns_light_off_on_error(tmp_path):
    light = FakeLight()
    svc = CaptureService(light, FakeCamera(file_size=1024), sleep=lambda _s: None)
    with pytest.raises(CaptureError):
        svc.capture_rgb_single(_settings(tmp_path))
    assert light.off_called
    assert os.listdir(tmp_path) == []


def test_capture_channel_lights_one_led_and_names_the_file(tmp_path):
    light, camera = FakeLight(), FakeCamera()
    service = CaptureService(light, camera, sleep=lambda _s: None)
    (tmp_path / "Roll001_Frame002_IR.ARW").write_bytes(b"old")

    path = service.capture_channel(
        roll_name="Roll001", frame_number=2, output_folder=str(tmp_path), channel=Channel.IR, level=200, shutter="1"
    )

    assert os.path.basename(path) == "Roll001_Frame002_IR.ARW"
    assert os.listdir(tmp_path) == ["Roll001_Frame002_IR.ARW"]  # the old exposure is replaced, nothing kept beside it
    assert light.ir_levels == [200, 0] and light.colors == [(0, 0, 0, 0), (0, 0, 0, 0)]  # infrared alone, then off
    assert camera.shutters == ["1"]


def test_a_retaken_triplet_replaces_its_files_and_keeps_nothing_beside_them(tmp_path):
    for letter in "RGB":
        (tmp_path / f"Roll001_Frame001_{letter}.ARW").write_bytes(b"old-" + letter.encode())
    service = CaptureService(FakeLight(), FakeCamera(), sleep=lambda _s: None)

    service.capture_triplet(_settings(tmp_path))

    assert sorted(os.listdir(tmp_path)) == [f"Roll001_Frame001_{letter}.ARW" for letter in "BGR"]
    for letter in "RGB":
        assert (tmp_path / f"Roll001_Frame001_{letter}.ARW").stat().st_size == 21 * 1024 * 1024


def test_a_triplet_writes_iso_and_aperture_once_and_the_shutter_when_it_changes(tmp_path):
    camera = FakeCamera()
    service = CaptureService(FakeLight(), camera, sleep=lambda _s: None)

    service.capture_triplet(_settings(tmp_path, shutters=("1/4", "1/4", "1/8"), iso="100", aperture="f/8"))

    assert camera.isos == ["100", None, None] and camera.apertures == ["f/8", None, None]
    assert camera.shutters == ["1/4", None, "1/8"]  # the body cannot drift inside one frame


def test_capture_triplet_leaves_the_end_light_lit_with_no_off_step(tmp_path):
    light = FakeLight()
    svc = CaptureService(light, FakeCamera(), sleep=lambda _s: None)

    svc.capture_triplet(_settings(tmp_path), end_light=(25, 12, 11, 0))

    assert not light.off_called
    assert light.colors == [(200, 0, 0, 0), (0, 180, 0, 0), (0, 0, 255, 0), (25, 12, 11, 0)]


def test_capture_triplet_turns_the_light_off_on_error_despite_an_end_light(tmp_path):
    light = FakeLight()
    svc = CaptureService(light, FakeCamera(file_size=100), sleep=lambda _s: None)

    with pytest.raises(CaptureError):
        svc.capture_triplet(_settings(tmp_path), end_light=(25, 12, 11, 0))
    assert light.colors[-1] == (0, 0, 0, 0)


def test_capture_triplet_turns_the_light_off_on_cancel_despite_an_end_light(tmp_path):
    light = FakeLight()
    cancel = threading.Event()
    cancel.set()
    svc = CaptureService(light, FakeCamera(), sleep=lambda _s: None)

    with pytest.raises(CaptureError):
        svc.capture_triplet(_settings(tmp_path), cancel=cancel, end_light=(25, 12, 11, 0))
    assert light.colors[-1] == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "shoot",
    [
        lambda svc, tmp, end: svc.capture_rgb_single(_settings(tmp), end_light=end),
        lambda svc, tmp, end: svc.capture_white(roll_name="Roll001", frame_number=1, output_folder=str(tmp), w_level=200, end_light=end),
        lambda svc, tmp, end: svc.capture_channel(
            roll_name="Roll001", frame_number=1, output_folder=str(tmp), channel=Channel.IR, level=255, end_light=end
        ),
    ],
    ids=["single", "white", "channel"],
)
def test_single_exposures_leave_the_end_light_lit(tmp_path, shoot):
    light = FakeLight()
    svc = CaptureService(light, FakeCamera(), sleep=lambda _s: None)

    shoot(svc, tmp_path, (0, 0, 0, 120))

    assert not light.off_called
    assert light.colors[-1] == (0, 0, 0, 120)
