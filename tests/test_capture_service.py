"""CaptureService triplet-sequence unit tests (fake light + camera)."""

import os
import threading

import pytest

from negpy.infrastructure.capture.base import BulbRelease, CaptureSettings, Channel
from negpy.infrastructure.capture.protocol import encode_shutter_pulse
from negpy.infrastructure.capture.settings import ScanlightSettings
from negpy.services.capture.service import CaptureError, CaptureService, blend_plan


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


# ---- Blend: one Bulb exposure with R, G and B lit in turn ----------------------------------


class FakeReleaseLight(FakeLight):
    """A light with the 3.5mm shutter output."""

    def __init__(self):
        super().__init__()
        self.pulses: list[float] = []

    def shutter_pulse(self, seconds):
        self.pulses.append(seconds)
        self.colors.append(("pulse", seconds))


class FakeBulbCamera(FakeCamera):
    """Runs the exposure callback inside the Bulb shot, as the shutter would hold open."""

    def __init__(self, light, file_size=21 * 1024 * 1024):
        super().__init__(file_size)
        self.light = light
        self.bulbs: list[dict] = []

    def capture_bulb(self, out_path, expose, *, usb_release=True, iso=None, aperture=None, keep_bulb=False, on_closed=None):
        start = len(self.light.colors)
        expose()
        if on_closed is not None:
            on_closed()
        self.bulbs.append({"usb_release": usb_release, "iso": iso, "aperture": aperture, "lit": self.light.colors[start:]})
        self.kept = keep_bulb
        out_path = os.path.splitext(out_path)[0] + ".ARW"
        with open(out_path, "wb") as f:
            f.write(b"\0" * self.file_size)
        return out_path


def _blend(tmp_path, light, cam, waits, **kw):
    svc = CaptureService(light, cam, sleep=waits.append)
    return svc.capture_blend(_settings(tmp_path, iso="100", aperture="8", **kw.pop("settings", {})), **kw)


def test_blend_lights_each_channel_alone_in_one_bulb_exposure(tmp_path):
    light = FakeLight()
    cam, waits = FakeBulbCamera(light), []
    path = _blend(tmp_path, light, cam, waits, seconds=0.5)

    assert os.path.basename(path) == "Roll001_Frame001.ARW" and os.path.exists(path)
    (bulb,) = cam.bulbs
    assert bulb == {
        "usb_release": True,
        "iso": "100",
        "aperture": "8",
        "lit": [(0, 0, 0, 0), (200, 0, 0, 0), (0, 180, 0, 0), (0, 0, 255, 0), (0, 0, 0, 0)],
    }
    # Dark lead, then one on-time per channel at the Trichrome shutter. Over USB the release follows
    # the light-off at once: a slow shutter close cannot cut a channel.
    assert waits == pytest.approx([0.35, 0.5, 0.5, 0.5])


def test_blend_dims_the_leds_to_lengthen_a_short_channel(tmp_path):
    light = FakeLight()
    cam, waits = FakeBulbCamera(light), []
    _blend(tmp_path, light, cam, waits, seconds=0.05)

    levels = [max(c) for c in cam.bulbs[0]["lit"] if any(c)]
    times = waits[1:4]
    assert levels == [50, 45, 64]
    assert times == pytest.approx([0.2, 0.2, 0.05 * 255 / 64])
    # Each channel keeps its Trichrome light: level times seconds.
    assert [lv * t for lv, t in zip(levels, times)] == pytest.approx([200 * 0.05, 180 * 0.05, 255 * 0.05])


def test_blend_dims_no_led_under_the_calibration_floor(tmp_path):
    light = FakeLight()
    cam, waits = FakeBulbCamera(light), []
    _blend(tmp_path, light, cam, waits, seconds=0.02)

    # The dimmest LED stops at 40, so the channels stay shorter than 0.2 s.
    assert [max(c) for c in cam.bulbs[0]["lit"] if any(c)] == [44, 40, 57]


def test_blend_by_cable_pulses_the_shutter_and_leaves_the_camera_to_wait(tmp_path):
    light = FakeReleaseLight()
    cam, waits = FakeBulbCamera(light), []
    _blend(tmp_path, light, cam, waits, seconds=0.4, release=BulbRelease.CABLE)

    assert cam.bulbs[0]["usb_release"] is False
    assert light.pulses == pytest.approx([0.35 + 3 * 0.4 + 0.2])
    assert light.colors.index(("pulse", light.pulses[0])) < light.colors.index((200, 0, 0, 0))


def test_blend_by_cable_brightens_the_leds_to_fit_the_pulse(tmp_path):
    light = FakeReleaseLight()
    cam, waits = FakeBulbCamera(light), []
    _blend(tmp_path, light, cam, waits, seconds=1.0, settings={"levels": (100, 90, 120)}, release=BulbRelease.CABLE)

    assert sum(waits[1:4]) <= 2.55 - 0.35 - 0.2 + 1e-9
    assert light.pulses[0] <= 2.55 + 1e-9
    assert encode_shutter_pulse(light.pulses[0])[-1] == 255  # the light's own encoder takes it, at its limit


def test_blend_by_cable_refuses_an_exposure_longer_than_the_pulse(tmp_path):
    light = FakeReleaseLight()
    cam = FakeBulbCamera(light)
    with pytest.raises(CaptureError, match="release cable"):
        _blend(tmp_path, light, cam, [], seconds=2.0, release=BulbRelease.CABLE)
    assert cam.bulbs == [] and light.pulses == []


def test_blend_by_cable_needs_a_light_with_a_shutter_output(tmp_path):
    light = FakeLight()
    with pytest.raises(CaptureError, match="shutter release"):
        _blend(tmp_path, light, FakeBulbCamera(light), [], seconds=0.5, release=BulbRelease.CABLE)


def test_blend_needs_a_bulb_camera(tmp_path):
    with pytest.raises(CaptureError, match="Bulb"):
        _blend(tmp_path, FakeLight(), FakeCamera(), [], seconds=0.5)


def test_blend_reports_each_channel_and_leaves_the_end_light_lit(tmp_path):
    light = FakeLight()
    seen = []
    _blend(tmp_path, light, FakeBulbCamera(light), [], seconds=0.5, on_channel=seen.append, end_light=(10, 10, 10, 0))
    assert seen == ["R", "G", "B"]
    assert light.colors[-1] == (10, 10, 10, 0)


def test_blend_stop_between_channels_ends_the_light_and_keeps_no_file(tmp_path):
    light = FakeLight()
    cancel = threading.Event()

    def _stop_after_red(letter):
        if letter == "R":
            cancel.set()

    with pytest.raises(CaptureError, match="cancelled"):
        _blend(tmp_path, light, FakeBulbCamera(light), [], seconds=0.5, cancel=cancel, on_channel=_stop_after_red)
    assert (0, 180, 0, 0) not in light.colors
    assert light.colors[-1] == (0, 0, 0, 0)
    assert not [n for n in os.listdir(tmp_path) if n.endswith(".ARW")]


def test_blend_plan_skips_an_unlit_channel():
    plan = blend_plan((200, 0, 255), 0.5)
    assert [step.channel for step in plan] == [Channel.RED, Channel.BLUE]


def test_blend_passes_keep_bulb_to_the_camera(tmp_path):
    light = FakeLight()
    cam = FakeBulbCamera(light)
    _blend(tmp_path, light, cam, [], seconds=0.5, keep_bulb=True)
    assert cam.kept is True


def test_blend_calls_on_closed_after_the_light_is_off(tmp_path):
    light = FakeLight()
    seen = []
    _blend(tmp_path, light, FakeBulbCamera(light), [], seconds=0.5, on_closed=lambda: seen.append(light.colors[-1]))
    assert seen == [(0, 0, 0, 0)]


def test_blend_by_cable_keeps_the_dark_tail_inside_the_pulse(tmp_path):
    light = FakeReleaseLight()
    waits = []
    _blend(tmp_path, light, FakeBulbCamera(light), waits, seconds=0.4, release=BulbRelease.CABLE)
    assert waits[-1] == pytest.approx(0.2)


class FakeTimedLight(FakeLight):
    """A light whose set_color waits for a reply, so a timed exposure must switch through set_color_now."""

    def __init__(self):
        super().__init__()
        self.now: list[tuple[int, int, int]] = []

    def set_color_now(self, r=0, g=0, b=0):
        self.now.append((r, g, b))


def test_blend_switches_a_waiting_light_through_set_color_now(tmp_path):
    light = FakeTimedLight()
    _blend(tmp_path, light, FakeBulbCamera(light), [], seconds=0.5)
    assert light.now == [(0, 0, 0), (200, 0, 0), (0, 180, 0), (0, 0, 255), (0, 0, 0)]
    assert light.colors and all(c == (0, 0, 0, 0) for c in light.colors)  # off() only for the end-of-frame release
