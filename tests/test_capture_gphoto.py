"""GphotoCamera unit tests — a fake libgphoto2, so CI needs no camera.

The fake reproduces the three behaviours that shaped the driver: a choice widget with no
choices whose value must never be read, asynchronous property writes, and a capture that
fails unless the event queue was drained.
"""

import json
import logging
import os

import pytest

from negpy.infrastructure.capture.base import Camera
from negpy.infrastructure.capture.gphoto import GphotoCamera, GphotoError, _pin_locale, _safe_value

# ---- fake libgphoto2 --------------------------------------------------------


class _Err(Exception):
    pass


class _Widget:
    def __init__(self, fake, name, value, choices, readonly=False, kind="radio"):
        self._fake, self._name = fake, name
        self.value, self.choices, self.readonly, self.kind = value, choices, readonly, kind

    def get_type(self):
        return {"radio": FakeGP.GP_WIDGET_RADIO, "text": FakeGP.GP_WIDGET_TEXT}[self.kind]

    def get_readonly(self):
        return self.readonly

    def count_choices(self):
        return len(self.choices)

    def get_choice(self, i):
        return self.choices[i]

    def get_value(self):
        if self.kind == "radio" and not self.choices:
            raise AssertionError("reading a choiceless widget segfaults the real library")
        return self.value

    def set_value(self, v):
        self._pending = v

    @property
    def pending(self):
        return getattr(self, "_pending", self.value)


class _File:
    def __init__(self, data):
        self._data = data

    def get_data_and_size(self):
        return self._data


class _Path:
    folder = "/"

    def __init__(self, name="capt0001.ARW"):
        self.name = name


class _Camera:
    def __init__(self, fake):
        self._fake = fake

    def _check(self):
        if self._fake.gone:
            raise _Err("[-7] I/O problem")

    def init(self):
        if self._fake.init_error:
            raise _Err(self._fake.init_error)
        self._fake.opened = True

    def exit(self):
        self._fake.opened = False

    def get_summary(self):
        return "Model: FAKE-1\nSerial: x"

    def get_single_config(self, name):
        self._check()
        if self._fake.readback_error and self._fake.writes and self._fake.writes[-1][0] == name:
            raise _Err("[-7] I/O problem during readback")
        if name not in self._fake.props:
            raise _Err(f"no property {name}")
        return self._fake.props[name]

    def set_single_config(self, name, widget):
        if self._fake.reject_writes:
            raise _Err("[-2] bad parameters")
        self._fake.writes.append((name, widget.pending))
        self._fake.settle(name, widget.pending)

    def capture(self, _kind):
        if self._fake.undrained:
            raise _Err("[-1] unspecified error")
        self._fake.undrained = True
        self._fake.captures += 1
        return _Path(self._fake.raw_name)

    def file_get(self, _folder, _name, _kind):
        return _File(b"RAWDATA" * 3)

    def capture_preview(self):
        self._check()
        self._fake.previews += 1
        return _File(b"\xff\xd8JPEG")

    def wait_for_event(self, _ms):
        if self._fake.undrained:
            self._fake.undrained = False
            return FakeGP.GP_EVENT_CAPTURE_COMPLETE, None
        return FakeGP.GP_EVENT_TIMEOUT, None


class _CameraList:
    def __init__(self, items):
        self._items = items

    def count(self):
        return len(self._items)

    def get_name(self, i):
        return self._items[i][0]

    def get_value(self, i):
        return self._items[i][1]


class FakeGP:
    GP_WIDGET_RADIO, GP_WIDGET_MENU, GP_WIDGET_TEXT = 5, 6, 2
    GP_CAPTURE_IMAGE, GP_FILE_TYPE_NORMAL = 0, 1
    GP_EVENT_TIMEOUT, GP_EVENT_CAPTURE_COMPLETE = 0, 3
    GPhoto2Error = _Err

    def __init__(
        self,
        *,
        settle_writes=True,
        init_error=None,
        cameras=(("FAKE-1", "usb:1"),),
        raw_name="capt0001.ARW",
        magnifier="sony",  # "sony" | "canon" | None
        aperture_name="f-number",
        capture_target="sdram",
    ):
        self.opened = False
        self.undrained = False
        self.gone = False  # set by unplug(): the body stops answering, as on a pulled cable
        self.reject_writes = False  # a value this body does not offer
        self.readback_error = False
        self.captures = self.previews = 0
        self.writes: list[tuple[str, str]] = []
        self.init_error = init_error
        self.raw_name = raw_name
        self._cameras = list(cameras)
        self._settle_writes = settle_writes
        self.props = {
            "iso": _Widget(self, "iso", "100", ["Auto", "100", "200"]),
            "shutterspeed": _Widget(self, "shutterspeed", "1/5", ["1/2", "1/5", "1/10"]),
            "capturetarget": _Widget(self, "capturetarget", capture_target, ["card", "sdram"]),
        }
        # Sony calls it f-number; Canon, Nikon, Fujifilm and Olympus call it aperture.
        # Either way, a lens with no electronic diaphragm offers no choices, and reading
        # its value in the real library is a SIGSEGV.
        self.props[aperture_name] = _Widget(self, aperture_name, None, [], readonly=True)
        if magnifier == "sony":  # ratio and position packed into one value
            self.props["focusmagnifier"] = _Widget(self, "focusmagnifier", "Off,320,240", ["Off", "1", "6.9", "13.7"])
        elif magnifier == "canon":  # ratio only; choices[0] == "1" *is* off
            self.props["eoszoom"] = _Widget(self, "eoszoom", "1", ["1", "5", "10"])
        self.Camera = lambda: _Camera(self)
        self.Camera.autodetect = lambda: _CameraList(self._cameras)

    def unplug(self):
        self.gone = True
        self._cameras = []

    def settle(self, name, value):
        if not self._settle_writes:
            return
        widget = self.props[name]
        # The magnifier echoes the position back alongside the ratio, and clamps it.
        widget.value = f"{value.split(',')[0]},589,438" if name == "focusmagnifier" else value


@pytest.fixture
def fake():
    return FakeGP()


@pytest.fixture
def cam(fake):
    camera = GphotoCamera(gp_module=fake, jpeg_path="/tmp/negpy_t.jpg", settings_path="/tmp/negpy_t.json")
    camera.open()
    yield camera
    camera.close()


# ---- the segfault guard -----------------------------------------------------


def test_safe_value_refuses_to_read_a_choiceless_widget(fake):
    # Reading it in the real library dereferences NULL and kills the process.
    assert _safe_value(fake, fake.props["f-number"]) is None
    assert _safe_value(fake, fake.props["iso"]) == "100"


def test_read_settings_omits_a_property_with_no_choices(cam):
    settings = cam.read_settings()
    assert "aperture" not in settings  # the UI greys the stepper out on a missing key
    assert set(settings) == {"iso", "shutter"}


def test_read_settings_shape_matches_the_ui_contract(cam):
    iso = cam.read_settings()["iso"]
    assert iso["cur"] == 1  # index of "100"
    assert iso["writable"] is True
    assert iso["options"] == [{"label": "Auto", "raw": 0}, {"label": "100", "raw": 1}, {"label": "200", "raw": 2}]


# ---- capture ----------------------------------------------------------------


def test_gphoto_camera_satisfies_the_camera_protocol(cam):
    assert isinstance(cam, Camera)


def test_capture_writes_the_raw_and_drains_events(cam, fake, tmp_path):
    out = tmp_path / "Roll1_Frame001_R.ARW"
    assert cam.capture(str(out)) == str(out)
    assert out.read_bytes() == b"RAWDATA" * 3
    assert not fake.undrained  # a leftover event makes the *next* capture fail


def test_three_captures_in_a_row_succeed(cam, tmp_path):
    # Without the drain the second capture raises "[-1] unspecified error".
    for channel in "RGB":
        cam.capture(str(tmp_path / f"f_{channel}.ARW"))


def test_capture_sets_the_shutter_first(cam, fake, tmp_path):
    cam.capture(str(tmp_path / "f.ARW"), shutter="1/10")
    assert ("shutterspeed", "1/10") in fake.writes


def test_capture_rejects_a_shutter_the_camera_will_not_accept(cam, fake, tmp_path):
    fake.reject_writes = True

    with pytest.raises(GphotoError, match="could not set shutter"):
        cam.capture(str(tmp_path / "f.ARW"), shutter="1/10")

    assert fake.captures == 0


def test_capture_rejects_a_shutter_that_never_settles(cam, fake, tmp_path, monkeypatch):
    fake._settle_writes = False
    clock = iter((0.0, 4.0))
    monkeypatch.setattr("negpy.infrastructure.capture.gphoto.time.monotonic", lambda: next(clock))

    with pytest.raises(GphotoError, match="did not settle"):
        cam.capture(str(tmp_path / "f.ARW"), shutter="1/10")

    assert fake.captures == 0


def test_capture_skips_an_unchanged_shutter(cam, fake, tmp_path):
    cam.capture(str(tmp_path / "f.ARW"), shutter="1/5")  # already 1/5
    assert fake.writes == []


def test_capture_failure_raises_gphoto_error(cam, fake, tmp_path):
    fake.undrained = True  # simulate a queue the driver failed to drain
    with pytest.raises(GphotoError):
        cam.capture(str(tmp_path / "f.ARW"))


def test_capture_rejects_processed_jpeg_instead_of_accepting_it_as_raw(tmp_path):
    fake = FakeGP(raw_name="IMG_0001.JPG")
    camera = GphotoCamera(gp_module=fake)
    camera.open()

    with pytest.raises(GphotoError, match="camera to RAW"):
        camera.capture(str(tmp_path / "Roll1_Frame001_R.raw"))

    assert not (tmp_path / "Roll1_Frame001_R.JPG").exists()
    assert not fake.undrained
    camera.close()


# ---- properties -------------------------------------------------------------


def test_set_iso_writes_the_chosen_label(cam, fake):
    cam.set_iso(2)
    assert fake.writes == [("iso", "200")]


def test_set_choice_ignores_an_out_of_range_index(cam, fake):
    cam.set_iso(99)
    assert fake.writes == []


def test_a_write_that_never_settles_is_reported_not_raised(fake, caplog):
    fake._settle_writes = False
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    camera._set_verified("iso", "200", settle_s=0.05)
    assert "did not settle" in caplog.text


def test_a_readback_error_is_reported_not_raised(cam, fake, caplog):
    fake.readback_error = True

    assert cam._set_verified("iso", "200", settle_s=0.05) is False
    assert "could not verify iso" in caplog.text


# ---- focus magnifier --------------------------------------------------------


def test_magnifier_ratios_are_read_from_the_body(cam):
    # ['Off', '1', '6.9', '13.7'] -> the first step only repositions, so 'on' is 6.9.
    assert cam._probe_magnifier() is not None
    assert cam._magnifier_ratios == ("6.9", "13.7")
    assert cam._magnifier_off == "Off"


# ---- other vendors ----------------------------------------------------------


def test_a_body_without_a_magnifier_disables_it_instead_of_raising():
    """Most bodies expose no magnifier at all, and an exception here would cross a Qt slot
    and abort the process."""
    fake = FakeGP(magnifier=None)
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    assert camera._probe_magnifier() is None
    camera.set_focus_magnifier_at(100, 100)  # must not raise
    camera.set_focus_magnifier(True)
    camera.set_focus_magnifier(False)
    assert fake.writes == []  # nothing was sent to the body
    camera.close()


def test_aperture_is_found_under_the_vendor_s_own_name():
    # Canon, Nikon, Fujifilm and Olympus expose "aperture" rather than "f-number".
    fake = FakeGP(aperture_name="aperture")
    fake.props["aperture"] = _Widget(fake, "aperture", "5.6", ["2.8", "5.6", "8"])
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    assert camera._property("aperture") == "aperture"
    assert camera.read_settings()["aperture"]["cur"] == 1
    camera.set_aperture(2)
    assert ("aperture", "8") in fake.writes
    camera.close()


def test_reopening_for_another_body_reprobes_property_names():
    fake = FakeGP(aperture_name="f-number")
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    assert "aperture" not in camera.read_settings()
    camera.close()

    del fake.props["f-number"]
    fake.props["aperture"] = _Widget(fake, "aperture", "5.6", ["2.8", "5.6", "8"])
    camera.open()
    camera.set_aperture(2)

    assert ("aperture", "8") in fake.writes
    camera.close()


def test_the_raw_suffix_comes_from_the_camera(tmp_path):
    # A Canon writes .CR3, a Nikon .NEF. Inventing ".ARW" would mislabel both.
    fake = FakeGP(raw_name="IMG_0001.CR3")
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    written = camera.capture(str(tmp_path / "Roll1_Frame001_R.raw"))
    assert written == str(tmp_path / "Roll1_Frame001_R.CR3")
    assert (tmp_path / "Roll1_Frame001_R.CR3").exists()
    camera.close()


def test_magnifier_position_is_kept_off_the_edges(cam, fake):
    # The body reads (0, 0) as "switch off", so the origin must never be sent.
    cam.set_focus_magnifier_at(0, 0)
    name, value = fake.writes[-1]
    assert name == "focusmagnifier"
    ratio, x, y = value.split(",")
    assert ratio == "6.9" and int(x) > 0 and int(y) > 0


def test_magnifier_position_is_clamped_into_the_grid(cam, fake):
    cam.set_focus_magnifier_at(9999, 9999)
    _name, value = fake.writes[-1]
    _ratio, x, y = value.split(",")
    assert int(x) <= 640 and int(y) <= 480


def test_magnifier_off_is_verified_on_the_ratio_only(cam, fake, caplog):
    # The body answers "Off,589,438" — comparing the whole string would never match.
    cam.set_focus_magnifier(False)
    assert ("focusmagnifier", "Off,320,240") in fake.writes
    assert "did not settle" not in caplog.text


def test_a_canon_style_magnifier_zooms_without_aiming(caplog):
    caplog.set_level(logging.INFO)
    """Canon splits ratio and position, and only the packed Sony form carries a coordinate
    space we know — so the click zooms, it just cannot aim."""
    fake = FakeGP(magnifier="canon")
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    assert camera._probe_magnifier() is not None
    assert camera._magnifier_ratios == ("5", "10")  # choices[0] == "1" is off, not a step

    camera.set_focus_magnifier_at(200, 150)
    assert fake.writes[-1] == ("eoszoom", "5")  # zoomed, no position sent
    assert "cannot be aimed" in caplog.text

    camera.set_focus_magnifier(False)
    assert fake.writes[-1] == ("eoszoom", "1")
    camera.close()


def test_reopening_for_another_body_reprobes_the_magnifier():
    fake = FakeGP(magnifier="sony")
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    camera.set_focus_magnifier(True)
    assert fake.writes[-1][0] == "focusmagnifier"
    camera.close()

    del fake.props["focusmagnifier"]
    fake.props["eoszoom"] = _Widget(fake, "eoszoom", "1", ["1", "5", "10"])
    fake.writes.clear()
    camera.open()
    camera.set_focus_magnifier(True)

    assert fake.writes[-1] == ("eoszoom", "5")
    camera.close()


def test_reopening_resets_the_magnifier_aim_to_centre():
    fake = FakeGP(magnifier="sony")
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    camera.set_focus_magnifier_at(100, 100)
    camera.close()

    fake.writes.clear()
    camera.open()
    camera.set_focus_magnifier(True)

    assert fake.writes[-1] == ("focusmagnifier", "6.9,320,240")
    camera.close()


def test_reopening_resets_the_unavailable_aim_warning(caplog):
    caplog.set_level(logging.INFO)
    fake = FakeGP(magnifier="canon")
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    camera.set_focus_magnifier_at(100, 100)
    assert "cannot be aimed" in caplog.text
    camera.close()

    caplog.clear()
    camera.open()
    camera.set_focus_magnifier_at(100, 100)

    assert "cannot be aimed" in caplog.text
    camera.close()


# ---- live view --------------------------------------------------------------


def test_a_fresh_camera_is_not_open(fake):
    assert not GphotoCamera(gp_module=fake).is_open()


def test_unplugged_camera_stops_reporting_itself_open(fake, tmp_path):
    """A handle left behind by a vanished body must not keep claiming to be open."""
    import time

    camera = GphotoCamera(gp_module=fake, jpeg_path=str(tmp_path / "lv.jpg"), settings_path=str(tmp_path / "lv.json"))
    camera.start()
    assert camera.is_open()

    fake.unplug()  # every camera call now raises, as it does when the cable is pulled
    for _ in range(300):
        if not camera.is_open():
            break
        time.sleep(0.01)
    else:
        raise AssertionError("the session still reports itself open")
    assert not camera.is_running()  # the preview thread gave up rather than spinning forever
    camera.close()


def test_reopening_replaces_a_dead_handle(fake):
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    camera._alive = False  # as the preview thread marks it when the body stops answering
    camera.open()  # must tear the corpse down and init a fresh session, not return early
    assert camera.is_open()
    camera.close()


def test_live_view_publishes_frames_and_settings(fake, tmp_path):
    jpeg, settings = tmp_path / "lv.jpg", tmp_path / "lv.json"
    camera = GphotoCamera(gp_module=fake, jpeg_path=str(jpeg), settings_path=str(settings))
    camera.start()
    try:
        for _ in range(200):
            if jpeg.exists() and settings.exists():
                break
            import time

            time.sleep(0.01)
    finally:
        camera.close()
    assert jpeg.read_bytes().startswith(b"\xff\xd8")
    assert set(json.loads(settings.read_text())) == {"iso", "shutter"}
    assert not camera.is_running()


def test_capture_target_is_moved_into_memory():
    """Canon and Nikon default to the card and refuse to shoot without one; the option is
    named differently everywhere, so match the word rather than a label."""
    fake = FakeGP(capture_target="card")
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    assert ("capturetarget", "sdram") in fake.writes
    camera.close()


def test_capture_target_left_alone_when_the_body_offers_no_memory_option(caplog):
    caplog.set_level(logging.INFO)
    fake = FakeGP(capture_target="card")
    fake.props["capturetarget"] = _Widget(fake, "capturetarget", "card", ["card"])
    camera = GphotoCamera(gp_module=fake)
    camera.open()
    assert fake.writes == []
    assert "no in-memory capture target" in caplog.text
    camera.close()


def test_setting_a_value_the_body_rejects_warns_instead_of_raising(cam, fake, caplog):
    """A shutter label from the built-in fallback ladder may not exist on this body. The
    scan should degrade to the current exposure, not die."""
    fake.reject_writes = True
    assert cam._set_verified("iso", "999") is False
    assert "could not set iso" in caplog.text


def test_pin_locale_overrides_a_preset_language(monkeypatch):
    """Linux desktops ship LANGUAGE preset (e.g. 'de_DE:de'); it must be overridden, not
    left in place, or gphoto's choice strings stay translated and the word-matched lookups
    miss."""
    monkeypatch.setenv("LANGUAGE", "de_DE:de")
    _pin_locale()
    assert os.environ["LANGUAGE"] == "C"
