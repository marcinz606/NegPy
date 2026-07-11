"""ETTR auto-calibration unit tests (fake light + camera + linear demosaic)."""

import os
import numpy as np
import pytest

from negpy.services.capture.calibration import (
    MAX_CLIP_FRACTION,
    CalibrationService,
    Roi,
    clip_fraction,
    faster_shutter,
    meter_base,
    meter_black,
    shutter_at_least,
    shutter_seconds,
    slower_shutter,
    target_for_black_level,
)

BLACK = 512.0
# Per-channel linear response (counts per LED-level per second). Blue is the dimmest (orange
# mask), so it needs the most exposure — but only ~1 stop below red, within the LED's ~2.7-stop
# range, so one shared shutter fits all three (matches the real Portra/Vision3 measurements).
K = {0: 2250.0, 1: 1700.0, 2: 1200.0}


def test_shutter_seconds():
    assert shutter_seconds("1/100") == 0.01
    assert shutter_seconds("0.4") == 0.4
    assert shutter_seconds("1") == 1.0


def test_target_for_black_level():
    assert target_for_black_level(512.0, 0.9) == round(0.9 * (65535 - 512))


def test_roi_pixels_clamps_and_orders():
    assert Roi(0.0, 0.0, 1.0, 1.0).pixels(100, 80) == (0, 0, 100, 80)
    assert Roi(0.25, 0.5, 0.5, 0.25).pixels(100, 80) == (25, 40, 75, 60)
    # Degenerate / out-of-range still yields a non-empty crop.
    x0, y0, x1, y1 = Roi(0.99, 0.99, 0.5, 0.5).pixels(100, 80)
    assert x1 > x0 and y1 > y0


def test_meter_base_p999_ignores_hot_pixels():
    plane = np.full((100, 100), 1000.0)
    plane.flat[:3] = 60000.0  # 3 hot pixels = 0.03% < 0.1%, above the p99.9 cut
    # whole-frame ROI, black 200 → ~800 counts of base signal
    assert abs(meter_base(plane, Roi(0, 0, 1, 1), 200.0) - 800.0) < 1.0


def test_meter_black_median_robust_to_hot_pixels():
    plane = np.full((100, 100), 512.0)
    plane.flat[:50] = 65535.0  # 0.5% hot pixels would wreck a p99.9 black estimate
    assert meter_black(plane, Roi(0, 0, 1, 1)) == 512.0  # the median ignores them


def test_clip_fraction_counts_saturated_pixels():
    plane = np.full((100, 100), 1000.0)
    plane.flat[:5] = 65535.0  # 5 / 10000 = 0.05% at the ceiling
    assert abs(clip_fraction(plane, Roi(0, 0, 1, 1)) - 0.0005) < 1e-9
    assert clip_fraction(np.full((10, 10), 1000.0), Roi(0, 0, 1, 1)) == 0.0  # nothing saturated


def test_faster_shutter_steps_up_the_ladder():
    assert faster_shutter("0.8") == "0.6"
    assert faster_shutter("1/15") == "1/20"  # ladder is fastest-first
    assert faster_shutter("1/250") is None  # already the fastest


def test_slower_shutter_steps_down_the_ladder():
    assert slower_shutter("0.8") == "1"
    assert slower_shutter("1/250") == "1/200"
    assert slower_shutter("1") is None  # already the slowest


def test_shutter_at_least_picks_fastest_that_fits():
    assert shutter_at_least(0.2) == "1/5"  # 1/5 = 0.2 s, the fastest candidate ≥ 0.2 s
    assert shutter_at_least(0.19) == "1/5"  # 1/8=0.125 too fast, 1/5=0.2 is the first ≥ 0.19
    assert shutter_at_least(999.0) == "1"  # nothing slow enough → the slowest candidate


# ---- full loop with injected hardware -------------------------------------


class FakeLight:
    def __init__(self):
        self.last = (0, 0, 0)

    def set_color(self, r=0, g=0, b=0, w=0, save=False):
        self.last = (r, g, b)

    def off(self):
        self.last = (0, 0, 0)

    def close(self):
        pass


class FakeCamera:
    def __init__(self):
        self.last_shutter = "1/15"

    def capture(self, out_path, shutter=None):
        if shutter:
            self.last_shutter = shutter
        return os.path.splitext(out_path)[0] + ".ARW"  # the camera picks the suffix

    def close(self):
        pass


def _make_demosaic(light, camera, sliver: int = 0):
    """Linear fake sensor: 128×128 so a sub-0.1% clip sliver fits below the p99.9 cut.
    `sliver` over-bright pixels (base × 1.25) clip at the ETTR solve but the clip guard's
    LED-down resolves them (exposure-dependent, unlike a fixed hot pixel)."""

    def demosaic(_path):
        sec = shutter_seconds(camera.last_shutter)
        img = np.full((128, 128, 3), BLACK)
        for i, level in enumerate(light.last):
            val = BLACK + K[i] * level * sec
            img[..., i] = val
            if sliver:
                img.reshape(-1, 3)[:sliver, i] = min(65535.0, val * 1.25)
        np.clip(img, 0, 65535, out=img)
        return img

    return demosaic


def _make_capped_demosaic(light, camera, level_cap: int = 200):
    """Linear in shutter but LED response saturates above `level_cap`: raising the LED past it
    adds nothing, so a channel solved to max LED can land under target → shutter escalation
    (a slower shutter) is the only way to add exposure."""

    def demosaic(_path):
        sec = shutter_seconds(camera.last_shutter)
        img = np.full((32, 32, 3), BLACK)
        for i, level in enumerate(light.last):
            img[..., i] = BLACK + K[i] * min(level, level_cap) * sec
        np.clip(img, 0, 65535, out=img)
        return img

    return demosaic


def _service(light, cam, sliver: int = 0):
    # source_clip stubbed to 0 → the hardware-free path never touches rawpy.
    return CalibrationService(light, cam, _make_demosaic(light, cam, sliver), source_clip=lambda *_a: 0.0, sleep=lambda _s: None)


def test_source_clip_reads_the_file_the_camera_actually_wrote():
    """The camera names the file after its own RAW format, so the raw-Bayer clip check
    must be handed the path it returned — not the stem we asked for. Getting this wrong
    silently disables the clip guard: rawpy raises, the error is logged, and clip reads 0."""
    light, cam = FakeLight(), FakeCamera()
    seen: list[str] = []

    def record(path, _channel, _roi):
        seen.append(path)
        return 0.0

    service = CalibrationService(light, cam, _make_demosaic(light, cam, 0), source_clip=record, sleep=lambda _s: None)
    service.calibrate(Roi(0, 0, 1, 1), "/tmp/_negpy_calibration.raw")

    assert seen, "the clip guard never ran"
    assert all(p.endswith(".ARW") for p in seen), seen  # the fake camera's suffix, not ".raw"


def test_calibrate_fails_closed_when_the_raw_clip_check_errors(monkeypatch):
    light, cam = FakeLight(), FakeCamera()

    def unavailable(*_args):
        raise OSError("RAW decode failed")

    monkeypatch.setattr("negpy.infrastructure.capture.raw_demosaic.raw_channel_clip_fraction", unavailable)
    service = CalibrationService(light, cam, _make_demosaic(light, cam), sleep=lambda _s: None)

    with pytest.raises(RuntimeError, match="raw source-clip check failed") as caught:
        service.calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")

    assert isinstance(caught.value.__cause__, OSError)


def test_calibrate_fails_closed_on_a_nonfinite_raw_clip_measurement():
    light, cam = FakeLight(), FakeCamera()
    service = CalibrationService(
        light,
        cam,
        _make_demosaic(light, cam),
        source_clip=lambda *_args: np.nan,
        sleep=lambda _s: None,
    )

    with pytest.raises(RuntimeError, match="non-finite raw source-clip"):
        service.calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")


def test_calibrate_converges_with_one_shared_shutter():
    light, cam = FakeLight(), FakeCamera()
    result = _service(light, cam).calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")

    for letter in ("R", "G", "B"):
        c = result.channels[letter]
        assert 40 <= c.level <= 255
        assert abs(c.signal - c.target) <= 0.06 * c.target  # each channel hit ETTR target via LED

    r, g, b = result.shutters
    assert r == g == b  # ONE shutter shared across R/G/B — the whole point of the rebuild
    assert result.channels["B"].level > result.channels["R"].level  # blue dimmest → most LED
    assert light.last == (0, 0, 0)  # light off when done


def test_calibrate_rejects_a_probe_with_no_signal():
    light, cam = FakeLight(), FakeCamera()

    def dark_frame(_path):
        return np.full((16, 16, 3), BLACK)

    service = CalibrationService(light, cam, dark_frame, source_clip=lambda *_a: 0.0, sleep=lambda _s: None)

    with pytest.raises(RuntimeError, match="no signal from R"):
        service.calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")

    assert light.last == (0, 0, 0)


def test_calibrate_clip_guard_pulls_led_down():
    clean = _service(FakeLight(), FakeCamera()).calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")
    clipped = _service(FakeLight(), FakeCamera(), sliver=8).calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")

    # An over-bright sliver clips at the ETTR solve; the guard pulls the LED down until clean,
    # which necessarily lands the base a touch below the clean run's target.
    for c in clipped.channels.values():
        assert c.clip_fraction <= MAX_CLIP_FRACTION  # resolved, not left clipping
    assert clipped.channels["B"].signal < clean.channels["B"].signal  # LED was reduced


def test_calibrate_respects_camera_shutter_ladder():
    ladder = ("1/100", "1/50", "1/25", "1/10", "1/4", "1")  # a sparse, camera-specific ladder
    result = _service(FakeLight(), FakeCamera()).calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW", candidates=ladder)
    r, g, b = result.shutters
    assert r == g == b and r in ladder  # one shared shutter, from the camera's own set


def test_source_clip_pulls_led_down_when_demosaic_is_clean():
    clean = _service(FakeLight(), FakeCamera()).calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")
    # Demosaic reads clean, but the injected raw-Bayer check reports blue's photosites clipping
    # on the first look (resolving once the LED is pulled down) — the hidden-source-clip case.
    calls = {"b": 0}

    def src(_p, ch_i, _roi):
        if ch_i != 2:
            return 0.0
        calls["b"] += 1  # blue: 1=probe, 2=solve (clip here), 3=after LED-down (resolved)
        return 0.02 if calls["b"] == 2 else 0.0

    light, cam = FakeLight(), FakeCamera()
    guarded = CalibrationService(light, cam, _make_demosaic(light, cam), source_clip=src, sleep=lambda _s: None).calibrate(
        Roi(0, 0, 1, 1), "/tmp/cal.ARW"
    )
    assert guarded.channels["B"].clip_fraction == 0.0  # resolved after the LED-down
    assert guarded.channels["B"].signal < clean.channels["B"].signal  # LED was reduced to clear it


def test_calibrate_escalates_shared_shutter_when_led_saturates():
    clean = _service(FakeLight(), FakeCamera()).calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")
    light, cam = FakeLight(), FakeCamera()
    capped = CalibrationService(
        light, cam, _make_capped_demosaic(light, cam), source_clip=lambda *_a: 0.0, sleep=lambda _s: None
    ).calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")

    r, g, b = capped.shutters
    assert r == g == b  # still one shared shutter
    # LED saturates above 200 → blue can't reach target at the clean run's shutter, so the SHARED
    # shutter escalates to a slower one, and every channel is re-solved onto target.
    assert shutter_seconds(b) > shutter_seconds(clean.shutters[2])
    assert abs(capped.channels["B"].signal - capped.channels["B"].target) < 0.1 * capped.channels["B"].target


def test_calibrate_rejects_a_channel_below_target_at_the_hardware_limit():
    light, cam = FakeLight(), FakeCamera()
    service = CalibrationService(
        light,
        cam,
        _make_capped_demosaic(light, cam, level_cap=1),
        source_clip=lambda *_a: 0.0,
        sleep=lambda _s: None,
    )

    with pytest.raises(RuntimeError, match="R channel.*below target"):
        service.calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")


def test_calibrate_rejects_a_final_channel_materially_above_target():
    light, cam = FakeLight(), FakeCamera()
    ordinary = _make_demosaic(light, cam)
    calls = 0

    def nonlinear_response(path):
        nonlocal calls
        calls += 1
        if calls <= 4:  # dark frame + the three response probes
            return ordinary(path)
        img = np.full((32, 32, 3), BLACK)
        for i, level in enumerate(light.last):
            if level:
                img[..., i] = BLACK + 64500.0  # above ETTR, just below SATURATION_VALUE
        return img

    service = CalibrationService(light, cam, nonlinear_response, source_clip=lambda *_a: 0.0, sleep=lambda _s: None)

    with pytest.raises(RuntimeError, match="R channel.*above target"):
        service.calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")


def test_calibrate_rejects_a_final_channel_that_is_still_clipping():
    light, cam = FakeLight(), FakeCamera()
    service = CalibrationService(
        light,
        cam,
        _make_demosaic(light, cam),
        source_clip=lambda *_a: 0.02,
        sleep=lambda _s: None,
    )

    with pytest.raises(RuntimeError, match="R channel.*still clipping"):
        service.calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")


def test_calibrate_rejects_a_nonfinite_final_channel():
    light, cam = FakeLight(), FakeCamera()
    ordinary = _make_demosaic(light, cam)
    calls = 0

    def nonfinite_response(path):
        nonlocal calls
        calls += 1
        if calls <= 4:  # dark frame + the three response probes
            return ordinary(path)
        img = np.full((32, 32, 3), BLACK)
        for i, level in enumerate(light.last):
            if level:
                img[..., i] = np.nan
        return img

    service = CalibrationService(light, cam, nonfinite_response, source_clip=lambda *_a: 0.0, sleep=lambda _s: None)

    with pytest.raises(RuntimeError, match="R channel.*non-finite"):
        service.calibrate(Roi(0, 0, 1, 1), "/tmp/cal.ARW")
