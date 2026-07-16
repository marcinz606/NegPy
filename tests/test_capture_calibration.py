"""ETTR auto-calibration unit tests.

Fake linear sensor: Signal = k · level · true_seconds, no bias (rawpy removes it → black = 0). It
meters the ladder's true exposure, like a real body — NOT the rounded label, which is a display
name ("1/3" exposes 0.315 s).
Only the lit channel gets signal (one LED on per probe, as on the real rig). `k_scale` scales all
three uniformly, like opening the aperture; per-channel k models the deep-red weakness (R lowest).
"""

import os

import numpy as np
import pytest

from negpy.services.capture.calibration import (
    MAX_CLIP_FRACTION,
    PWM_MAX,
    PWM_MAX_SAFE,
    PWM_MIN,
    REFERENCE_LEVELS,
    REFERENCE_SHUTTER,
    SHUTTER_CANDIDATES,
    CalibrationService,
    ChannelCalibration,
    Roi,
    _channel_status,
    _solve_shared,
    _spread_stops,
    aperture_fnumber,
    clip_fraction,
    meter_base,
    _ladder_stops,
    nearest_shutter,
    normalize_start_point,
    shutter_at_least,
    shutter_seconds,
    target_signal,
    true_seconds,
    usable_ladder,
)

# Per-channel response (counts per LED-level per second). R is the weakest (665 nm, low sensor QE),
# G≈B — the ~1.6-stop spread measured from Robin's f/8 C-41 logs.
K = {"R": 250.0, "G": 700.0, "B": 760.0}


# ---- pure functions -------------------------------------------------------


def test_shutter_seconds():
    assert shutter_seconds("1/100") == 0.01
    assert shutter_seconds("0.4") == 0.4
    assert shutter_seconds("1") == 1.0


def test_shutter_seconds_rejects_zero_denominator():
    # The a7 IV publishes a bulb-like "1/0"; it must raise ValueError (not ZeroDivisionError) so the
    # shutter-ladder filter drops it instead of crashing calibration.
    with pytest.raises(ValueError):
        shutter_seconds("1/0")


# Real third-stop ladder as published by the a7C II (slowest first).
_THIRDS = ("1", "8/10", "6/10", "5/10", "4/10", "1/3", "1/4", "1/5", "1/6", "1/8", "1/10", "1/13", "1/15", "1/20")
# A half-stop body: same style of labels, different rungs — "1/3" here is 2^(-3/2) s, not 2^(-5/3).
_HALVES = ("1", "1/1.5", "1/2", "1/3", "1/4", "1/6", "1/8", "1/11", "1/15", "1/22", "1/30")


def test_true_seconds_undoes_the_label_rounding():
    # Labels are rounded display names for a geometric ladder. Metering against the fraction put a
    # rig run ~4 % under target with the LED clamped: it probed at "0.4" (0.8 % off) and solved at
    # "1/3" (5.8 % off), and the difference went straight into k.
    assert true_seconds("1", _THIRDS) == pytest.approx(1.0, rel=1e-3)  # exact rungs stay exact
    assert true_seconds("1/4", _THIRDS) == pytest.approx(0.25, rel=1e-3)
    assert true_seconds("1/8", _THIRDS) == pytest.approx(0.125, rel=1e-3)
    assert true_seconds("1/3", _THIRDS) == pytest.approx(2 ** (-5 / 3), rel=0.01)  # 0.315, not 0.333
    assert true_seconds("1/6", _THIRDS) == pytest.approx(2 ** (-8 / 3), rel=0.01)  # 0.157, not 0.167
    assert true_seconds("4/10", _THIRDS) == pytest.approx(2 ** (-4 / 3), rel=0.01)  # 0.397, not 0.4
    # The nominal parse is untouched — ordering/snapping still use it, and the ladder is monotonic
    # either way, so no caller that only sorts is affected.
    assert shutter_seconds("1/3") == pytest.approx(1 / 3)


def test_ladder_stops_is_measured_from_the_body_not_assumed():
    # Which rung a rounded label denotes depends on the ladder's spacing, so it is read off the
    # body's own labels. Assuming thirds on a half-stop body would be worse than not correcting at
    # all (it would "fix" 1/3 to 0.315 when the body really exposes 0.354).
    assert _ladder_stops(_THIRDS) == pytest.approx(1 / 3)
    assert _ladder_stops(_HALVES) == pytest.approx(1 / 2)
    assert true_seconds("1/3", _HALVES) == pytest.approx(2 ** (-3 / 2), rel=0.02)  # 0.354, not 0.315
    # Too short to read → falls back to thirds rather than inventing a spacing.
    assert _ladder_stops(("1/4", "1/8")) == pytest.approx(1 / 3)


def test_true_seconds_leaves_labels_that_are_not_on_the_ladder_alone():
    # A correction may never exceed the rounding it undoes. A value that sits nowhere near a rung
    # isn't a rounded rung — it's something else (bulb, an oddly labelled body), and then the label
    # is the better guess.
    assert true_seconds("0.7", _THIRDS) == pytest.approx(0.7, rel=1e-6)
    for label in _THIRDS:  # every real rung IS corrected, and only slightly
        assert true_seconds(label, _THIRDS) == pytest.approx(shutter_seconds(label), rel=0.08)


def test_calibrate_survives_a_raw_unfiltered_ladder_from_the_body():
    # #478 was exactly this: a body publishes "1/0" (bulb-like) and calibration died while building
    # the ladder. The UI filters those out, but the solver must not depend on that — a second caller
    # handing over a raw ladder would reopen the bug. The ladder is cleaned on the way in instead.
    raw = ("1/250", "1/0", "Bulb", "", "1/60", "1/4", "2", "30")
    # Only unparseable labels go: this cleans, it does not range-filter. Which speeds are *sensible*
    # (the PWM-banding floor, the 2 s ceiling) is the UI's call in _available_shutters — the solver
    # solves on whatever ladder it is handed, it just must not choke on it.
    assert usable_ladder(raw) == ("1/250", "1/60", "1/4", "2", "30")  # junk dropped, ascending
    assert _ladder_stops(raw) > 0  # no ValueError from the unparseable entries
    assert shutter_at_least(0.5, usable_ladder(raw)) == "2"
    light, cam = FakeLight(), FakeCamera()
    result = _service(light, cam).calibrate(Roi(0, 0, 1, 1), "/tmp/_negpy_cal.raw", candidates=raw)
    assert set(result.channels) == {"R", "G", "B"}


def test_calibrate_falls_back_when_no_ladder_entry_is_usable():
    # A body that publishes only bulb-like labels leaves nothing to solve on. Falling back to the
    # built-in ladder beats crashing on an empty tuple (shutter_at_least indexes candidates[-1]).
    light, cam = FakeLight(), FakeCamera()
    result = _service(light, cam).calibrate(Roi(0, 0, 1, 1), "/tmp/_negpy_cal.raw", candidates=("1/0", "Bulb", ""))
    assert result.channels["R"].shutter in SHUTTER_CANDIDATES


def test_solve_uses_the_true_exposure_not_the_rounded_label():
    # The regression this fixes: solving levels against the label over-states the light by up to
    # 5.8 %, the channel lands under target, and the trim runs into the PWM_MAX clamp.
    T = target_signal()
    shutter, levels = _solve_shared(K, T, _THIRDS)
    secs = true_seconds(shutter, _THIRDS)
    for c, level in levels.items():
        if PWM_MIN < level < PWM_MAX:  # unclamped channels must land on target at the TRUE exposure
            assert K[c] * level * secs == pytest.approx(T, rel=0.02)


def test_aperture_fnumber_parses_labels_and_manual_lens():
    assert aperture_fnumber("f/8") == 8.0
    assert aperture_fnumber("F5.6") == 5.6
    assert aperture_fnumber("11") == 11.0
    assert aperture_fnumber("") is None  # manual lens, no electronic aperture
    assert aperture_fnumber("—") is None


def test_normalize_start_point_scales_shutter_by_iso_and_aperture():
    # Reference is ISO 100 / f8 / 0.4 s. Exposure ∝ ISO·t/f², so t scales by (100/ISO)·(f/8)².
    levels, shutter = normalize_start_point("100", "f/8")
    assert levels == REFERENCE_LEVELS and shutter == REFERENCE_SHUTTER
    # ISO 400 = 2 stops more sensitive → 2 stops faster (0.4 s → 1/10).
    assert normalize_start_point("400", "f/8")[1] == "1/10"
    # f/11 ≈ 1 stop less light → ~1 stop slower (0.4 s → 0.8 s), NOT mislabelled as "1/1".
    assert normalize_start_point("100", "f/11")[1] == "0.8"
    # f/16 = 2 stops less light → 2 stops slower (0.4 s → 1.6 s), still on the ladder.
    assert normalize_start_point("100", "f/16")[1] == "1.6"
    # Manual lens (aperture unreadable) → ISO-only correction, no crash.
    assert normalize_start_point("200", "")[1] == "1/5"


def test_channel_status_is_measured_based_not_level_based():
    # Bug caught in review: a clip-guard can pull the LED well below PWM_MAX yet leave the signal
    # materially under target — that must read "under", not "target". The status keys off the
    # measured signal, not the level.
    T = target_signal()
    assert _channel_status(T, 0.0, T) == "target"
    assert _channel_status(0.85 * T, 0.0, T) == "target"  # small undershoot within the margin
    assert _channel_status(0.5 * T, 0.0, T) == "under"  # materially under (level irrelevant here)
    assert _channel_status(T, 0.01, T) == "over"  # still clipping → over


def test_spread_stops():
    assert _spread_stops({"R": 250.0, "G": 700.0, "B": 760.0}) == pytest.approx(1.60, abs=0.02)
    assert _spread_stops({"R": 100.0, "G": 100.0, "B": 100.0}) == 0.0


def test_solve_shared_seats_the_dimmest_channel_near_pwm_max_safe():
    T = target_signal()
    shutter, levels = _solve_shared(K, T, SHUTTER_CANDIDATES)
    # Dimmest channel (R) gets the highest level, seated just under PWM_MAX_SAFE (not 255 — the red
    # LED saturates up there); brighter G/B lower. Never above PWM_MAX_SAFE, and at most one ladder
    # third-stop below it after the shutter snap.
    assert levels["R"] > levels["G"] > levels["B"]
    assert PWM_MAX_SAFE * 0.79 <= levels["R"] <= PWM_MAX_SAFE
    # Every channel is inside the LED window at the chosen shutter.
    secs = shutter_seconds(shutter)
    for c, lvl in levels.items():
        assert 40 <= lvl <= 255
        assert K[c] * lvl * secs == pytest.approx(T, rel=0.06)


def test_nearest_shutter_snaps_to_the_closest_candidate():
    assert nearest_shutter("0.16", ("1/8", "1/6", "1/5")) == "1/6"  # 0.16 s closest to 1/6 (0.167)
    assert nearest_shutter("1/5", ("1/8", "1/6", "1/5")) == "1/5"  # already a candidate → unchanged


def test_shutter_at_least_picks_fastest_that_fits():
    # "≥ seconds" is a claim about light, so it compares TRUE exposure. "1/5" exposes 2^(-7/3) =
    # 0.198 s — it does NOT satisfy a 0.2 s demand, however its label reads. Trusting the label
    # here is what let the solved level exceed PWM_MAX_SAFE and pin R at the 255 clamp on the rig.
    assert shutter_at_least(0.198) == "1/5"
    assert shutter_at_least(0.19) == "1/5"
    assert shutter_at_least(0.2) == "1/4"  # 1/5 is really 0.198 s → too fast, take the next rung
    assert shutter_at_least(999.0) == "2"  # nothing slow enough → slowest candidate (now 2 s)


def test_solve_never_exceeds_pwm_max_safe_on_the_dimmest_channel():
    # The guarantee shutter_at_least exists for: t_ideal is derived so the dimmest channel sits at
    # PWM_MAX_SAFE, so any rung that truly exposes ≥ t_ideal keeps it at or below that. Comparing
    # nominal labels broke the guarantee (the rung exposed less than promised), the level overshot,
    # and the trim then had no headroom left.
    T = target_signal()
    for ladder in (_THIRDS, SHUTTER_CANDIDATES):
        _shutter, levels = _solve_shared(K, T, ladder)
        dimmest = min(K, key=lambda c: K[c])
        assert levels[dimmest] <= PWM_MAX_SAFE


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
    def __init__(self, start="1/5"):
        self.last_shutter = start

    def capture(self, out_path, shutter=None, iso=None, aperture=None):
        if shutter:
            self.last_shutter = shutter
        return os.path.splitext(out_path)[0] + ".ARW"  # the camera picks the suffix

    def close(self):
        pass


def _make_demosaic(light, camera, *, k_scale=1.0, level_cap=None, sliver=0):
    """Linear fake sensor (128×128 so a sub-0.1 % clip sliver fits below the p99.9 cut). No bias.
    `k_scale` scales all channels uniformly (like aperture); `level_cap` saturates the LED above a
    level (a channel solved to max LED lands under target → under-exposed); `sliver` over-bright
    pixels clip at the solve but the LED-down clip guard resolves them."""

    def demosaic(_path):
        # A body exposes the ladder's TRUE time, not the label's fraction ("1/3" is 0.315 s). The
        # fake sensor must do the same, or it silently absorbs the rounding the solver has to
        # handle — and the rig failure it caused would be untestable here.
        sec = true_seconds(camera.last_shutter, SHUTTER_CANDIDATES)
        img = np.zeros((128, 128, 3))
        for i, level in enumerate(light.last):
            eff = min(level, level_cap) if level_cap is not None else level
            val = K["RGB"[i]] * k_scale * eff * sec
            img[..., i] = val
            if sliver:
                img.reshape(-1, 3)[:sliver, i] = min(65535.0, val * 1.25)
        np.clip(img, 0, 65535, out=img)
        return img

    return demosaic


def _service(light, cam, **demo):
    # source_clip stubbed to 0 → the hardware-free path never touches rawpy.
    return CalibrationService(light, cam, _make_demosaic(light, cam, **demo), source_clip=lambda *_a: 0.0, sleep=lambda _s: None)


def _calibrate(service, roi=Roi(0, 0, 1, 1)):
    return service.calibrate(roi, "/tmp/_negpy_cal.raw")


def test_calibrate_converges_with_one_shared_shutter():
    light, cam = FakeLight(), FakeCamera()
    result = _calibrate(_service(light, cam))
    T = target_signal()
    # One shared shutter, every channel on target, R (dimmest) at the highest level.
    assert len(set(result.shutters)) == 1
    assert result.levels[0] > result.levels[1] and result.levels[0] > result.levels[2]
    for ch in result.channels.values():
        assert ch.status == "target"
        assert ch.signal == pytest.approx(T, rel=0.06)
    assert result.spread_stops == pytest.approx(1.60, abs=0.05)


def test_calibrate_recovers_from_a_much_brighter_than_expected_start():
    # Aperture way open (~6 stops brighter than the start point) → the probe clips hard. The halving
    # back-off must recover within the step budget — a 1/3-stop walk would run out of steps here.
    light, cam = FakeLight(), FakeCamera()
    result = _calibrate(_service(light, cam, k_scale=64.0))
    for ch in result.channels.values():
        assert ch.status == "target"
        assert ch.clip_fraction <= MAX_CLIP_FRACTION


def test_calibrate_recovers_from_a_too_fast_start_shutter():
    # Start shutter far too fast (1/250) but the light is fine — the analytic solve still lands on
    # target from the probe's clean (if dim) reading, no ladder walk needed.
    light, cam = FakeLight(), FakeCamera(start="1/250")
    result = _service(light, cam).calibrate(Roi(0, 0, 1, 1), "/tmp/_negpy_cal.raw", start_shutter="1/250")
    for ch in result.channels.values():
        assert ch.status == "target"


def test_calibrate_is_graceful_when_a_channel_cannot_reach_target():
    # Aperture too closed: even max LED at the slowest shutter (2 s) leaves the dim R channel short.
    # Must not raise — return a best-effort result with R flagged "under".
    light, cam = FakeLight(), FakeCamera()
    result = _calibrate(_service(light, cam, k_scale=0.02))
    assert result.channels["R"].status == "under"
    assert result.channels["R"].level == PWM_MAX
    assert result.status == "under"  # headline reflects the worst channel


def test_calibrate_is_graceful_when_over_exposed():
    # Aperture way too open: even the fastest shutter + lowest LED clips. Must not raise (symmetric
    # to under-exposure) — return a best-effort result at minimum exposure with the channels flagged
    # "over", so the UI can tell the user to stop down.
    light, cam = FakeLight(), FakeCamera(start="1/250")
    result = _service(light, cam, k_scale=3000.0).calibrate(Roi(0, 0, 1, 1), "/tmp/_negpy_cal.raw", start_shutter="1/250")
    assert result.status == "over"
    assert all(ch.status == "over" for ch in result.channels.values())
    assert all(ch.level == PWM_MIN for ch in result.channels.values())  # seated at minimum LED


def test_deep_over_exposure_from_the_default_start_never_exhausts_the_probe():
    # The probe budget must cover the whole reachable range (ladder + LED), or a deeply-over scene
    # exhausts it mid-descent and raises "no signal … check the Scanlight is on" — the exact
    # opposite of what is happening. Both cases below did exactly that with the old 8-step budget.
    #
    # ~8 stops over (manual f/1.4 lens the body can't report, no film in the holder): within the
    # ladder's ~9-stop reach below the start, so with enough steps it now CALIBRATES, on target.
    light, cam = FakeLight(), FakeCamera()
    result = _calibrate(_service(light, cam, k_scale=300.0))
    assert result.status == "target"
    T = target_signal()
    assert all(abs(ch.signal - T) <= 0.06 * T for ch in result.channels.values())
    # ~11.6 stops over: beyond even minimum exposure — must degrade to "over", never raise.
    light, cam = FakeLight(), FakeCamera()
    result = _calibrate(_service(light, cam, k_scale=3000.0))
    assert result.status == "over"


def test_calibrate_raises_only_when_a_channel_has_no_signal_at_all():
    # A dead LED / ROI off the base gives no signal even at max exposure — the one case that still
    # errors clearly (nothing can be calibrated), distinct from graceful over/under-exposure.
    light, cam = FakeLight(), FakeCamera()
    with pytest.raises(RuntimeError, match="no signal from the R channel"):
        _calibrate(_service(light, cam, k_scale=0.0))


def test_calibrate_clip_guard_pulls_the_led_down_below_clipping():
    # A bright sliver clips at the solved level; the guard lowers the LED until the base is clean.
    light, cam = FakeLight(), FakeCamera()
    result = _calibrate(_service(light, cam, sliver=40))
    for ch in result.channels.values():
        assert ch.clip_fraction <= MAX_CLIP_FRACTION


def test_calibrate_measures_spread_matching_the_channel_responses():
    light, cam = FakeLight(), FakeCamera()
    result = _calibrate(_service(light, cam))
    # log2(760/250) ≈ 1.60 — the value that confirms one shutter can serve all three (< 2.7).
    assert result.spread_stops == pytest.approx(1.60, abs=0.05)
    assert result.spread_stops < 2.7


# ---- source-clip guard (fail-closed) --------------------------------------


def test_source_clip_reads_the_file_the_camera_actually_wrote():
    """The raw-Bayer clip check must be handed the path the camera returned (its own RAW suffix),
    not the stem we asked for — else the clip guard is silently disabled."""
    light, cam = FakeLight(), FakeCamera()
    seen: list[str] = []

    def record(path, _channel, _roi):
        seen.append(path)
        return 0.0

    service = CalibrationService(light, cam, _make_demosaic(light, cam), source_clip=record, sleep=lambda _s: None)
    _calibrate(service)
    assert seen and all(p.endswith(".ARW") for p in seen), seen


def test_calibrate_fails_closed_when_the_raw_clip_check_errors(monkeypatch):
    light, cam = FakeLight(), FakeCamera()

    def unavailable(*_args):
        raise OSError("RAW decode failed")

    monkeypatch.setattr("negpy.infrastructure.capture.raw_demosaic.raw_channel_clip_fraction", unavailable)
    service = CalibrationService(light, cam, _make_demosaic(light, cam), sleep=lambda _s: None)
    with pytest.raises(RuntimeError, match="raw source-clip check failed") as caught:
        _calibrate(service)
    assert isinstance(caught.value.__cause__, OSError)


def test_calibrate_fails_closed_on_a_nonfinite_raw_clip_measurement():
    light, cam = FakeLight(), FakeCamera()
    service = CalibrationService(light, cam, _make_demosaic(light, cam), source_clip=lambda *_a: np.nan, sleep=lambda _s: None)
    with pytest.raises(RuntimeError, match="non-finite raw source-clip"):
        _calibrate(service)


# ---- metering helpers -----------------------------------------------------


def test_meter_base_is_p999_of_the_roi():
    plane = np.full((100, 100), 30000.0)
    plane.reshape(-1)[:5] = 65000.0  # 0.05 % bright — inside the top 0.1 % that p99.9 discards
    assert meter_base(plane, Roi(0, 0, 1, 1)) == pytest.approx(30000.0, abs=1.0)


def test_clip_fraction_counts_saturated_pixels():
    plane = np.zeros((100, 100))
    plane.reshape(-1)[:100] = 65500.0  # 1 % at/above saturation
    assert clip_fraction(plane, Roi(0, 0, 1, 1)) == pytest.approx(0.01, abs=1e-4)


def test_channel_calibration_defaults_to_target_status():
    c = ChannelCalibration(channel="R", level=200, shutter="1/5", signal=59000, target=58982)
    assert c.status == "target"
