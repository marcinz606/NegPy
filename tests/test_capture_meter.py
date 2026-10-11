import numpy as np
import pytest

from negpy.services.capture.calibration import CLIP_CEILING, SHUTTER_CANDIDATES, true_seconds
from negpy.services.capture.meter import CLIPPED_STEP_STOPS, PEAK_MAP_EDGE, meter_frame, shutter_at_most


def _frame(border: float, picture: float, h: int = 600, w: int = 900, orange: bool = True) -> np.ndarray:
    """A film strip across the canvas: `border` outside the picture, textured `picture` inside.
    `orange` tints it like a C-41 base, so the classifier reads a negative."""
    rng = np.random.default_rng(2)
    img = np.full((h, w, 3), border, np.float32)
    y1, y2, x1, x2 = int(0.15 * h), int(0.85 * h), int(0.2 * w), int(0.8 * w)
    img[y1:y2, x1:x2] = picture * (0.5 + 0.5 * rng.random((y2 - y1, x2 - x1, 1)))
    if orange:
        img[:, :, 1] *= 0.7
        img[:, :, 2] *= 0.5
    return img * CLIP_CEILING


def _is_ceiling(label: str, ideal: float) -> bool:
    """`label` is the slowest rung at or under `ideal` seconds."""
    i = SHUTTER_CANDIDATES.index(label)
    slower = SHUTTER_CANDIDATES[i + 1] if i + 1 < len(SHUTTER_CANDIDATES) else None
    return true_seconds(label) <= ideal and (slower is None or true_seconds(slower) > ideal)


def test_a_negative_places_its_base_at_the_target():
    reading = meter_frame(_frame(border=0.6, picture=0.25), "1/30", medium="negative")

    assert (reading.medium, reading.region) == ("negative", "base")
    assert reading.measured == pytest.approx(0.6 * CLIP_CEILING, rel=0.02)
    ideal = true_seconds("1/30") * 0.9 / 0.6
    assert reading.recommended is not None and _is_ceiling(reading.recommended, ideal)
    # The stops reported are the move the rung makes, not the ideal.
    assert reading.stops == pytest.approx(np.log2(true_seconds(reading.recommended) / true_seconds("1/30")), abs=1e-6)


def test_a_positive_places_its_highlights_at_the_target():
    reading = meter_frame(_frame(border=0.02, picture=0.5, orange=False), "1/30", medium="positive")

    assert (reading.medium, reading.region) == ("positive", "highlights")
    assert reading.measured == pytest.approx(0.5 * CLIP_CEILING, rel=0.02)
    assert reading.clipped_fraction == 0.0
    assert reading.recommended is not None and _is_ceiling(reading.recommended, true_seconds("1/30") * 0.9 / 0.5)


def test_without_a_medium_the_classifier_decides():
    # An orange base reads as a color negative to the import's classifier.
    reading = meter_frame(_frame(border=0.6, picture=0.25), "1/30")

    assert (reading.medium, reading.region) == ("negative", "base")


def test_a_negative_whose_rebate_is_not_brighter_than_its_picture_meters_the_picture():
    # No visible base: the ring the refiner leaves is picture, which the base never is darker than.
    reading = meter_frame(_frame(border=0.02, picture=0.25), "1/30", medium="negative")

    assert reading.region == "highlights"
    assert reading.measured == pytest.approx(0.25 * CLIP_CEILING, rel=0.02)


def test_a_clipped_base_is_a_floor_and_steps_down():
    reading = meter_frame(_frame(border=1.0, picture=0.25), "1/30", medium="negative")

    assert reading.region == "base" and reading.clipped
    assert reading.stops == pytest.approx(CLIPPED_STEP_STOPS, abs=0.2)
    assert reading.recommended is not None and true_seconds(reading.recommended) <= true_seconds("1/30") / 4


def test_clipped_highlights_step_down_and_say_how_much():
    reading = meter_frame(_frame(border=0.02, picture=1.5, orange=False), "1/30", medium="positive")

    assert reading.clipped
    assert reading.clipped_fraction > 0.02
    assert reading.recommended is not None and true_seconds(reading.recommended) <= true_seconds("1/30") / 4


def test_fine_clipped_highlights_count_at_the_probe_resolution():
    # 2x2 glints on a quarter percent of the picture vanish at detection size; the probe's own pixels keep them.
    img = _frame(border=0.02, picture=0.3, h=2400, w=3600, orange=False)
    rng = np.random.default_rng(5)
    ys, xs = rng.integers(400, 2000, 2200), rng.integers(760, 2860, 2200)
    for y, x in zip(ys, xs):
        img[y : y + 2, x : x + 2] = CLIP_CEILING

    reading = meter_frame(img, "1/30", medium="positive")

    assert reading.clipped_fraction == pytest.approx(8800 / (1680 * 2160), rel=0.1)
    # p99.9 sees the glints at saturation: a lower bound, so the probe steps down and asks again.
    assert reading.clipped and reading.stops == pytest.approx(CLIPPED_STEP_STOPS, abs=0.4)


def test_a_saturated_p999_is_never_metered_as_a_level():
    """A saturated p99.9 is a lower bound: the probe steps down and asks again instead of
    nudging a third of a stop."""
    img = _frame(border=0.02, picture=0.3, h=600, w=900, orange=False)
    img[120:200, 200:700] = CLIP_CEILING  # about 9% of the picture blown
    reading = meter_frame(img, "1/30", medium="positive")
    assert reading.clipped and reading.stops == pytest.approx(CLIPPED_STEP_STOPS, abs=0.4)


def test_a_probe_with_no_film_box_meters_the_picture_not_the_light_around_it():
    """Without a film box there is no rebate: the bright surround is not taken for a base."""
    rng = np.random.default_rng(3)
    img = (0.2 + 0.1 * rng.random((600, 900, 3))).astype(np.float32) * CLIP_CEILING  # no edges at all
    reading = meter_frame(img, "1/30", medium="negative")
    assert reading.region == "highlights"


def test_a_target_past_the_fastest_rung_is_reported_not_overshot():
    # Base over target at the fastest rung: no rung is fast enough, and none is written.
    reading = meter_frame(_frame(border=0.97, picture=0.25), SHUTTER_CANDIDATES[0], medium="negative")

    assert reading.recommended is None
    assert reading.stops == pytest.approx(np.log2(0.9 / 0.97), abs=0.05)  # the move still needed


def test_a_blank_probe_meters_the_whole_frame():
    reading = meter_frame(np.zeros((120, 180, 3), np.float32), "1/30", medium="positive")

    assert reading.region == "highlights"
    assert reading.recommended == SHUTTER_CANDIDATES[-1]


def test_shutter_at_most_never_overshoots():
    assert true_seconds(shutter_at_most(0.05, SHUTTER_CANDIDATES)) <= 0.05
    assert shutter_at_most(1e-6, SHUTTER_CANDIDATES) is None
    assert shutter_at_most(1e6, SHUTTER_CANDIDATES) == SHUTTER_CANDIDATES[-1]


def test_the_peak_map_keeps_a_clip_narrower_than_a_block():
    img = _frame(border=0.6, picture=0.25, h=2400, w=3600)
    img[1200, 1800] = CLIP_CEILING  # one saturated pixel
    reading = meter_frame(img, "1/30", medium="negative")
    assert max(reading.peaks.shape) <= PEAK_MAP_EDGE
    assert float(reading.peaks.max()) >= 1.0 - 1e-6


def test_the_raw_clip_mask_predicts_the_metered_shutter():
    """The RAW is linear: a probe pixel at half saturation clips one stop slower, not at the
    probe's own shutter."""
    from dataclasses import replace

    from negpy.services.capture.meter import raw_clip_mask

    reading = meter_frame(_frame(border=0.6, picture=0.25), "1/30", medium="negative")
    peaks = np.full((4, 6), 0.3, np.float32)
    peaks[0, 0] = 0.55  # clips one stop up
    peaks[1, 1] = 1.0  # clipped at the probe: unknown, so it stays marked
    at_probe = raw_clip_mask(replace(reading, peaks=peaks, stops=0.0))
    one_up = raw_clip_mask(replace(reading, peaks=peaks, stops=1.0))
    one_down = raw_clip_mask(replace(reading, peaks=peaks, stops=-1.0))
    assert at_probe.sum() == 1 and at_probe[1, 1]
    assert one_up.sum() == 2 and one_up[0, 0] and one_up[1, 1]
    assert one_down.sum() == 1 and one_down[1, 1]
    assert raw_clip_mask(replace(reading, recommended=None)) is None


def test_the_peak_map_keeps_the_edge_blocks():
    from negpy.services.capture.meter import _peak_map

    brightest = np.zeros((3001, 4003), np.float32)
    brightest[-1, -1] = 1.0  # a clip in the last row and column, past a whole block
    peaks = _peak_map(brightest)
    assert float(peaks[-1, -1]) == 1.0
    assert _peak_map(np.ones((2, 5000), np.float32)).shape[0] == 1  # a short edge under one block


def _scene() -> np.ndarray:
    """An asymmetric picture: a diagonal light falloff and a bright block off center."""
    h, w = 600, 900
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lin = 0.1 + 0.5 * (xx / w) * (1 - 0.6 * yy / h)
    lin[100:220, 600:760] = 0.9
    return np.dstack([lin, lin * 0.9, lin * 0.8]) * CLIP_CEILING


def _jpeg(img: np.ndarray) -> np.ndarray:
    return (255 * np.clip(img / CLIP_CEILING, 0, 1) ** (1 / 2.2)).astype(np.uint8)


def test_the_framing_check_passes_a_tone_curved_preview_of_the_same_picture():
    from negpy.services.capture.meter import framing_mismatch

    img = _scene()
    reading = meter_frame(img, "1/30", medium="positive")
    assert framing_mismatch(reading, _jpeg(img)[::6, ::6]) is None


def test_the_framing_check_refuses_another_aspect_and_another_picture():
    from negpy.services.capture.meter import framing_mismatch

    img = _scene()
    reading = meter_frame(img, "1/30", medium="positive")
    preview = _jpeg(img)
    assert "preview is" in framing_mismatch(reading, preview[75:-75])  # a 16:9 crop of the 3:2 frame
    advanced = np.ascontiguousarray(np.roll(preview, 300, axis=1))  # the same size, another framing
    assert framing_mismatch(reading, advanced) == "the preview does not frame the RAW the same way"


def test_a_clipped_probe_past_the_fastest_rung_still_steps_to_it():
    """Two stops down passes the ladder's end, but the fastest rung is still a step down."""
    img = _frame(border=0.02, picture=0.3, h=600, w=900, orange=False)
    img[120:200, 200:700] = CLIP_CEILING
    ladder = ("1/250", "1/125", "1/60")
    reading = meter_frame(img, "1/125", ladder, medium="positive")
    assert reading.clipped and reading.recommended == "1/250"
    at_fastest = meter_frame(img, "1/250", ladder, medium="positive")
    assert at_fastest.recommended is None  # nothing faster to step to
