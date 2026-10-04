import numpy as np

from negpy.services.capture.focus_meter import FocusMeter, sharpness


def _scene(blur: int = 0) -> np.ndarray:
    rng = np.random.default_rng(3)
    img = rng.uniform(40.0, 200.0, (120, 160))
    for _ in range(blur):
        img = sum(np.roll(np.roll(img, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    return img


def test_sharpness_falls_as_the_frame_blurs():
    scores = [sharpness(_scene(blur)) for blur in range(4)]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1] * 10


def test_sharpness_does_not_depend_on_the_light_level():
    scene = _scene(1)
    assert np.isclose(sharpness(scene), sharpness(scene * 0.4), rtol=1e-4)


def test_a_dark_or_degenerate_frame_has_no_sharpness():
    assert sharpness(np.zeros((120, 160))) == 0.0
    assert sharpness(np.full((2, 2), 128.0)) == 0.0


def test_the_meter_holds_the_peak_until_reset():
    meter = FocusMeter()
    assert meter.update(_scene(2)) == 1.0
    for _ in range(30):
        reading = meter.update(_scene(0))
    assert reading == 1.0
    for _ in range(30):
        reading = meter.update(_scene(2))
    assert reading < 0.5
    meter.reset()
    assert meter.update(_scene(2)) == 1.0


def test_a_dark_frame_gives_no_reading_and_keeps_the_peak():
    meter = FocusMeter()
    meter.update(_scene(0))
    assert meter.update(np.zeros((120, 160))) is None
    assert meter.update(_scene(0)) == 1.0
