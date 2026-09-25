import numpy as np

from negpy.features.exposure.logic import (
    CharacteristicCurve,
    apply_characteristic_curve,
    highlight_hold_offset,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS

SLOPE, PIVOT = 2.9, 0.21


def _density(preflash: float, x: np.ndarray, d_min: float = 0.0) -> np.ndarray:
    return np.asarray(CharacteristicCurve(SLOPE, PIVOT, d_min=d_min, preflash=preflash)(x)).ravel()


def test_zero_preflash_is_bit_identical():
    x = np.linspace(-0.3, 1.2, 32, dtype=np.float32)
    img = np.ascontiguousarray(np.repeat(x[None, :, None], 3, axis=2))
    args = ((PIVOT, SLOPE), (PIVOT, SLOPE), (PIVOT, SLOPE))
    base = apply_characteristic_curve(img, *args, frame_grade=115.0)
    off = apply_characteristic_curve(img, *args, frame_grade=115.0, preflash=0.0)
    np.testing.assert_array_equal(base, off)


def test_a_full_flash_alone_stays_under_threshold():
    threshold = EXPOSURE_CONSTANTS["preflash_threshold_density"]
    for d_min in (0.0, 0.06):
        bare = np.array([-5.0])
        fog = _density(1.0, bare, d_min) - _density(0.0, bare, d_min)
        assert 0.0 < fog[0] <= threshold + 0.005


def test_the_flash_darkens_highlights_most_and_keeps_the_curve_monotone():
    x = np.linspace(-1.0, 1.2, 400)
    d0 = _density(0.0, x)
    d1 = _density(0.8, x)
    lift = d1 - d0
    assert lift.min() >= 0.0
    assert np.all(np.diff(d1) > 0.0)
    peak = int(np.argmax(lift))
    assert d0[peak] < 0.8
    assert lift[int(np.argmin(np.abs(d0 - 2.0)))] < 0.35 * lift[peak]


def test_highlight_hold_tops_up_only_what_the_flash_leaves():
    held = highlight_hold_offset(SLOPE, PIVOT, 0.05)
    assert held > 0.0
    assert highlight_hold_offset(SLOPE, PIVOT, 0.05, preflash=0.2) < held
    assert highlight_hold_offset(SLOPE, PIVOT, 0.05, preflash=0.5) == 0.0


def test_the_flash_does_not_depend_on_the_scan_range():
    """A paper exposure: the same print curve gets the same flash whatever log range the scan had."""
    x = np.linspace(-0.3, 1.2, 32, dtype=np.float32)
    img = np.ascontiguousarray(np.repeat(x[None, :, None], 3, axis=2))
    args = ((PIVOT, SLOPE), (PIVOT, SLOPE), (PIVOT, SLOPE))
    wide = apply_characteristic_curve(img, *args, ev_scale=(0.1, 0.1, 0.1), preflash=0.5)
    narrow = apply_characteristic_curve(img, *args, ev_scale=(0.9, 0.9, 0.9), preflash=0.5)
    np.testing.assert_array_equal(wide, narrow)
