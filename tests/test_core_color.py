import numpy as np
import pytest

from negpy.kernel.image.logic import (
    _WORKING_GAMMA,
    _WORKING_TO_XYZ,
    _WORKING_WHITE,
    lab_to_rgb_working,
    rgb_to_lab_working,
    working_oetf_decode,
    working_oetf_encode,
)

_RNG = np.random.default_rng(11)


def _f32_ulps(a, b):
    ia, ib = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    return np.abs(ia - ib)


def test_oetf_is_the_float64_power_rounded_once():
    x = _RNG.random(200_000, dtype=np.float32) * 1.2 - 0.1
    x[:4] = [0.0, 1.0, -0.5, 1.5]
    clamped = np.clip(x, 0.0, 1.0).astype(np.float64)
    want = (clamped ** np.float64(np.float32(1.0 / _WORKING_GAMMA))).astype(np.float32)
    got = working_oetf_encode(x)
    assert np.max(_f32_ulps(got, want)) <= 1
    assert np.mean(got != want) < 1e-5
    back = working_oetf_decode(got)
    want = (np.maximum(got, 0.0).astype(np.float64) ** np.float64(np.float32(_WORKING_GAMMA))).astype(np.float32)
    assert np.max(_f32_ulps(back, want)) <= 1


def test_oetf_keeps_nan():
    assert np.isnan(working_oetf_encode(np.array([np.nan], np.float32))[0])
    assert np.isnan(working_oetf_decode(np.array([np.nan], np.float32))[0])


def _lab_reference(rgb):
    lin = np.maximum(rgb.astype(np.float64), 0.0)
    t = (lin @ _WORKING_TO_XYZ.astype(np.float64).T) / _WORKING_WHITE.astype(np.float64)
    eps, kappa, c = float(np.float32(0.008856)), float(np.float32(7.787)), float(np.float32(16.0 / 116.0))
    f = np.where(t > eps, np.cbrt(t), kappa * t + c)
    return np.stack([116.0 * f[:, 1] - 16.0, 500.0 * (f[:, 0] - f[:, 1]), 200.0 * (f[:, 1] - f[:, 2])], axis=1)


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_lab_matches_a_float64_reference_and_round_trips(scale):
    rgb = (_RNG.random((50_000, 3), dtype=np.float32) * scale).astype(np.float32)
    lab = rgb_to_lab_working(rgb)
    assert np.max(np.abs(lab - _lab_reference(rgb))) < 1e-4
    back = lab_to_rgb_working(lab)
    assert np.max(np.abs(back - rgb)) < 1e-5 * max(scale, 0.1)


def test_lab_keeps_the_shape():
    img = _RNG.random((7, 5, 3), dtype=np.float32)
    assert rgb_to_lab_working(img).shape == (7, 5, 3)
    assert lab_to_rgb_working(rgb_to_lab_working(img)).shape == (7, 5, 3)
