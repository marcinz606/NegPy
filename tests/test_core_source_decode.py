"""The Positive source decode kernels against the NumPy code they replaced, and a pin."""

import negpy_core
import numpy as np
import pytest
from core_pin import digest, linux_pin

from negpy.infrastructure.loaders import source_encoding as se

_W = se._WORKING_COLORANTS_D50[1].astype(np.float32)


def _apply_curves_numpy(img, curves):
    out = np.empty(img.shape[:2] + (3,), dtype=np.float32)
    if img.dtype in (np.uint8, np.uint16):
        for ch in range(3):
            out[..., ch] = curves[ch][img[..., ch]]
        return out
    grid = np.linspace(0.0, 1.0, curves[0].shape[0], dtype=np.float32)
    top = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    flat = np.clip(img.astype(np.float32) / np.float32(top), 0.0, 1.0)
    for ch in range(3):
        out[..., ch] = np.interp(flat[..., ch], grid, curves[ch])
    return out


def _into_gamut_numpy(rgb):
    if rgb.min() >= 0.0 and rgb.max() <= 1.0:
        return rgb
    y = np.clip(rgb @ _W, 0.0, 1.0)[..., None]
    lo = rgb.min(axis=-1, keepdims=True)
    hi = rgb.max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.minimum(np.where(lo < 0.0, y / (y - lo), 1.0), np.where(hi > 1.0, (1.0 - y) / (hi - y), 1.0))
    t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
    return np.clip(y + t * (rgb - y), 0.0, 1.0)


def _curves(samples):
    x = np.linspace(0.0, 1.0, samples)
    srgb = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    flat = np.clip(x * 3.0 - 1.0, 0.0, 1.0)
    return tuple(c.astype(np.float32) for c in (srgb, x**1.8, flat))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [3, 4])
def test_integer_curves_match_numpy(dtype, channels):
    top = np.iinfo(dtype).max
    img = np.random.default_rng(1).integers(0, top, (37, 53, channels), endpoint=True, dtype=dtype)
    curves = _curves(top + 1)
    np.testing.assert_array_equal(se._apply_curves(img, curves), _apply_curves_numpy(img, curves))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("samples", [2, 256, 65536])
def test_float_curves_match_numpy(dtype, samples):
    rng = np.random.default_rng(samples)
    if dtype == np.int32:
        img = rng.integers(-5, np.iinfo(dtype).max, (41, 29, 3), dtype=dtype)
    else:
        img = rng.uniform(-0.2, 1.2, (41, 29, 3)).astype(dtype)
        grid = np.linspace(0.0, 1.0, samples, dtype=np.float32)
        img[0, :3] = grid[[0, -1, samples // 2]]
        img[1, :3] = np.nextafter(grid[[1, -1, samples // 2]], np.float32(0.5))
    curves = _curves(samples)
    np.testing.assert_array_equal(se._apply_curves(img, curves), _apply_curves_numpy(img, curves))


def _rgb(seed, lo, hi):
    rgb = np.random.default_rng(seed).uniform(lo, hi, (43, 31, 3)).astype(np.float32)
    rgb[0, 0] = (1.2, 0.3, -0.2)
    rgb[0, 1] = (np.inf, 0.5, 0.5)
    rgb[0, 2] = (np.nan, 0.5, 0.5)
    rgb[0, 3] = (-np.inf, 0.5, 2.0)
    rgb[0, 4] = (0.0, 0.0, -1e-30)
    return rgb


@pytest.mark.parametrize("lo,hi", [(-0.3, 1.3), (-2.0, 0.5), (0.5, 4.0)])
def test_into_gamut_matches_numpy(lo, hi):
    rgb = _rgb(int(hi * 10), lo, hi)
    with np.errstate(invalid="ignore"):
        np.testing.assert_array_equal(se._into_gamut(rgb), _into_gamut_numpy(rgb))


def test_one_out_of_range_value_moves_every_pixel_through_the_pass():
    rgb = np.random.default_rng(2).random((17, 19, 3), dtype=np.float32)
    rgb[5, 5, 1] = 1.0001
    np.testing.assert_array_equal(se._into_gamut(rgb), _into_gamut_numpy(rgb))
    nan = rgb.copy()
    nan[3, 3, 0] = np.nan
    np.testing.assert_array_equal(se._into_gamut(nan), _into_gamut_numpy(nan))


@linux_pin
def test_positive_decode_pin():
    prophoto = se._to_working(np.array([[0.7977, 0.1352, 0.0313], [0.2880, 0.7119, 0.0001], [0.0, 0.0, 0.8249]]).T)
    rng = np.random.default_rng(9)
    u16 = rng.integers(0, 65535, (23, 31, 3), endpoint=True, dtype=np.uint16)
    u8 = rng.integers(0, 255, (23, 31, 3), endpoint=True, dtype=np.uint8)
    f32 = rng.uniform(-0.1, 1.1, (23, 31, 3)).astype(np.float32)
    w = _W

    def decode(img, curves):
        # apply_decode without its BLAS luminance: BLAS sums in an order that depends on the CPU.
        rgb = negpy_core.matmul_3x3(np.ascontiguousarray(se._apply_curves(img, curves)), np.ascontiguousarray(prophoto))
        y = rgb[..., 0] * w[0] + rgb[..., 1] * w[1] + rgb[..., 2] * w[2]
        return negpy_core.into_gamut(rgb, np.ascontiguousarray(y))

    got = (decode(u16, _curves(65536)), decode(u8, _curves(256)), decode(f32, _curves(65536)))
    assert digest(got) == "6a88e9528457a0eb6e69ceb195bb2d0725627f0d07ef5132d8f305e536e7d877"
