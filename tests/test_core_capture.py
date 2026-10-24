"""The preview and capture helpers: NumPy oracles for one-line rules, hash pins for the rest."""

import numpy as np
import pytest
from core_pin import digest, linux_pin

import negpy_core
from negpy.desktop.view.widgets.prescan_dialog import _preview_u8
from negpy.desktop.view.widgets.scan_preview_common import preview_positive
from negpy.infrastructure.scanners.params import FilmType
from negpy.services.capture.focus_meter import MIN_MEAN_LEVEL, sharpness
from negpy.services.scanning.writer import _to_grey


def _scan(seed, shape=(151, 203, 3), dtype=np.uint16):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.05, 0.9, shape) * np.iinfo(dtype).max).astype(dtype)


@pytest.mark.parametrize(
    "shape, pin",
    [
        ((151, 203, 3), "a79edc8a71b01c096e649f3138296d6e5da370dd753fb98609740559fd353973"),
        ((40, 50), "ff93f26f21d9407ddd96f63bc16c5ddadbc8afce54301caec8659bbcf837065d"),
        ((30, 20, 4), "64e4ff502b446f464c1482f3a0efebfe7a922309d01782ad73d5dd5f74a2b4ee"),
    ],
)
@linux_pin
def test_preview_stretches_are_pinned(shape, pin):
    rgb = _scan(1, shape)
    if len(shape) == 3:
        rgb[..., -1] = 7  # a flat channel
    assert digest(preview_positive(rgb, "negative"), preview_positive(rgb, FilmType.POSITIVE.value), _preview_u8(rgb)) == pin


def test_sharpness_matches_numpy():
    img = _scan(2, (120, 160), np.uint8).astype(np.float32)
    mean = float(img.mean())
    lap = 4.0 * img[1:-1, 1:-1] - img[:-2, 1:-1] - img[2:, 1:-1] - img[1:-1, :-2] - img[1:-1, 2:]
    assert sharpness(img) == float(np.mean(lap * lap)) / (mean * mean)
    assert sharpness(np.full((10, 10), MIN_MEAN_LEVEL / 2, np.float32)) == 0.0


def test_level_stats_match_numpy():
    for patch in (_scan(3, (50, 70), np.uint16), _scan(4, (33, 41), np.uint16).astype(np.float32) / 65535):
        level = 60000 if patch.dtype == np.uint16 else 0.9
        frac, p = negpy_core.patch_level_stats(np.ascontiguousarray(patch), float(level), 99.9)
        assert frac == float(np.mean(patch >= level))
        assert p == float(np.percentile(patch, 99.9))
    img = _scan(5, (40, 60, 4)).astype(np.float32) / 65535
    assert negpy_core.any_channel_at_least(img, 0.85) == float(np.mean(np.any(img[:, :, :3] >= 0.85, axis=2)))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_to_grey_matches_numpy(dtype):
    rgb = _scan(6, (31, 47, 3), dtype)
    np.testing.assert_array_equal(_to_grey(rgb), ((np.sum(rgb, axis=-1, dtype=np.uint32) + 1) // 3).astype(dtype))


def test_level_stats_of_float64_match_numpy():
    patch = np.random.default_rng(7).uniform(0, 65535, (37, 29))
    frac, p = negpy_core.patch_level_stats(patch, 60000.0, 99.9)
    assert (frac, p) == (float(np.mean(patch >= 60000.0)), float(np.percentile(patch, 99.9)))


@linux_pin
def test_meter_picture_output_is_pinned():
    rng = np.random.default_rng(8)
    linear = np.clip(rng.uniform(0.2, 1.05, (90, 120, 3)).astype(np.float32), 0, 1)
    clipped, blown, thin = negpy_core.meter_picture(linear, (10, 80, 15, 110), float(np.float32(0.998)), 0.02)
    assert digest(clipped, blown, thin) == "7d4c9464db51330308763ecdfda9dfc363885f9093c038f4e5d358c0258b4ab2"


@pytest.mark.parametrize(
    "plateau, pin",
    [
        (None, "dac4aa1cceb2176ffaab1b38d64d8a0f26d0e6c1be64934e6dfe0fbb5607efb0"),
        (15500, "acf38442574a9df4812abaad9b2eda7105008c4f88a5abe3431ef2504ff9d4e5"),
        (15790, "c28d47a503fd602c1140fbff84a67118b867804585cb0f8ea87f70ffbed30bd3"),
    ],
)
@linux_pin
def test_raw_clip_fractions_are_pinned(plateau, pin):
    from negpy.infrastructure.capture import raw_demosaic as R

    rng = np.random.default_rng(9)
    img = rng.integers(14000, 15800, (60, 80)).astype(np.uint16)
    if plateau is not None:
        img[rng.uniform(0, 1, img.shape) < 0.3] = plateau
    colors = np.tile(np.array([[0, 1], [3, 2]], np.uint8), (30, 40))
    got = negpy_core.raw_clip_fractions(
        img, colors, [1, 3], 15000, 16383, R._PLATEAU_WINDOW, R._MIN_PLATEAU_SITES, R._PLATEAU_NEIGHBOR_BINS, R._PLATEAU_DENSITY_RATIO
    )
    assert digest(got) == pin


@linux_pin
def test_column_means_match_numpy_and_ring_medians_and_dmax_are_pinned():
    rng = np.random.default_rng(10)
    a = rng.uniform(0, 1, (700, 300)).astype(np.float32)
    peak = float(a.max())
    np.testing.assert_array_equal(negpy_core.column_means(a, 2, peak), (a[::2] / peak).mean(axis=0))
    arr = rng.uniform(0, 0.2, (256, 190)).astype(np.float32)
    px = rng.integers(0, 256, (80, 60, 3)).astype(np.uint8)
    assert (
        digest(negpy_core.ring_inner_medians(arr, 10), negpy_core.print_dmax(px, 0.01))
        == "6bee0fb07082c762e491db108839174266925af6fc3eafff3cc3b65c9f5e945e"
    )


def _srgb_numpy(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4).astype(np.float32)


def test_srgb_to_linear_stays_within_an_ulp():
    from negpy.kernel.image.logic import srgb_to_linear

    img = np.linspace(0, 1, 70001, dtype=np.float32)
    base = ((img + np.float32(0.055)) / np.float32(1.055)).astype(np.float64)
    exact = np.where(img <= 0.04045, img / np.float32(12.92), (base**2.4).astype(np.float32)).astype(np.float32)
    got = srgb_to_linear(img)
    assert np.max(np.abs(got.view(np.int32) - exact.view(np.int32))) <= 1
    assert np.max(np.abs(got.view(np.int32) - _srgb_numpy(img).view(np.int32))) <= 4


@linux_pin
def test_thumbnail_log_stretch_is_pinned():
    rng = np.random.default_rng(12)
    linear = _srgb_numpy((rng.integers(0, 256, (90, 130, 3)) / 255).astype(np.float32))
    linear[..., 2] *= 0.01  # a near-empty channel borrows the strongest
    assert digest(negpy_core.log_stretch_u8(linear)) == "926248e2592ff86509f1988ba9aaf606bbc24fe6143d29d55bab0941c681d062"
