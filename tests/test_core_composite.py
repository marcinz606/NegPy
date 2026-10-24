"""The HDR, RGB Scan and stitch helpers against the NumPy code they replace."""

import numpy as np
import pytest

import negpy_core
from negpy.features.rgbscan import logic as rgbscan
from negpy.features.stitch import logic as stitch


def _img(seed, shape=(123, 211, 3), dtype=np.float32):
    return np.random.default_rng(seed).uniform(-0.1, 1.3, shape).astype(dtype)


def test_mean_channels_matches_numpy():
    img = _img(1)
    np.testing.assert_array_equal(negpy_core.mean_channels(img), img.mean(axis=2))


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.uint8])
def test_pick_channels_matches_numpy(dtype):
    r, g, b = (np.random.default_rng(s).integers(0, 250, (31, 47, 3)).astype(dtype) for s in (2, 3, 4))
    want = np.empty_like(r)
    want[..., 0], want[..., 1], want[..., 2] = r[..., 0], g[..., 1], b[..., 2]
    np.testing.assert_array_equal(negpy_core.pick_channels(r, g, b), want)
    np.testing.assert_array_equal(rgbscan.assemble_rgb(r, g, b, align=False), want)


def test_extract_channel_matches_numpy():
    img = _img(5, (20, 30, 4))
    np.testing.assert_array_equal(negpy_core.extract_channel(img, 2), img[..., 2])


def test_proxy_stretch_matches_numpy():
    gray = _img(6, (600, 900))
    lo, hi = np.percentile(gray, (1.0, 99.0))
    want = (np.clip((gray - lo) / max(hi - lo, 1e-9), 0.0, 1.0) ** (1.0 / 2.2) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(negpy_core.proxy_stretch(gray), want)


@pytest.mark.parametrize("take_max", [True, False])
def test_ir_blend_matches_numpy(take_max):
    rng = np.random.default_rng(7)
    acc = np.full((40, 50), -1.0, np.float32)
    want = acc.copy()
    for _ in range(3):
        warped = rng.uniform(0, 1, (40, 50)).astype(np.float32)
        sel = rng.uniform(0, 1, (40, 50)) > 0.5
        negpy_core.ir_blend_into(acc, warped, sel, take_max)
        want[sel] = np.maximum(want[sel], warped[sel]) if take_max else warped[sel]
    negpy_core.ir_finish(acc)
    want[want < 0.0] = 1.0
    np.testing.assert_array_equal(acc, want)


def test_mask_overlap_scale_and_clip_match_numpy():
    rng = np.random.default_rng(8)
    a, b = rng.uniform(0, 1, (2, 64, 80)) > 0.4
    assert negpy_core.mask_overlap(a, b) == int((a & b).sum())
    img = _img(9)
    want = img * np.array([0.9, 1.1, 1.3], dtype=np.float32)
    negpy_core.scale_channels(img, [0.9, 1.1, 1.3])
    np.testing.assert_array_equal(img, want)
    want = np.clip(img, 0.0, 1.0)
    negpy_core.clip_unit(img)
    np.testing.assert_array_equal(img, want)


def test_channel_means_match_numpy():
    rng = np.random.default_rng(10)
    mosaic = rng.uniform(0, 4000, (300, 401)).astype(np.float32)
    colors = np.tile(np.array([[0, 1], [3, 2]], np.uint8), (150, 201))[:, :401]

    def mean_of(*idx):
        mask = np.isin(colors, idx)
        return float(mosaic[mask].mean()) - 64.0 if mask.any() else 0.0

    assert rgbscan._channel_means(mosaic, colors, 64.0) == (mean_of(0), mean_of(1, 3), mean_of(2))


def test_build_proxy_keeps_its_contract():
    proxy, scale = stitch.build_proxy(_img(11, (2000, 3000, 3)))
    assert proxy.dtype == np.uint8 and max(proxy.shape) == stitch.PROXY_MAX_EDGE and scale < 1.0
