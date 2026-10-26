"""The normalization planes and metrics: NumPy and cv2 oracles, and a hash pin for the clip fractions."""

import cv2
import numpy as np
import pytest
from core_oracle import cv2_oracle
from core_pin import digest, linux_pin

import negpy_core
from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R
from negpy.features.exposure import normalization as N


def _rgb(seed, shape=(301, 457)):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, (*shape, 3)).astype(np.float32)


@linux_pin
def test_clip_fractions_are_pinned():
    img = _rgb(1)
    img[img > 0.97] = 1.0
    rois = [(None, 0.0), (None, 0.1), ((13, 250, 7, 401), 0.0), ((13, 250, 7, 401), 0.05), ((0, 500, 0, 900), 0.0)]
    assert (
        digest(*(N.measure_clip_fractions(img, roi, buffer) for roi, buffer in rois))
        == "1c64edb8eb44aacea60cc6233b28a4bf29f6428ee48610892d85ae1c868f0b1f"
    )


def _lum32(val):
    return LUMA_R * val[:, :, 0] + LUMA_G * val[:, :, 1] + LUMA_B * val[:, :, 2]


@cv2_oracle
def test_luma_median_matches_numpy():
    grid = -3.0 * _rgb(3)
    w = np.array([LUMA_R, LUMA_G, LUMA_B], dtype=np.float64)
    assert negpy_core.luma_median(grid) == float(np.median(grid.astype(np.float64).reshape(-1, 3) @ w))


@cv2_oracle
@pytest.mark.parametrize("sigma", [3.0, 12.4])
def test_contrast_mask_tail_matches_numpy(sigma):
    val = _rgb(4)
    blurred = cv2.GaussianBlur(
        np.ascontiguousarray(_lum32(val), dtype=np.float32), (0, 0), sigma, sigmaY=0, borderType=cv2.BORDER_REPLICATE
    )
    centre = float(blurred.mean())
    plane, got = negpy_core.contrast_mask_tail(val, sigma)
    assert got == np.float32(centre)
    np.testing.assert_array_equal(plane, blurred - centre)


@cv2_oracle
def test_diffusion_tail_matches_numpy():
    unmixed = -2.5 * _rgb(5)
    light = np.ascontiguousarray(10.0**unmixed, dtype=np.float32)
    want = np.log10(cv2.GaussianBlur(light, (0, 0), 6.0, sigmaY=0, borderType=cv2.BORDER_REPLICATE))
    np.testing.assert_allclose(negpy_core.diffusion_tail(unmixed, 6.0), want, rtol=0, atol=1e-6)
